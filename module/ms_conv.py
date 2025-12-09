from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


class dvs_pooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)

        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        x = x + identity
        return x, hook

# MoE Implementation

class MS_MLP_Expert(nn.Module):
    """Single expert network - same structure as MS_MLP_Conv"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        tau=2.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend="cupy")
        
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend="cupy")
        
        self.c_hidden = hidden_features
        self.c_output = out_features

    def reset(self):
        """Reset LIF neuron states - critical for independent processing of samples"""
        for m in self.modules():
            if hasattr(m, "reset"):
                m.reset()

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        
        if self.res:
            x = identity + x
            identity = x
        
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        x = x + identity
        print(f"[Expert called] tokens = {x.shape[1]}")
        return x

# Spike router input (T,B,C,H,W), output: top_k_weights, top_k_indices, router_logits(used for load balancing loss)
class SpikeRouter(nn.Module):
    """Spike-based router for expert selection"""
    def __init__(
        self,
        in_features,
        num_experts,
        top_k=2,
        spike_mode="lif",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network
        self.router_conv = nn.Conv2d(in_features, num_experts, kernel_size=1, stride=1)
        self.router_bn = nn.BatchNorm2d(num_experts)
        
        if spike_mode == "lif":
            self.router_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.router_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (T, B, C, H, W)
        
        Returns:
            routing_weights: Softmax probabilities (T*B, num_experts)
            selected_experts: Top-k expert indices (T*B, top_k)
            router_logits: Raw router outputs (T*B, num_experts)
        """
        T, B, C, H, W = x.shape
        
        # Generate routing logits through spike network
        router_out = self.router_lif(x)
        router_out = self.router_conv(router_out.flatten(0, 1))
        router_out = self.router_bn(router_out).reshape(T, B, self.num_experts, H, W)
        
        # Global average pooling over spatial dimensions for routing decision
        # Shape: (T, B, num_experts)
        router_logits = router_out.mean(dim=[-2, -1])
        
        # Flatten temporal and batch dimensions
        # Shape: (T*B, num_experts)
        router_logits = router_logits.reshape(T * B, self.num_experts)
        
        # Apply softmax to get routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Renormalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices, router_logits


class MS_MoE_Conv(nn.Module):
    """
    Sparse Mixture of Experts for SNN - CORRECTED VERSION
    
    Key design decisions:
    1. Token-level routing: Each (t,b) position independently routed
    2. NO temporal grouping: Experts see only their assigned tokens
    3. Preserves baseline MLP behavior when top_k=1, num_experts=1
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        num_experts=8,
        top_k=2,
        spike_mode="lif",
        layer=0,
        aux_loss_weight=0.01,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer = layer
        self.aux_loss_weight = aux_loss_weight
        
        # Create router
        self.router = SpikeRouter(
            in_features=in_features,
            num_experts=num_experts,
            top_k=top_k,
            spike_mode=spike_mode,
        )
        tau_min=1.5
        tau_max=4.0
        # Create expert networks
        self.experts = nn.ModuleList([
            MS_MLP_Expert(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        spike_mode=spike_mode,
        tau=tau_min + (tau_max - tau_min) * i / (num_experts - 1) if num_experts > 1 else tau_min,
    )
    for i in range(num_experts)
])
        
        self.c_output = out_features
        self.load_balancing_loss = None

    def compute_load_balancing_loss(self, router_logits, selected_experts):
        """
        Compute auxiliary load balancing loss to encourage uniform expert usage
        
        Args:
            router_logits: Raw router outputs (T*B, num_experts)
            selected_experts: Selected expert indices (T*B, top_k)
        """
        # Compute the fraction of tokens routed to each expert
        expert_counts = torch.bincount(
            selected_experts.view(-1),
            minlength=self.num_experts
        ).float()
        
        # Normalize to get fraction
        expert_fraction = expert_counts / selected_experts.numel()
        
        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        router_prob_per_expert = router_probs.mean(dim=0)
        
        # Load balancing loss: encourages uniform distribution
        load_balancing_loss = self.num_experts * (expert_fraction * router_prob_per_expert).sum()
        
        return load_balancing_loss

    def forward(self, x, hook=None):
        """
        Token-level MoE routing that preserves baseline MLP behavior.
        
        Each (t, b) token is independently routed to top_k experts.
        Experts process tokens WITHOUT seeing unassigned temporal context.
        
        This matches baseline behavior: each token through MLP independently.
        
        Args:
            x: Input tensor of shape (T, B, C, H, W)
            hook: Optional dictionary for storing intermediate activations
        
        Returns:
            output: Mixed expert outputs (T, B, C, H, W)
            hook: Updated hook dictionary
        """

        T, B, C, H, W = x.shape
        identity = x
        
        # Reset all experts at the start of forward (once per forward call)
        for expert in self.experts:
            expert.reset()
        
        # Get routing decisions
        top_k_weights, top_k_indices, router_logits = self.router(x)
        # top_k_weights: (T*B, top_k)
        # top_k_indices: (T*B, top_k)
        
        # Detach routing weights from expert gradients (critical for MoE sparsity)
        top_k_weights = top_k_weights.detach()
        top_k_indices = top_k_indices.detach()
        
        # Store routing information in hook if provided
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_routing_weights"] = top_k_weights.detach()
            hook[self._get_name() + str(self.layer) + "_routing_indices"] = top_k_indices.detach()
        
        # Compute load balancing loss (before detach, needs gradients)
        self.load_balancing_loss = self.compute_load_balancing_loss(router_logits, top_k_indices)
        
        # Token-level expert processing
        # Each expert processes only tokens assigned to it, in batches
        
        output = torch.zeros_like(x)
        
        # Process each expert with its assigned tokens
        for expert_idx in range(self.num_experts):
            # Find all (T*B, top_k) positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx)
            
            if not expert_mask.any():
                continue  # Skip if no tokens assigned
            
            # Get indices where this expert is selected
            tb_indices, k_indices = torch.where(expert_mask)
            
            if len(tb_indices) == 0:
                continue
            
            # Convert flat TB indices to (T, B) coordinates
            t_indices = tb_indices // B
            b_indices = tb_indices % B
            
            # Get weights for this expert's assignments
            expert_weights = top_k_weights[tb_indices, k_indices]  # (num_assignments,)
            
            # CRITICAL FIX: Process each token independently
            # Gather only the assigned tokens (no temporal grouping)
            num_tokens = len(t_indices)
            
            # Create batch of tokens for this expert: (1, num_tokens, C, H, W)
            # Using T=1 because each token is processed independently
            expert_input = torch.zeros(1, num_tokens, C, H, W, device=x.device, dtype=x.dtype)
            
            for i, (t_idx, b_idx) in enumerate(zip(t_indices, b_indices)):
                expert_input[0, i] = x[t_idx, b_idx]
            
            # Process through expert - each token independent
            expert_output = self.experts[expert_idx](expert_input)  # (1, num_tokens, C, H, W)
            
            # Distribute weighted outputs back to their positions
            for i, (t_idx, b_idx, weight) in enumerate(zip(t_indices, b_indices, expert_weights)):
                output[t_idx, b_idx] += weight * expert_output[0, i]
        
        # Add residual connection (matches baseline MLP)
        output = output + identity
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_moe_output"] = output.detach()
        
        return output, hook

# Original Classes (unchanged)

class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mode="direct_xor",
        dvs=False,
        layer=0,
        attention_mode="T_STAtten",
        chunk_size=2,
        spike_mode="lif"
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.attention_mode = attention_mode
        if dvs:
            self.pool = dvs_pooling()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        self.talking_heads_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)
        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.mode = mode
        self.layer = layer
        self.chunk_size = chunk_size

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        head_dim = C // self.num_heads
        identity = x
        N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()
        if self.dvs:
            x_pool = self.pool(x)

        x_for_qkv = x.flatten(0, 1)

        # Q
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        if self.dvs:
            q_conv_out = self.pool(q_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (q_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())

        # K
        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (k_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())

        # V
        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (v_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())
        # Shape: (T B head N C//head)

        ###### Attention #####
        if self.attention_mode == "STAtten":
            if self.dvs:
                scaling_factor = 1 / (H*H*self.chunk_size)
            else:
                scaling_factor = 1 / H

            # Vectorized Attention
            num_chunks = T // self.chunk_size
            # Reshape q, k, v to process all chunks at once: (num_chunks, B, num_heads, chunk_size, N, head_dim)
            q_chunks = q.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            k_chunks = k.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            v_chunks = v.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)

            # Merge chunk_size and N dimensions: (num_chunks, B, num_heads, chunk_size * N, head_dim)
            q_chunks = q_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            k_chunks = k_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            v_chunks = v_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)

            # Compute attention for all chunks simultaneously
            attn = torch.matmul(k_chunks.transpose(-2, -1),
                                v_chunks) * scaling_factor  # (num_chunks, B, num_heads, head_dim, head_dim)
            out = torch.matmul(q_chunks, attn)  # (num_chunks, B, num_heads, chunk_size * N, head_dim)

            # Reshape back to separate temporal and spatial dimensions
            out = out.reshape(num_chunks, B, self.num_heads, self.chunk_size, N, head_dim).permute(0, 3, 1, 2, 4, 5)
            # Flatten chunks back to T: (T, B, num_heads, N, head_dim)
            output = out.reshape(T, B, self.num_heads, N, head_dim)

            x = output.transpose(4,3).reshape(T, B, C, N).contiguous() # (T, B, head, C//h, N)
            x = self.attn_lif(x).reshape(T, B, C, H, W)
            if self.dvs:
                x = x.mul(x_pool)
                x = x + x_pool

            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_after_qkv"] = x

            x = (
                self.proj_bn(self.proj_conv(x.flatten(0, 1)))
                .reshape(T, B, C, H, W)
                .contiguous()
            )


        """Spike-driven Transformer"""
        if self.attention_mode == "SDT":
            kv = k.mul(v)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
            if self.dvs:
                kv = self.pool(kv)
            kv = kv.sum(dim=-2, keepdim=True)
            kv = self.talking_heads_lif(kv)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
            x = q.mul(kv)
            if self.dvs:
                x = self.pool(x)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

            x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
            x = (
                self.proj_bn(self.proj_conv(x.flatten(0, 1)))
                .reshape(T, B, C, H, W)
                .contiguous()
            )

        assert self.attention_mode in ["STAtten", "SDT"], \
            f"Unsupported attention_mode: {self.attention_mode}"

        x = x + identity
        return x, v, hook


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        attention_mode="STAtten",
        chunk_size=2,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            mode=attn_mode,
            dvs=dvs,
            layer=layer,
            attention_mode=attention_mode,
            chunk_size=chunk_size
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook


# MoE-enabled Block

class MS_Block_Conv_MoE(nn.Module):
    """Block with optional MoE support"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        attention_mode="STAtten",
        chunk_size=2,
        # MoE specific parameters
        use_moe=True,
        num_experts=8,
        expert_top_k=2,
        aux_loss_weight=0.01,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            mode=attn_mode,
            dvs=dvs,
            layer=layer,
            attention_mode=attention_mode,
            chunk_size=chunk_size
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.use_moe = use_moe
        self.aux_loss_weight = aux_loss_weight
        
        if use_moe:
            self.mlp = MS_MoE_Conv(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                num_experts=num_experts,
                top_k=expert_top_k,
                spike_mode=spike_mode,
                layer=layer,
                aux_loss_weight=aux_loss_weight,
            )
        else:
            self.mlp = MS_MLP_Conv(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                spike_mode=spike_mode,
                layer=layer,
            )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        # Note: drop_path defined but not used in original MS_Block_Conv
        # MLP already contains internal residual connections
        return x, attn, hook
    
    def get_aux_loss(self):
        """Get auxiliary load balancing loss for training"""
        if self.use_moe and hasattr(self.mlp, 'load_balancing_loss'):
            if self.mlp.load_balancing_loss is not None:
                return self.aux_loss_weight * self.mlp.load_balancing_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)
