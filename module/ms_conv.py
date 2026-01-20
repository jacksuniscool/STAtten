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


# MoE Implementation

class MS_MLP_Expert(nn.Module):
    """Single expert network - Same structure as the original MLP"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        tau=2.0
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
        """Reset LIF neuron states"""
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
        
        return x


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
        router_logits = router_out.mean(dim=[-2, -1])
        
        # Flatten temporal and batch dimensions
        router_logits = router_logits.reshape(T * B, self.num_experts)
        
        # Apply softmax to get routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Renormalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices, router_logits


class MS_MoE_Conv(nn.Module):

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

        tau_min = 1.9
        tau_max = 2.1
        
        # Create router
        self.router = SpikeRouter(
            in_features=in_features,
            num_experts=num_experts,
            top_k=top_k,
            spike_mode=spike_mode,
        )
        
        # Create expert networks with varying tau values
        self.experts = nn.ModuleList([
            MS_MLP_Expert(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                spike_mode=spike_mode,
                tau=tau_min + i * (tau_max - tau_min) / (num_experts - 1) if num_experts > 1 else tau_min,
            )
            for i in range(num_experts) 
        ])
        
        # Removed unused self.c_output
        self.load_balancing_loss = None

    def compute_load_balancing_loss(self, router_logits, selected_experts):
        expert_counts = torch.bincount(
            selected_experts.view(-1),
            minlength=self.num_experts
        ).float()
        
        expert_fraction = expert_counts / selected_experts.numel()
        router_probs = F.softmax(router_logits, dim=-1)
        router_prob_per_expert = router_probs.mean(dim=0)
        
        load_balancing_loss = self.num_experts * (expert_fraction * router_prob_per_expert).sum()
        return load_balancing_loss

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        
        # 1. Reset all experts
        for expert in self.experts:
            expert.reset()
        
        # 2. Get routing decisions
        top_k_weights, top_k_indices, router_logits = self.router(x)
        
        # Only detach indices. Keep weights connected for gradient flow.
        top_k_indices = top_k_indices.detach()
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_routing_weights"] = top_k_weights.detach()
            hook[self._get_name() + str(self.layer) + "_routing_indices"] = top_k_indices.detach()
        
        self.load_balancing_loss = self.compute_load_balancing_loss(router_logits, top_k_indices)
        
        # 3. Vectorized Expert Processing
        x_flat = x.flatten(0, 1)
        output_flat = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            indices_for_expert, kth_expert_indices = torch.where(top_k_indices == expert_idx)
            
            if indices_for_expert.numel() == 0:
                continue
            
            expert_input_flat = x_flat[indices_for_expert]
            expert_input = expert_input_flat.unsqueeze(0)
            
            expert_output = self.experts[expert_idx](expert_input)
            expert_output = expert_output.squeeze(0)
            
            weights = top_k_weights[indices_for_expert, kth_expert_indices]
            weights = weights.view(-1, 1, 1, 1)
            
            output_flat.index_add_(0, indices_for_expert, expert_output * weights)
            
        # 4. Reshape and Residual
        output = output_flat.view(T, B, C, H, W)
        output = output + identity
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_moe_output"] = output.detach()
        
        return output, hook


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
            
        # Removed unused self.scale
        
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        # Removed duplicate self.attn_lif definition here

        # Conditional definition for SDT mode specific layers
        if self.attention_mode == "SDT":
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

        ###### Attention #####
        if self.attention_mode == "STAtten":
            if self.dvs:
                scaling_factor = 1 / (H*H*self.chunk_size)
            else:
                scaling_factor = 1 / H

            # Vectorized Attention
            num_chunks = T // self.chunk_size
            q_chunks = q.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            k_chunks = k.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            v_chunks = v.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)

            q_chunks = q_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            k_chunks = k_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            v_chunks = v_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)

            attn = torch.matmul(k_chunks.transpose(-2, -1), v_chunks) * scaling_factor 
            out = torch.matmul(q_chunks, attn) 

            out = out.reshape(num_chunks, B, self.num_heads, self.chunk_size, N, head_dim).permute(0, 3, 1, 2, 4, 5)
            output = out.reshape(T, B, self.num_heads, N, head_dim)

            x = output.transpose(4,3).reshape(T, B, C, N).contiguous() 
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
    """
    Unified Transformer Block (Forced MoE Mode)
    Refactored to remove unused parameters logic while keeping signature compatibility.
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False, # Kept for signature compatibility
        qk_scale=None,  # Kept for signature compatibility
        drop=0.0,       # Kept for signature compatibility
        attn_drop=0.0,  # Kept for signature compatibility
        drop_path=0.0,  # Kept for signature compatibility
        sr_ratio=1,     # Kept for signature compatibility
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        attention_mode="STAtten",
        chunk_size=2,
        # MoE parameters
        num_experts=8,
        expert_top_k=2,
        aux_loss_weight=0.01,
        use_moe=True, 
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
        
        # Removed unused self.drop_path definition
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.num_experts = num_experts
        self.aux_loss_weight = aux_loss_weight
        
        # Forced MoE
        self.mlp = MS_MoE_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            num_experts=num_experts,
            top_k=expert_top_k,
            spike_mode=spike_mode,
            layer=layer,
            aux_loss_weight=aux_loss_weight,
        )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook
    
    def get_aux_loss(self):
        if hasattr(self.mlp, 'load_balancing_loss'):
            if self.mlp.load_balancing_loss is not None:
                return self.aux_loss_weight * self.mlp.load_balancing_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)
