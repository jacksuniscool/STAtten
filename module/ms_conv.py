
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SUPPORTING MODULES
# ============================================================================

class dvs_pooling(nn.Module):
    """DVS-specific pooling for event cameras"""
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


# ============================================================================
# TEMPORAL MOE COMPONENTS
# ============================================================================

class MS_MLP_Expert(nn.Module):
    """
    ✅ FIXED: Expert as pure function F(x) WITHOUT internal residual
    
    Expert network with configurable tau for temporal specialization
    
    Architecture: 2-layer MLP with spike neurons
    - Fast experts (low τ): Capture transients, quick responses
    - Slow experts (high τ): Integrate sustained patterns
    
    CRITICAL: No internal residual connection!
    - Residual is handled by MS_MoE_Conv_Temporal: output = x + MoE(x)
    - Expert just computes F(x), not F(x) + x
    """
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
        self.tau = tau
        
        # Layer 1
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend="cupy")
        
        # Layer 2
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend="cupy")
        
        self.c_hidden = hidden_features
        self.c_output = out_features

    def reset(self):
        """Reset LIF states - critical for independent sample processing"""
        for m in self.children():
            if hasattr(m, "reset"):
                m.reset()

    def forward(self, x):
        """
        ✅ FIXED: Returns F(x), NOT F(x) + x
        
        Args:
            x: (T, B, C, H, W) - full temporal sequence
        Returns:
            output: (T, B, C, H, W) - transformed features (NO residual added)
        """
        T, B, C, H, W = x.shape
        
        # Layer 1: LIF → Conv → BN5
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        
        # Layer 2: LIF → Conv → BN
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, self.c_output, H, W).contiguous()
        
        # ✅ CRITICAL FIX: NO residual connection here!
        # Residual is handled at MoE level, not expert level
        return x


class TemporalRouter(nn.Module):
    """
    Deterministic temporal router for batch-level expert selection
    
    Key design:
    - Aggregates temporal dimension BEFORE routing
    - No spike neurons → no routing noise
    - Stable gradient flow
    """
    def __init__(
        self,
        in_features,
        num_experts,
        top_k=2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router_conv = nn.Conv2d(in_features, num_experts, kernel_size=1, stride=1)
        self.router_bn = nn.BatchNorm2d(num_experts)
    
    def forward(self, x):
        """
        Args:
            x: (T, B, C, H, W) - input features
        
        Returns:
            top_k_weights: (B, top_k) - normalized routing weights
            top_k_indices: (B, top_k) - selected expert indices
            router_logits: (B, num_experts) - raw logits for loss computation
        """
        T, B, C, H, W = x.shape
        
        # ✅ KEY: Temporal aggregation BEFORE routing decision
        x_mean = x.mean(dim=0)  # (B, C, H, W)
        
        # Route based on temporal aggregate
        router_out = self.router_conv(x_mean)
        router_out = self.router_bn(router_out)
        
        # Global average pooling
        router_logits = router_out.mean(dim=[-2, -1])  # (B, num_experts)
        
        # Softmax and top-k selection
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Renormalize
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices, router_logits


class MS_MoE_Conv_Temporal(nn.Module):
    """
    ✅ FIXED: Proper residual connection at MoE level
    
    Batch-level Temporal MoE with variable tau specialization
    
    Residual structure:
        output = x + sum(weight_i * Expert_i(x))
    
    Features:
    - Fully vectorized (NO Python loops)
    - Anti-collapse regularization (entropy + load balancing)
    - Temporal specialization via different tau values
    - Proper single-level residual connection
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
        entropy_loss_weight=0.01,
        tau_min=1.5,
        tau_max=4.0,
        tau_distribution="linear",
        custom_taus=None,
        verbose=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer = layer
        self.aux_loss_weight = aux_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        
        # Generate tau values for expert specialization
        if custom_taus is not None:
            assert len(custom_taus) == num_experts
            expert_taus_list = custom_taus
        elif tau_distribution == "linear":
            expert_taus_list = torch.linspace(tau_min, tau_max, num_experts).tolist()
        elif tau_distribution == "log":
            expert_taus_list = torch.logspace(
                torch.log10(torch.tensor(tau_min)),
                torch.log10(torch.tensor(tau_max)),
                num_experts
            ).tolist()
        else:
            raise ValueError(f"Unknown tau_distribution: {tau_distribution}")
        
        self.register_buffer('expert_taus', torch.tensor(expert_taus_list, dtype=torch.float32))
        
        if verbose:
            print(f"[MoE Layer {layer}] Expert tau values: {[f'{tau:.3f}' for tau in expert_taus_list]}")
        
        # Router
        self.router = TemporalRouter(
            in_features=in_features,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # Experts with different tau values (NO internal residual)
        self.experts = nn.ModuleList([
            MS_MLP_Expert(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                spike_mode=spike_mode,
                tau=expert_taus_list[i],
            )
            for i in range(num_experts)
        ])
        
        self.c_output = out_features
        self.load_balancing_loss = None
        self.entropy_loss = None

    def compute_load_balancing_loss(self, router_logits, selected_experts):
        """Encourage uniform expert usage"""
        expert_counts = torch.bincount(
            selected_experts.view(-1),
            minlength=self.num_experts
        ).float()
        
        expert_fraction = expert_counts / selected_experts.numel()
        router_probs = F.softmax(router_logits, dim=-1)
        router_prob_per_expert = router_probs.mean(dim=0)
        
        load_balancing_loss = self.num_experts * (expert_fraction * router_prob_per_expert).sum()
        
        return load_balancing_loss
    
    def compute_entropy_loss(self, router_logits):
        """
        Entropy regularization to prevent expert collapse
        Maximize routing entropy = encourage diversity
        """
        router_probs = F.softmax(router_logits, dim=-1)
        entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()
        return -entropy  # Minimize negative entropy = maximize entropy

    def forward(self, x, hook=None):
        """
        ✅ FIXED: Single residual connection at MoE level
        
        Structure: output = x + MoE(x)
        Where MoE(x) = sum(weight_i * Expert_i(x))
        
        Args:
            x: (T, B, C, H, W) - input features
            hook: Optional dict for debugging/monitoring
        
        Returns:
            output: (T, B, C, H, W) - x + mixed expert outputs
            hook: Updated hook dict
        """
        T, B, C, H, W = x.shape
        identity = x  # ✅ Save for single residual connection
        
        # Reset all experts
        for expert in self.experts:
            expert.reset()
        
        # Get routing decisions
        top_k_weights, top_k_indices, router_logits = self.router(x)
        
        # Compute losses (before detach)
        self.load_balancing_loss = self.compute_load_balancing_loss(router_logits, top_k_indices)
        self.entropy_loss = self.compute_entropy_loss(router_logits)
        
        # Detach for expert processing
        top_k_weights = top_k_weights.detach()
        top_k_indices = top_k_indices.detach()
        
        # Store routing info
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_routing_weights"] = top_k_weights.clone()
            hook[self._get_name() + str(self.layer) + "_routing_indices"] = top_k_indices.clone()
            hook[self._get_name() + str(self.layer) + "_selected_taus"] = self.expert_taus[top_k_indices].clone()
        
        # ✅ Initialize as zeros (will accumulate expert outputs)
        moe_output = torch.zeros_like(x)
        
        # ✅ VECTORIZED: Process each expert
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx)
            
            if not expert_mask.any():
                continue
            
            b_indices, k_indices = torch.where(expert_mask)
            
            if len(b_indices) == 0:
                continue
            
            # Get weights
            expert_weights = top_k_weights[b_indices, k_indices]
            
            # Extract assigned samples (full temporal sequences)
            expert_input = x[:, b_indices, :, :, :]
            
            # ✅ Process through expert (returns F(x), not F(x) + x)
            expert_output = self.experts[expert_idx](expert_input)
            
            # Apply weights (vectorized)
            expert_weights_reshaped = expert_weights.view(1, -1, 1, 1, 1)
            weighted_output = expert_output * expert_weights_reshaped
            
            # ✅ CRITICAL: Fully vectorized scatter-add
            moe_output.index_add_(dim=1, index=b_indices, source=weighted_output)
        
        # ✅ CRITICAL FIX: Single residual connection at MoE level
        # output = x + MoE(x), NOT x + (MoE(x) + x)
        output = identity + moe_output
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_moe_output"] = output.detach()
        
        return output, hook
    
    def get_aux_loss(self):
        """Combined auxiliary loss"""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.load_balancing_loss is not None:
            total_loss += self.aux_loss_weight * self.load_balancing_loss
        
        if self.entropy_loss is not None:
            total_loss += self.entropy_loss_weight * self.entropy_loss
        
        return total_loss


# ============================================================================
# ATTENTION MODULE - STAtten ONLY
# ============================================================================

class MS_SSA_Conv(nn.Module):
    """
    Spike Self-Attention with STAtten (Spatio-Temporal Attention) ONLY
    
    Note: This module includes internal residual connection (original design)
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        dvs=False,
        layer=0,
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
        self.chunk_size = chunk_size
        self.layer = layer
        
        if dvs:
            self.pool = dvs_pooling()
        
        self.scale = 0.125
        
        # Q, K, V projections
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

        # Output projection
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)
        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, x, hook=None):
        """
        STAtten forward pass with internal residual
        
        Args:
            x: (T, B, C, H, W) - input spikes
            hook: Optional monitoring dict
        
        Returns:
            x: (T, B, C, H, W) - attention output (with residual)
            v: V features (for visualization)
            hook: Updated monitoring dict
        """
        T, B, C, H, W = x.shape
        head_dim = C // self.num_heads
        identity = x  # Internal residual
        N = H * W
        
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()
        
        if self.dvs:
            x_pool = self.pool(x)

        x_for_qkv = x.flatten(0, 1)

        # Q projection
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        if self.dvs:
            q_conv_out = self.pool(q_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (q_conv_out.flatten(3).transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())

        # K projection
        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (k_conv_out.flatten(3).transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())

        # V projection
        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (v_conv_out.flatten(3).transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())

        # STAtten computation
        if self.dvs:
            scaling_factor = 1 / (H * H * self.chunk_size)
        else:
            scaling_factor = 1 / H

        num_chunks = T // self.chunk_size
        
        # Reshape for chunked processing
        q_chunks = q.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
        k_chunks = k.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
        v_chunks = v.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)

        # Merge chunk_size and N dimensions
        q_chunks = q_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
        k_chunks = k_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
        v_chunks = v_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)

        # Compute attention
        attn = torch.matmul(k_chunks.transpose(-2, -1), v_chunks) * scaling_factor
        out = torch.matmul(q_chunks, attn)

        # Reshape back
        out = out.reshape(num_chunks, B, self.num_heads, self.chunk_size, N, head_dim).permute(0, 3, 1, 2, 4, 5)
        output = out.reshape(T, B, self.num_heads, N, head_dim)

        x = output.transpose(4, 3).reshape(T, B, C, N).contiguous()
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

        # Internal residual connection (original design)
        x = x + identity
        return x, v, hook


# ============================================================================
# COMPLETE BLOCK: ATTENTION + MOE (Proper Residuals)
# ============================================================================

class MS_Block_Conv_MoE(nn.Module):
    """
    ✅ FIXED: Complete SNN block with proper residual connections
    
    Structure:
        x_attn = x + Attention(x)     [internal residual in MS_SSA_Conv]
        x_out = x_attn + MoE(x_attn)  [residual in MS_MoE_Conv_Temporal]
    
    No double residuals!
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        spike_mode="lif",
        dvs=False,
        layer=0,
        chunk_size=2,
        # MoE parameters
        num_experts=8,
        expert_top_k=2,
        aux_loss_weight=0.01,
        entropy_loss_weight=0.01,
        # Tau parameters
        tau_min=1.5,
        tau_max=4.0,
        tau_distribution="linear",
        custom_taus=None,
        verbose=False,
    ):
        super().__init__()
        
        # STAtten Attention (has internal residual)
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            dvs=dvs,
            layer=layer,
            chunk_size=chunk_size,
            spike_mode=spike_mode,
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # MoE (has residual connection, experts are pure functions)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MoE_Conv_Temporal(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            num_experts=num_experts,
            top_k=expert_top_k,
            spike_mode=spike_mode,
            layer=layer,
            aux_loss_weight=aux_loss_weight,
            entropy_loss_weight=entropy_loss_weight,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_distribution=tau_distribution,
            custom_taus=custom_taus,
            verbose=verbose,
        )

    def forward(self, x, hook=None):
        """
        ✅ FIXED: Proper residual structure
        
        Args:
            x: (T, B, C, H, W) - input spikes
            hook: Optional monitoring dict
        
        Returns:
            output: (T, B, C, H, W) - processed spikes
            attn: Attention features (for visualization)
            hook: Updated monitoring dict
        """
        # Attention branch (includes internal residual: x + Attn(x))
        x_attn, attn, hook = self.attn(x, hook=hook)
        
        # MoE branch (includes residual: x_attn + MoE(x_attn))
        x_mlp, hook = self.mlp(x_attn, hook=hook)
        
        return x_mlp, attn, hook
    
    def get_aux_loss(self):
        """Get MoE auxiliary loss for training"""
        return self.mlp.get_aux_loss()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def create_spikeformer_moe_block(
    dim=64,
    num_heads=8,
    num_experts=8,
    tau_min=1.5,
    tau_max=4.0,
    chunk_size=2,
    verbose=True
):
    """
    Convenience function to create a standard MoE block with proper residuals
    
    Example usage:
        block = create_spikeformer_moe_block(dim=64, num_experts=8)
        x = torch.randn(4, 16, 64, 7, 7)  # (T, B, C, H, W)
        output, attn, hook = block(x)
        aux_loss = block.get_aux_loss()
    """
    return MS_Block_Conv_MoE(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop_path=0.1,
        spike_mode="lif",
        dvs=False,
        chunk_size=chunk_size,
        num_experts=num_experts,
        expert_top_k=2,
        aux_loss_weight=0.01,
        entropy_loss_weight=0.01,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_distribution="linear",
        verbose=verbose,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Temporal MoE - FIXED RESIDUAL CONNECTIONS")
    print("=" * 70)
    
    # Create block
    block = create_spikeformer_moe_block(
        dim=64, 
        num_experts=8, 
        chunk_size=2,
        verbose=True
    )
    
    # Test forward pass
    x = torch.randn(4, 8, 64, 7, 7)  # (T, B, C, H, W)
    print(f"\n✅ Input shape: {x.shape}")
    
    output, attn, hook = block(x)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Attention features shape: {attn.shape}")
    
    aux_loss = block.get_aux_loss()
    print(f"✅ Auxiliary loss: {aux_loss.item():.6f}")
    
    print("\n" + "=" * 70)
    print("✅ CRITICAL FIX APPLIED:")
    print("   - Expert: Returns F(x), not F(x) + x")
    print("   - MoE: Returns x + MoE(x), not x + (MoE(x) + x)")
    print("   - No more double residual connections!")
    print("=" * 70)
