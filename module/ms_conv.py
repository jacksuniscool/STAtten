"""
Mixture of Experts (MoE) for Spiking Neural Networks with Variable Leak Rates

This module implements two variants of sparse MoE for SNNs:
1. MS_MoE_Conv: Token-level routing (τ as gain/nonlinearity difference)
2. MS_MoE_Conv_Temporal: Batch-level routing (τ as true temporal specialization)

Author: Based on original SNN architecture with MoE extensions
Date: 2025
"""

from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_MLP_Expert(nn.Module):
    """
    Single expert network with configurable leak rate (tau).
    
    Architecture:
        Input → LIF → Conv1x1 → BN → (residual) → LIF → Conv1x1 → BN → Output
    
    Key design notes:
    - Uses kernel_size=1, stride=1 convolutions (spatial dims preserved)
    - BatchNorm after convolutions
    - Residual connection when in_features == hidden_features
    - Reshape logic is robust to future architecture changes
    
    Args:
        in_features: Input channel dimension
        hidden_features: Hidden layer dimension (default: same as input)
        out_features: Output channel dimension (default: same as input)
        spike_mode: "lif" or "plif" for spiking neuron type
        tau: Leak time constant for LIF neurons
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
        self.res = in_features == hidden_features
        self.tau = tau
        
        # First layer: expand to hidden dimension
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend="cupy")
        else:
            raise ValueError(f"Unknown spike_mode: {spike_mode}")
        
        # Second layer: project back to output dimension
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend="cupy")
        else:
            raise ValueError(f"Unknown spike_mode: {spike_mode}")
        
        self.c_hidden = hidden_features
        self.c_output = out_features

    def reset(self):
        """Reset LIF neuron states - critical for independent processing of samples"""
        for m in self.modules():
            if hasattr(m, "reset"):
                m.reset()

    def forward(self, x):
        """
        Forward pass with robust spatial dimension handling.
        
        Shape transformation:
            Input:  (T, B, C, H, W)
            → LIF:  (T, B, C, H, W)
            → Conv: (T*B, C', H', W')  # H'=H, W'=W for kernel=1, stride=1
            → Reshape back to 5D
        
        The reshape dynamically computes output spatial dims, making it robust to:
        - Future architecture changes (e.g., stride>1, pooling)
        - Different input spatial sizes
        - Padding variations
        
        Current architecture uses kernel=1, stride=1, so H_out=H, W_out=W,
        but the code doesn't assume this - it measures it.
        """
        T, B, C, H, W = x.shape
        identity = x
        
        # First transformation
        x = self.fc1_lif(x)  # (T, B, C, H, W)
        x_flat = x.flatten(0, 1)  # (T*B, C, H, W)
        x = self.fc1_conv(x_flat)  # (T*B, c_hidden, H_out, W_out)
        x = self.fc1_bn(x)
        
        # Dynamically compute output spatial dims (robust to architecture changes)
        # Currently: kernel=1, stride=1 → H_out=H, W_out=W, but this is future-proof
        _, _, H_out, W_out = x.shape
        x = x.reshape(T, B, self.c_hidden, H_out, W_out).contiguous()
        
        # Optional residual connection (when dimensions match)
        if self.res:
            x = identity + x
            identity = x
        
        # Second transformation
        x = self.fc2_lif(x)  # (T, B, c_hidden, H_out, W_out)
        x_flat = x.flatten(0, 1)  # (T*B, c_hidden, H_out, W_out)
        x = self.fc2_conv(x_flat)  # (T*B, C_out, H_out2, W_out2)
        x = self.fc2_bn(x)
        
        # Again, compute output dims dynamically
        _, C_out, H_out2, W_out2 = x.shape
        x = x.reshape(T, B, C_out, H_out2, W_out2).contiguous()
        
        # Final residual connection
        x = x + identity
        
        return x


class SpikeRouter(nn.Module):
    """
    Spike-based router for expert selection.
    
    Uses a small spiking network to generate routing logits, then applies
    softmax + top-k selection for sparse expert assignment.
    
    Args:
        in_features: Input channel dimension
        num_experts: Total number of experts
        top_k: Number of experts to route each token to
        spike_mode: "lif" or "plif"
    """
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
        else:
            raise ValueError(f"Unknown spike_mode: {spike_mode}")
    
    def forward(self, x):
        """
        Generate routing decisions via spiking network.
        
        Args:
            x: Input tensor of shape (T, B, C, H, W)
        
        Returns:
            top_k_weights: Normalized routing probabilities (T*B, top_k)
            top_k_indices: Selected expert indices (T*B, top_k)
            router_logits: Raw router outputs for loss computation (T*B, num_experts)
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
        
        # Renormalize top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices, router_logits


class MS_MoE_Conv(nn.Module):
    """
    Sparse Mixture of Experts for SNN with Variable Leak Rates.
    
    ⚠️ DESIGN LIMITATION - Token-level routing (T=1 per expert):
    
    This implementation routes at token-level where each (t,b) position is
    independently routed to experts. Each expert processes tokens with T=1,
    meaning:
    
    - Tau values act more like different nonlinearities/gains than true 
      timescale specialization
    - No temporal integration happens within experts (single timestep)
    - Good for: Spatial/feature pattern specialization with different gains
    - Limited for: Temporal memory window specialization
    
    For TRUE temporal specialization where different tau values create different
    temporal integration windows, use MS_MoE_Conv_Temporal instead.
    
    Args:
        in_features: Input channel dimension
        hidden_features: Hidden layer dimension in experts
        out_features: Output channel dimension
        num_experts: Total number of expert networks
        top_k: Number of experts to activate per token
        spike_mode: "lif" or "plif"
        layer: Layer index (for logging/hooks)
        aux_loss_weight: Weight for load balancing auxiliary loss
        tau_min: Minimum tau value (fastest leak)
        tau_max: Maximum tau value (slowest leak)
        tau_distribution: "linear", "log", or "custom"
        custom_taus: Optional explicit list of tau values
        verbose: Print tau values during initialization
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
        
        # Generate tau values for each expert
        if custom_taus is not None:
            assert len(custom_taus) == num_experts, \
                f"custom_taus length ({len(custom_taus)}) must match num_experts ({num_experts})"
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
        
        # Store as tensor buffer for easy device management and vectorized indexing
        self.register_buffer(
            'expert_taus', 
            torch.tensor(expert_taus_list, dtype=torch.float32)
        )
        
        if verbose:
            print(f"[MoE Layer {layer}] Expert tau values: {[f'{tau:.3f}' for tau in expert_taus_list]}")
        
        # Create router
        self.router = SpikeRouter(
            in_features=in_features,
            num_experts=num_experts,
            top_k=top_k,
            spike_mode=spike_mode,
        )
        
        # Create expert networks with different tau values
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

    def compute_load_balancing_loss(self, router_logits, selected_experts):
        """
        Compute auxiliary load balancing loss to encourage uniform expert usage.
        
        Based on: "Switch Transformers" (Fedus et al., 2021)
        Loss = num_experts * sum_over_experts(fraction_i * probability_i)
        
        This encourages:
        - Uniform distribution of tokens across experts
        - Prevents expert collapse (all tokens to one expert)
        - Differentiable through router_logits
        
        Args:
            router_logits: Raw router outputs (T*B, num_experts) - gradients flow here
            selected_experts: Selected expert indices (T*B, top_k) - for counting
        
        Returns:
            load_balancing_loss: Scalar loss term
        """
        # Count how many tokens were routed to each expert
        expert_counts = torch.bincount(
            selected_experts.view(-1),
            minlength=self.num_experts
        ).float()
        
        # Normalize to get fraction (non-differentiable, just for measuring)
        expert_fraction = expert_counts / selected_experts.numel()
        
        # Compute router probabilities (differentiable)
        router_probs = F.softmax(router_logits, dim=-1)
        router_prob_per_expert = router_probs.mean(dim=0)
        
        # Load balancing loss: product of counts and probabilities
        load_balancing_loss = self.num_experts * (expert_fraction * router_prob_per_expert).sum()
        
        return load_balancing_loss

    def forward(self, x, hook=None):
        """
        Token-level MoE routing with variable leak rate experts.
        
        Process:
        1. Route each (t,b) token independently to top_k experts
        2. Each expert processes its assigned tokens with T=1
        3. Combine expert outputs with learned weights
        4. Add residual connection
        
        ⚠️ Note: Since T=1 per expert, tau creates gain differences, not
        temporal integration differences.
        
        Args:
            x: Input tensor (T, B, C, H, W)
            hook: Optional dict for storing intermediate activations
        
        Returns:
            output: Mixed expert outputs (T, B, C, H, W)
            hook: Updated hook dict
        """
        T, B, C, H, W = x.shape
        identity = x
        
        # Reset all experts at the start (critical for independent processing)
        for expert in self.experts:
            expert.reset()
        
        # Get routing decisions from spike-based router
        top_k_weights, top_k_indices, router_logits = self.router(x)
        # Shapes: (T*B, top_k), (T*B, top_k), (T*B, num_experts)
        
        # Compute load balancing loss BEFORE detaching
        # This ensures gradients flow through router_logits to train the router
        self.load_balancing_loss = self.compute_load_balancing_loss(
            router_logits, 
            top_k_indices
        )
        
        # Detach routing weights from expert gradients
        # This is critical for MoE sparsity - experts don't affect routing decisions
        # Router gradients flow only through load_balancing_loss
        top_k_weights = top_k_weights.detach()
        top_k_indices = top_k_indices.detach()
        
        # Store routing information in hooks for analysis
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_routing_weights"] = top_k_weights.clone()
            hook[self._get_name() + str(self.layer) + "_routing_indices"] = top_k_indices.clone()
            # Store which tau values were selected (vectorized indexing)
            selected_taus = self.expert_taus[top_k_indices]
            hook[self._get_name() + str(self.layer) + "_selected_taus"] = selected_taus.clone()
        
        # Token-level expert processing
        output = torch.zeros_like(x)
        
        # Process each expert with its assigned tokens
        for expert_idx in range(self.num_experts):
            # Find all positions where this expert was selected
            expert_mask = (top_k_indices == expert_idx)  # (T*B, top_k)
            
            if not expert_mask.any():
                continue  # Skip if no tokens assigned to this expert
            
            # Get flat indices of assignments
            tb_indices, k_indices = torch.where(expert_mask)
            
            if len(tb_indices) == 0:
                continue
            
            # Convert flat TB indices back to (T, B) coordinates
            t_indices = tb_indices // B
            b_indices = tb_indices % B
            
            # Get routing weights for this expert's assignments
            expert_weights = top_k_weights[tb_indices, k_indices]  # (num_assignments,)
            
            # Gather assigned tokens
            num_tokens = len(t_indices)
            expert_input = torch.zeros(1, num_tokens, C, H, W, 
                                      device=x.device, dtype=x.dtype)
            
            for i, (t_idx, b_idx) in enumerate(zip(t_indices, b_indices)):
                expert_input[0, i] = x[t_idx, b_idx]
            
            # Process through expert (T=1, so tau acts as gain, not temporal memory)
            expert_output = self.experts[expert_idx](expert_input)  # (1, num_tokens, C, H, W)
            
            # Distribute weighted outputs back to original positions
            for i, (t_idx, b_idx, weight) in enumerate(zip(t_indices, b_indices, expert_weights)):
                output[t_idx, b_idx] += weight * expert_output[0, i]
        
        # Add residual connection (preserves gradient flow)
        output = output + identity
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_moe_output"] = output.detach()
        
        return output, hook

    def get_expert_tau_stats(self):
        """Get statistics about expert tau configuration"""
        taus = self.expert_taus.cpu().numpy()
        return {
            'expert_taus': taus.tolist(),
            'min_tau': float(taus.min()),
            'max_tau': float(taus.max()),
            'mean_tau': float(taus.mean()),
            'std_tau': float(taus.std()),
        }


class MS_MoE_Conv_Temporal(nn.Module):
    """
    Alternative MoE implementation enabling TRUE temporal specialization.
    
    ✅ KEY DIFFERENCE - Batch-level routing with full temporal sequences:
    
    This version routes at (B) granularity and processes full temporal
    sequences (T>1) through each expert. This means:
    
    - Different tau values create REAL differences in temporal integration
    - Fast-tau experts respond to transients, slow-tau experts integrate over time
    - Each expert sees temporal evolution and uses its leak rate meaningfully
    - Good for: Temporal memory window specialization (short vs long)
    - Trade-off: Less granular routing (batch-level vs token-level)
    
    Use this when you want experts to specialize on different TIMESCALES.
    
    Args:
        [Same as MS_MoE_Conv, see above]
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
        
        # Generate tau values
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
            print(f"[MoE Layer {layer} TEMPORAL] Expert tau values: {[f'{tau:.3f}' for tau in expert_taus_list]}")
        
        # Create router (same as token-level version)
        self.router = SpikeRouter(
            in_features=in_features,
            num_experts=num_experts,
            top_k=top_k,
            spike_mode=spike_mode,
        )
        
        # Create experts
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

    def compute_load_balancing_loss(self, router_logits, selected_experts):
        """Same as token-level version"""
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
        """
        Batch-level routing with full temporal processing.
        
        Process:
        1. Generate per-timestep routing decisions
        2. Aggregate routing over time (mean per batch element)
        3. Route entire temporal sequences to experts
        4. Each expert processes full T, enabling temporal dynamics
        5. Combine outputs with learned weights
        
        ✅ Experts see T>1, so tau creates real temporal integration differences!
        
        Args:
            x: Input tensor (T, B, C, H, W)
            hook: Optional dict for intermediate activations
        
        Returns:
            output: Mixed expert outputs (T, B, C, H, W)
            hook: Updated hook dict
        """
        T, B, C, H, W = x.shape
        identity = x
        
        # Reset all experts
        for expert in self.experts:
            expert.reset()
        
        # Get per-timestep routing decisions
        top_k_weights, top_k_indices, router_logits = self.router(x)
        # Shapes: (T*B, top_k), (T*B, top_k), (T*B, num_experts)
        
        # Aggregate routing over time: use mean routing logits per batch element
        router_logits_per_batch = router_logits.reshape(T, B, self.num_experts).mean(dim=0)  # (B, num_experts)
        
        # Recompute top-k per batch element
        routing_weights_batch = F.softmax(router_logits_per_batch, dim=-1)
        top_k_weights_batch, top_k_indices_batch = torch.topk(
            routing_weights_batch, self.top_k, dim=-1
        )
        top_k_weights_batch = top_k_weights_batch / top_k_weights_batch.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss
        self.load_balancing_loss = self.compute_load_balancing_loss(
            router_logits_per_batch, 
            top_k_indices_batch
        )
        
        # Detach for expert processing
        top_k_weights_batch = top_k_weights_batch.detach()  # (B, top_k)
        top_k_indices_batch = top_k_indices_batch.detach()  # (B, top_k)
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_routing_weights"] = top_k_weights_batch.clone()
            hook[self._get_name() + str(self.layer) + "_routing_indices"] = top_k_indices_batch.clone()
            selected_taus = self.expert_taus[top_k_indices_batch]
            hook[self._get_name() + str(self.layer) + "_selected_taus"] = selected_taus.clone()
        
        # Batch-level expert processing
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            # Find batch elements assigned to this expert
            expert_mask = (top_k_indices_batch == expert_idx)  # (B, top_k)
            
            if not expert_mask.any():
                continue
            
            # Get batch indices and their k positions
            b_indices, k_indices = torch.where(expert_mask)
            
            if len(b_indices) == 0:
                continue
            
            # Get routing weights
            expert_weights = top_k_weights_batch[b_indices, k_indices]  # (num_assignments,)
            
            # Gather full temporal sequences for assigned batch elements
            # expert_input shape: (T, num_assignments, C, H, W)
            expert_input = x[:, b_indices, :, :, :]
            
            # Process through expert - NOW with full temporal sequence (T>1)!
            # This is where tau creates real temporal dynamics
            expert_output = self.experts[expert_idx](expert_input)  # (T, num_assignments, C, H, W)
            
            # Distribute weighted outputs back to original positions
            for i, (b_idx, weight) in enumerate(zip(b_indices, expert_weights)):
                output[:, b_idx, :, :, :] += weight * expert_output[:, i, :, :, :]
        
        # Add residual connection
        output = output + identity
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_moe_output"] = output.detach()
        
        return output, hook

    def get_expert_tau_stats(self):
        """Get statistics about expert tau configuration"""
        taus = self.expert_taus.cpu().numpy()
        return {
            'expert_taus': taus.tolist(),
            'min_tau': float(taus.min()),
            'max_tau': float(taus.max()),
            'mean_tau': float(taus.mean()),
            'std_tau': float(taus.std()),
        }


if __name__ == "__main__":
    print("=" * 80)
    print("Production-Ready MoE for SNNs with Variable Leak Rates")
    print("=" * 80)
    
    print("\n📋 DESIGN CHOICES SUMMARY:")
    print("-" * 80)
    
    print("\n1. MS_MoE_Conv (Token-level routing):")
    print("   - Routing: Each (t,b) token independently")
    print("   - Expert input: T=1 (single timesteps)")
    print("   - Tau effect: Gain/nonlinearity differences")
    print("   - Best for: Spatial/feature pattern specialization")
    print("   - Use when: Different processing characteristics matter more than timescales")
    
    print("\n2. MS_MoE_Conv_Temporal (Batch-level routing):")
    print("   - Routing: Entire batch elements (all T)")
    print("   - Expert input: T>1 (full sequences)")
    print("   - Tau effect: TRUE temporal integration differences")
    print("   - Best for: Temporal memory window specialization")
    print("   - Use when: Fast vs slow temporal dynamics are important")
    
    print("\n" + "=" * 80)
    print("🔧 KEY ROBUSTNESS FEATURES:")
    print("=" * 80)
    print("✓ Dynamic spatial dimension computation (future-proof reshaping)")
    print("✓ Vectorized tau indexing (efficient and clean)")
    print("✓ Proper gradient flow (load balancing before detach)")
    print("✓ Device-aware tensor buffers (automatic GPU/CPU movement)")
    print("✓ Comprehensive documentation and warnings")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION FOR TACTILE SENSOR ANALYSIS:")
    print("=" * 80)
    print("Based on your piezoelectric sensor work:")
    print("- Fast transients (taps, quick touches) → Low tau experts")
    print("- Sustained pressure (holds, slides) → High tau experts")
    print("- Use MS_MoE_Conv_Temporal for this temporal specialization!")
    print("=" * 80)
