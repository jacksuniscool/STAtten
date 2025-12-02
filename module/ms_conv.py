"""
AutoMoE: Automatic switching between Token-level and Temporal MoE

This module automatically selects the appropriate MoE variant based on input:
- T=1 or static data → MS_MoE_Conv (token-level, spatial specialization)
- T>1 with temporal dynamics → MS_MoE_Conv_Temporal (batch-level, temporal specialization)

Author: Based on SNN MoE architecture
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
from .ssa import MS_SSA_Conv
from .mlp import MS_MLP_Conv
from .moe_conv import MS_MoE_Conv
from .moe_conv import MS_MoE_Conv_Temporal

# Import the base MoE implementations (assumed to be in the same file or imported)
# from moe_final_production import MS_MoE_Conv, MS_MoE_Conv_Temporal, MS_MLP_Expert, SpikeRouter


class MS_AutoMoE(nn.Module):
    """
    Automatic MoE that intelligently switches between routing strategies.
    
    🎯 AUTOMATIC BEHAVIOR:
    
    1. Static/Single-timestep data (T=1):
       → Routes to MS_MoE_Conv (token-level)
       → Good for: CIFAR-10, ImageNet, static spatial tasks
       → Tau effect: Different gains/nonlinearities
    
    2. Temporal sequences (T>1):
       → Routes to MS_MoE_Conv_Temporal (batch-level)
       → Good for: Speech, ultrasound, tactile sensors, event cameras
       → Tau effect: TRUE temporal integration (fast vs slow memory)
    
    3. Hybrid mode (optional):
       → Can force one mode regardless of T
       → Useful for ablation studies
    
    This eliminates manual switching and enables the same model architecture
    to work optimally on both spatial and temporal datasets.
    
    Args:
        in_features: Input channel dimension
        hidden_features: Hidden layer dimension
        out_features: Output channel dimension
        num_experts: Number of expert networks
        top_k: Experts to activate per token/batch
        spike_mode: "lif" or "plif"
        layer: Layer index
        aux_loss_weight: Load balancing loss weight
        tau_min: Minimum tau (fast leak)
        tau_max: Maximum tau (slow leak)
        tau_distribution: "linear", "log", or "custom"
        custom_taus: Optional explicit tau list
        verbose: Print initialization info
        force_mode: "auto", "token", or "temporal" - override automatic selection
        temporal_threshold: Min T to use temporal mode (default: 2)
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
        force_mode="auto",  # "auto", "token", or "temporal"
        temporal_threshold=2,  # Minimum T to trigger temporal mode
    ):
        super().__init__()
        
        self.force_mode = force_mode
        self.temporal_threshold = temporal_threshold
        self.layer = layer
        self.aux_loss_weight = aux_loss_weight
        
        if verbose:
            print(f"[AutoMoE Layer {layer}] Initializing with force_mode='{force_mode}'")
        
        # Create token-level MoE (always needed, fallback for T=1)
        self.token_moe = MS_MoE_Conv(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            spike_mode=spike_mode,
            layer=layer,
            aux_loss_weight=aux_loss_weight,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_distribution=tau_distribution,
            custom_taus=custom_taus,
            verbose=verbose,
        )
        
        # Create temporal MoE (for T>1 sequences)
        self.temporal_moe = MS_MoE_Conv_Temporal(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            spike_mode=spike_mode,
            layer=layer,
            aux_loss_weight=aux_loss_weight,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_distribution=tau_distribution,
            custom_taus=custom_taus,
            verbose=verbose,
        )
        
        # Statistics tracking
        self.register_buffer('token_calls', torch.tensor(0))
        self.register_buffer('temporal_calls', torch.tensor(0))
        
        if verbose:
            print(f"[AutoMoE Layer {layer}] Both token and temporal MoE initialized")
            print(f"[AutoMoE Layer {layer}] Will use temporal mode when T >= {temporal_threshold}")
    
    def forward(self, x, hook=None):
        """
        Automatically route to appropriate MoE based on temporal dimension.
        
        Decision logic:
        1. If force_mode != "auto": Use forced mode
        2. If T < temporal_threshold: Use token-level MoE
        3. If T >= temporal_threshold: Use temporal MoE
        
        Args:
            x: Input tensor (T, B, C, H, W)
            hook: Optional activation storage dict
        
        Returns:
            output: MoE output (T, B, C, H, W)
            hook: Updated hook dict
        """
        T, B, C, H, W = x.shape
        
        # Determine which MoE to use
        if self.force_mode == "token":
            use_temporal = False
        elif self.force_mode == "temporal":
            use_temporal = True
        else:  # "auto"
            use_temporal = (T >= self.temporal_threshold)
        
        # Route to appropriate MoE
        if use_temporal:
            self.temporal_calls += 1
            output, hook = self.temporal_moe(x, hook=hook)
            
            # Track which mode was used in hooks
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_moe_mode"] = "temporal"
        else:
            self.token_calls += 1
            output, hook = self.token_moe(x, hook=hook)
            
            # Track which mode was used in hooks
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_moe_mode"] = "token"
        
        return output, hook
    
    def get_aux_loss(self):
        """
        Get auxiliary load balancing loss from whichever MoE was last used.
        
        Since both MoEs might be used in the same forward pass (different samples),
        we sum their losses.
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if hasattr(self.token_moe, 'load_balancing_loss'):
            if self.token_moe.load_balancing_loss is not None:
                loss += self.aux_loss_weight * self.token_moe.load_balancing_loss
        
        if hasattr(self.temporal_moe, 'load_balancing_loss'):
            if self.temporal_moe.load_balancing_loss is not None:
                loss += self.aux_loss_weight * self.temporal_moe.load_balancing_loss
        
        return loss
    
    def get_usage_stats(self):
        """Get statistics on which mode was used"""
        return {
            'token_calls': int(self.token_calls.item()),
            'temporal_calls': int(self.temporal_calls.item()),
            'total_calls': int(self.token_calls.item() + self.temporal_calls.item()),
            'temporal_ratio': float(self.temporal_calls / (self.token_calls + self.temporal_calls + 1e-8)),
        }
    
    def get_expert_tau_stats(self):
        """Get tau statistics (same for both MoEs)"""
        return self.token_moe.get_expert_tau_stats()


class MS_Block_Conv_AutoMoE(nn.Module):
    """
    Transformer block with AutoMoE that automatically adapts to data type.
    
    This is the recommended block to use in your architecture. It will:
    - Use token-level MoE for CIFAR-10, ImageNet, etc. (T=1)
    - Use temporal MoE for speech, ultrasound, tactile sensors (T>1)
    - Automatically leverage different tau values appropriately
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        [... other attention params ...]
        use_moe: Enable MoE (if False, uses standard MLP)
        num_experts: Number of experts
        expert_top_k: Experts per token/batch
        aux_loss_weight: Load balancing weight
        tau_min: Minimum tau
        tau_max: Maximum tau
        tau_distribution: Tau spacing strategy
        custom_taus: Optional explicit tau values
        force_moe_mode: "auto", "token", or "temporal"
        temporal_threshold: Min T for temporal mode
    """
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
        # MoE parameters
        use_moe=True,
        num_experts=8,
        expert_top_k=2,
        aux_loss_weight=0.01,
        # Variable tau parameters
        tau_min=1.5,
        tau_max=4.0,
        tau_distribution="linear",
        custom_taus=None,
        # AutoMoE parameters
        force_moe_mode="auto",  # "auto", "token", or "temporal"
        temporal_threshold=2,
        verbose=False,
    ):
        super().__init__()
        
        # Import attention module (assumed to be defined elsewhere)
        # from your_module import MS_SSA_Conv, MS_MLP_Conv
        
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
            # Use AutoMoE - automatically adapts to data
            self.mlp = MS_AutoMoE(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                num_experts=num_experts,
                top_k=expert_top_k,
                spike_mode=spike_mode,
                layer=layer,
                aux_loss_weight=aux_loss_weight,
                tau_min=tau_min,
                tau_max=tau_max,
                tau_distribution=tau_distribution,
                custom_taus=custom_taus,
                force_mode=force_moe_mode,
                temporal_threshold=temporal_threshold,
                verbose=verbose,
            )
        else:
            # Standard MLP (no MoE)
            self.mlp = MS_MLP_Conv(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                spike_mode=spike_mode,
                layer=layer,
            )
    
    def forward(self, x, hook=None):
        """Standard block forward pass"""
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook
    
    def get_aux_loss(self):
        """Get auxiliary loss from MoE if used"""
        if self.use_moe and hasattr(self.mlp, 'get_aux_loss'):
            return self.mlp.get_aux_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("AutoMoE: Automatic Token/Temporal MoE Switching")
    print("=" * 80)
    
    print("\n🎯 KEY FEATURE:")
    print("This module automatically uses the right MoE variant based on your data!")
    
    print("\n" + "=" * 80)
    print("Test 1: CIFAR-10 style data (T=1)")
    print("=" * 80)
    
    auto_moe = MS_AutoMoE(
        in_features=128,
        hidden_features=512,
        num_experts=4,
        top_k=2,
        tau_min=1.5,
        tau_max=4.0,
        verbose=True,
        force_mode="auto"
    )
    
    # Simulate CIFAR-10 data: T=1
    x_cifar = torch.randn(1, 8, 128, 7, 7)  # (T=1, B=8, C=128, H=7, W=7)
    print(f"\nInput shape: {x_cifar.shape}")
    
    output_cifar, _ = auto_moe(x_cifar)
    print(f"Output shape: {output_cifar.shape}")
    print(f"Mode used: {auto_moe.get_usage_stats()}")
    print("✓ Automatically used TOKEN-level MoE (tau as gain differences)")
    
    print("\n" + "=" * 80)
    print("Test 2: Speech/Temporal data (T>1)")
    print("=" * 80)
    
    # Simulate temporal data: T=40 (like speech commands)
    x_speech = torch.randn(40, 8, 128, 7, 7)  # (T=40, B=8, C=128, H=7, W=7)
    print(f"\nInput shape: {x_speech.shape}")
    
    output_speech, _ = auto_moe(x_speech)
    print(f"Output shape: {output_speech.shape}")
    print(f"Mode used: {auto_moe.get_usage_stats()}")
    print("✓ Automatically used TEMPORAL MoE (tau as temporal integration)")
    
    print("\n" + "=" * 80)
    print("Test 3: Force mode override")
    print("=" * 80)
    
    # Force token mode even for temporal data
    auto_moe_forced = MS_AutoMoE(
        in_features=128,
        hidden_features=512,
        num_experts=4,
        top_k=2,
        force_mode="token",  # Force token mode
        verbose=False
    )
    
    output_forced, _ = auto_moe_forced(x_speech)  # T=40, but forced to token
    print(f"Temporal data (T=40) with force_mode='token':")
    print(f"Mode used: {auto_moe_forced.get_usage_stats()}")
    print("✓ Forced to use TOKEN mode (useful for ablation)")
    
    print("\n" + "=" * 80)
    print("📊 REAL-WORLD DATASET BEHAVIOR:")
    print("=" * 80)
    
    datasets = [
        ("CIFAR-10", (1, 32, 3, 32, 32), "Token MoE → Spatial specialization"),
        ("ImageNet", (1, 64, 3, 224, 224), "Token MoE → Spatial specialization"),
        ("Speech Commands", (40, 32, 128, 7, 7), "Temporal MoE → Fast/slow dynamics"),
        ("Ultrasound Video", (128, 16, 64, 32, 32), "Temporal MoE → Transient/sustained"),
        ("Tactile Sensor", (200, 8, 32, 7, 7), "Temporal MoE → Tap/hold specialization"),
        ("Event Camera", (20, 32, 2, 128, 128), "Temporal MoE → Fast/slow motion"),
    ]
    
    for dataset_name, shape, behavior in datasets:
        T = shape[0]
        mode = "Token" if T < 2 else "Temporal"
        print(f"\n{dataset_name:20s} T={T:3d} → {mode:8s} MoE ({behavior})")
    
    print("\n" + "=" * 80)
    print("🚀 USAGE IN YOUR MODEL:")
    print("=" * 80)
    print("""
Replace your current blocks with:

# Old (manual):
self.mlp = MS_MoE_Conv(...)  # Always token-level

# New (automatic):
self.mlp = MS_AutoMoE(...)   # Adapts to your data!

Or use the complete block:
MS_Block_Conv_AutoMoE(
    dim=256,
    num_heads=8,
    use_moe=True,
    num_experts=8,
    expert_top_k=2,
    tau_min=1.5,
    tau_max=4.0,
    force_moe_mode="auto",  # Automatic switching!
)

Now the SAME architecture works optimally on:
✓ CIFAR-10 (spatial tasks)
✓ Speech commands (temporal tasks)
✓ Your tactile sensors (temporal dynamics)
✓ ANY dataset - automatically adapts!
""")
    
    print("=" * 80)
    print("💡 BENEFITS:")
    print("=" * 80)
    print("✓ No manual switching needed")
    print("✓ Same architecture for spatial and temporal tasks")
    print("✓ Tau values used appropriately for each task")
    print("✓ Easy ablation studies (force_mode parameter)")
    print("✓ Clear logging of which mode was used")
    print("=" * 80)
MS_Block_Conv = MS_Block_Conv_AutoMoE
