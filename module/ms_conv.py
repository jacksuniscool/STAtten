"""
Complete SNN MoE Implementation with Variable Tau and AutoMoE

This file contains:
1. Original MS_MLP_Conv, MS_SSA_Conv, MS_Block_Conv (unchanged)
2. MS_MLP_Expert with variable tau support
3. MS_MoE_Conv (token-level routing)
4. MS_MoE_Conv_Temporal (batch-level routing, true temporal dynamics)
5. MS_AutoMoE (automatic switching based on T)
6. MS_Block_Conv_AutoMoE (complete block with AutoMoE)

Usage:
- Replace MS_Block_Conv_MoE with MS_Block_Conv_AutoMoE in your model
- AutoMoE will automatically use token-level for CIFAR (T=1) and temporal for your sensors (T>1)
"""

from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ORIGINAL CLASSES (Unchanged from your code)
# ============================================================================

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

        ###### Attention #####
        if self.attention_mode == "STAtten":
            if self.dvs:
                scaling_factor = 1 / (H*H*self.chunk_size)
            else:
                scaling_factor = 1 / H

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


# ============================================================================
# MOE IMPLEMENTATION WITH VARIABLE TAU
# ============================================================================

class MS_MLP_Expert(nn.Module):
    """Expert network with configurable tau (leak rate)"""
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
        x_flat = x.flatten(0, 1)
        x = self.fc1_conv(x_flat)
        x = self.fc1_bn(x)
        _, _, H_out, W_out = x.shape
        x = x.reshape(T, B, self.c_hidden, H_out, W_out).contiguous()
        
        if self.res:
            x = identity + x
            identity = x
        
        x = self.fc2_lif(x)
        x_flat = x.flatten(0, 1)
        x = self.fc2_conv(x_flat)
        x = self.fc2_bn(x)
        _, C_out, H_out2, W_out2 = x.shape
        x = x.reshape(T, B, C_out, H_out2, W_out2).contiguous()
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
        
        self.router_conv = nn.Conv2d(in_features, num_experts, kernel_size=1, stride=1)
        self.router_bn = nn.BatchNorm2d(num_experts)
        
        if spike_mode == "lif":
            self.router_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.router_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
    
    def forward(self, x):
        T, B, C, H, W = x.shape
        
        router_out = self.router_lif(x)
        router_out = self.router_conv(router_out.flatten(0, 1))
        router_out = self.router_bn(router_out).reshape(T, B, self.num_experts, H, W)
        
        router_logits = router_out.mean(dim=[-2, -1])
        router_logits = router_logits.reshape(T * B, self.num_experts)
        
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices, router_logits


class MS_MoE_Conv(nn.Module):
    """Token-level MoE with variable tau (tau acts as gain differences, not temporal)"""
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
            print(f"[MoE Layer {layer}] Expert tau values: {[f'{tau:.3f}' for tau in expert_taus_list]}")
        
        self.router = SpikeRouter(
            in_features=in_features,
            num_experts=num_experts,
            top_k=top_k,
            spike_mode=spike_mode,
        )
        
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
        
        for expert in self.experts:
            expert.reset()
        
        top_k_weights, top_k_indices, router_logits = self.router(x)
        
        self.load_balancing_loss = self.compute_load_balancing_loss(router_logits, top_k_indices)
        
        top_k_weights = top_k_weights.detach()
        top_k_indices = top_k_indices.detach()
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_routing_weights"] = top_k_weights.clone()
            hook[self._get_name() + str(self.layer) + "_routing_indices"] = top_k_indices.clone()
            selected_taus = self.expert_taus[top_k_indices]
            hook[self._get_name() + str(self.layer) + "_selected_taus"] = selected_taus.clone()
        
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx)
            
            if not expert_mask.any():
                continue
            
            tb_indices, k_indices = torch.where(expert_mask)
            
            if len(tb_indices) == 0:
                continue
            
            t_indices = tb_indices // B
            b_indices = tb_indices % B
            expert_weights = top_k_weights[tb_indices, k_indices]
            
            num_tokens = len(t_indices)
            expert_input = torch.zeros(1, num_tokens, C, H, W, device=x.device, dtype=x.dtype)
            
            for i, (t_idx, b_idx) in enumerate(zip(t_indices, b_indices)):
                expert_input[0, i] = x[t_idx, b_idx]
            
            expert_output = self.experts[expert_idx](expert_input)
            
            for i, (t_idx, b_idx, weight) in enumerate(zip(t_indices, b_indices, expert_weights)):
                output[t_idx, b_idx] += weight * expert_output[0, i]
        
        output = output + identity
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_moe_output"] = output.detach()
        
        return output, hook


class MS_MoE_Conv_Temporal(nn.Module):
    """Batch-level MoE with full temporal sequences (tau creates REAL temporal dynamics)"""
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
        
        self.router = SpikeRouter(
            in_features=in_features,
            num_experts=num_experts,
            top_k=top_k,
            spike_mode=spike_mode,
        )
        
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
        
        for expert in self.experts:
            expert.reset()
        
        top_k_weights, top_k_indices, router_logits = self.router(x)
        
        router_logits_per_batch = router_logits.reshape(T, B, self.num_experts).mean(dim=0)
        
        routing_weights_batch = F.softmax(router_logits_per_batch, dim=-1)
        top_k_weights_batch, top_k_indices_batch = torch.topk(routing_weights_batch, self.top_k, dim=-1)
        top_k_weights_batch = top_k_weights_batch / top_k_weights_batch.sum(dim=-1, keepdim=True)
        
        self.load_balancing_loss = self.compute_load_balancing_loss(
            router_logits_per_batch, 
            top_k_indices_batch
        )
        
        top_k_weights_batch = top_k_weights_batch.detach()
        top_k_indices_batch = top_k_indices_batch.detach()
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_routing_weights"] = top_k_weights_batch.clone()
            hook[self._get_name() + str(self.layer) + "_routing_indices"] = top_k_indices_batch.clone()
            selected_taus = self.expert_taus[top_k_indices_batch]
            hook[self._get_name() + str(self.layer) + "_selected_taus"] = selected_taus.clone()
        
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices_batch == expert_idx)
            
            if not expert_mask.any():
                continue
            
            b_indices, k_indices = torch.where(expert_mask)
            
            if len(b_indices) == 0:
                continue
            
            expert_weights = top_k_weights_batch[b_indices, k_indices]
            
            expert_input = x[:, b_indices, :, :, :]
            
            expert_output = self.experts[expert_idx](expert_input)
            
            for i, (b_idx, weight) in enumerate(zip(b_indices, expert_weights)):
                output[:, b_idx, :, :, :] += weight * expert_output[:, i, :, :, :]
        
        output = output + identity
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_moe_output"] = output.detach()
        
        return output, hook


# ============================================================================
# AUTO MOE - AUTOMATIC SWITCHING
# ============================================================================

class MS_AutoMoE(nn.Module):
    """
    Automatically switches between token-level and temporal MoE based on T dimension.
    
    - T=1 (CIFAR, ImageNet) → Token MoE
    - T>1 (Speech, Sensors) → Temporal MoE
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
        force_mode="auto",
        temporal_threshold=2,
    ):
        super().__init__()
        
        self.force_mode = force_mode
        self.temporal_threshold = temporal_threshold
        self.layer = layer
        self.aux_loss_weight = aux_loss_weight
        
        if verbose:
            print(f"[AutoMoE Layer {layer}] mode='{force_mode}', threshold={temporal_threshold}")
        
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
        
        self.register_buffer('token_calls', torch.tensor(0))
        self.register_buffer('temporal_calls', torch.tensor(0))
    
    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        
        if self.force_mode == "token":
            use_temporal = False
        elif self.force_mode == "temporal":
            use_temporal = True
        else:  # "auto"
            use_temporal = (T >= self.temporal_threshold)
        
        if use_temporal:
            self.temporal_calls += 1
            output, hook = self.temporal_moe(x, hook=hook)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_moe_mode"] = "temporal"
        else:
            self.token_calls += 1
            output, hook = self.token_moe(x, hook=hook)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_moe_mode"] = "token"
        
        return output, hook
    
    def get_aux_loss(self):
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if hasattr(self.token_moe, 'load_balancing_loss'):
            if self.token_moe.load_balancing_loss is not None:
                loss += self.aux_loss_weight * self.token_moe.load_balancing_loss
        
        if hasattr(self.temporal_moe, 'load_balancing_loss'):
            if self.temporal_moe.load_balancing_loss is not None:
                loss += self.aux_loss_weight * self.temporal_moe.load_balancing_loss
        
        return loss
    
    def get_usage_stats(self):
        return {
            'token_calls': int(self.token_calls.item()),
            'temporal_calls': int(self.temporal_calls.item()),
            'total_calls': int(self.token_calls.item() + self.temporal_calls.item()),
        }


# ============================================================================
# COMPLETE BLOCKS WITH MOE
# ============================================================================

class MS_Block_Conv_MoE(nn.Module):
    """Original MoE block (token-level only)"""
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
        return x, attn, hook
    
    def get_aux_loss(self):
        if self.use_moe and hasattr(self.mlp, 'load_balancing_loss'):
            if self.mlp.load_balancing_loss is not None:
                return self.aux_loss_weight * self.mlp.load_balancing_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)


class MS_Block_Conv_AutoMoE(nn.Module):
    """
    ⭐ RECOMMENDED: Block with AutoMoE that adapts to your data!
    
    Use this instead of MS_Block_Conv_MoE for automatic token/temporal switching.
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
        force_moe_mode="auto",
        temporal_threshold=2,
        verbose=False,
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
    
    def get_aux_loss(self):
        if self.use_moe and hasattr(self.mlp, 'get_aux_loss'):
            return self.mlp.get_aux_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)
