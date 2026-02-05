from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

MIN_EXPERT_CAPACITY = 4

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]


class dvs_pooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class Experts(nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        hidden_features,
        out_features=None,
        spike_mode="lif",
    ):
        super().__init__()
        out_features = out_features or in_features
        self.num_experts = num_experts
        self.hidden_features = hidden_features
        
        w1 = torch.zeros(num_experts, in_features, hidden_features)
        w2 = torch.zeros(num_experts, hidden_features, out_features)
        
        std1 = 1 / math.sqrt(in_features)
        w1.uniform_(-std1, std1)
        std2 = 1 / math.sqrt(hidden_features)
        w2.uniform_(-std2, std2)
        
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

        if spike_mode == "lif":
            self.act = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.act = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
    
    def reset(self):
        if hasattr(self.act, 'reset'):
            self.act.reset()
    
    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        
        hidden = self.act(hidden)
        
        output = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return output




class SpikeRouter(nn.Module):

    def __init__(
        self,
        in_features,
        num_experts,
        spike_mode="lif",
        eps=1e-9,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.eps = eps
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        
        # Spike-based router network
        self.router_conv = nn.Conv2d(in_features, num_experts, kernel_size=1, stride=1)
        self.router_bn = nn.BatchNorm2d(num_experts)
        
        if spike_mode == "lif":
            self.router_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.router_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
    
    def reset(self):
        if hasattr(self.router_lif, 'reset'):
            self.router_lif.reset()
    
    def forward(self, x):

        T, B, C, H, W = x.shape
        N = H * W  
        
        if self.training:
            capacity_factor = self.capacity_factor_train
        else:
            capacity_factor = self.capacity_factor_eval
        
        router_out = self.router_lif(x)
        router_out = self.router_conv(router_out.flatten(0, 1))  
        router_out = self.router_bn(router_out)
        
     
        raw_gates = router_out.permute(0, 2, 3, 1).reshape(T * B, N, self.num_experts)
        raw_gates = raw_gates.softmax(dim=-1)
        
        num_gates = self.num_experts
        group_size = N  
        

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates
        
        gates_without_top_1 = raw_gates * (1. - mask_1)
        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()
        
        denom = gate_1 + gate_2 + self.eps
        gate_1 = gate_1 / denom
        gate_2 = gate_2 / denom
        
        density_1 = mask_1.mean(dim=-2)
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)
        
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
       
        mask_1 = mask_1 * (position_in_expert_1 < expert_capacity_f).float()
        
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        
        mask_1_flat = mask_1.sum(dim=-1)
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        gate_1 = gate_1 * mask_1_flat
        

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 = position_in_expert_2 * mask_2
        mask_2 = mask_2 * (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)
        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 = gate_2 * mask_2_flat
        
        # combine_tensor shape: [T*B, N, num_experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :]
            +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )
        
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        
        return dispatch_tensor, combine_tensor, loss, expert_capacity


class MS_MoE_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        num_experts=8,
        spike_mode="lif",
        layer=0,
        aux_loss_weight=0.01,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.num_experts = num_experts
        self.layer = layer
        self.aux_loss_weight = aux_loss_weight
        self.in_features = in_features
        self.out_features = out_features
        
        self.router = SpikeRouter(
            in_features=in_features,
            num_experts=num_experts,
            spike_mode=spike_mode,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
        )
        
        self.experts = Experts(
            num_experts=num_experts,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            spike_mode=spike_mode,
        )
          
        self.load_balancing_loss = None
    
    def reset(self):
        self.router.reset()
        self.experts.reset()
    
    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        N = H * W
        identity = x
        
        dispatch_tensor, combine_tensor, loss, expert_capacity = self.router(x)
        self.load_balancing_loss = loss
        
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(T * B, N, C)
        
        expert_inputs = torch.einsum('bnd,bnec->ebcd', x_flat, dispatch_tensor)
        
        E = self.num_experts
        orig_shape = expert_inputs.shape
        
        expert_inputs = expert_inputs.reshape(E, -1, C) 
        expert_outputs = self.experts(expert_inputs)
        
        expert_outputs = expert_outputs.reshape(*orig_shape[:-1], self.out_features)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        
        output = output.reshape(T, B, H, W, self.out_features).permute(0, 1, 4, 2, 3)
        
        return output + identity, hook



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
    
    def reset(self):
        for m in self.children():
            if hasattr(m, "reset"):
               m.reset()
            
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
        num_experts=8,
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
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.num_experts = num_experts
        self.aux_loss_weight = aux_loss_weight
        
        self.mlp = MS_MoE_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            num_experts=num_experts,
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
