import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.spiketrack.ni_lif import mem_update
import os
import matplotlib.pyplot as plt

def make_conv_layer(in_channels, out_channels, norm_layer,
                     kernel_size=1, stride=1, padding=0, groups=1, bias=False):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding=padding, groups=groups, bias=bias),
        norm_layer(out_channels)
    )

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias

class FrozenBatchNorm1d(nn.Module):
    """
    BatchNorm1d where the batch statistics and the affine parameters are fixed.
    """
    def __init__(self, num_features):
        super(FrozenBatchNorm1d, self).__init__()
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm1d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # x: [N, C, L] or [N, C]
        w = self.weight.reshape(1, -1, 1)  # for [N, C, L]
        b = self.bias.reshape(1, -1, 1)
        rv = self.running_var.reshape(1, -1, 1)
        rm = self.running_mean.reshape(1, -1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def downsample(x, target_h, target_w):
    """
    avg pool: (t, b, c, h, w) -> (t, b, c, target_h, target_w)
    """
    t, b, c, h, w = x.shape
    pooled = []
    for i in range(t):
        pooled.append(
            F.adaptive_avg_pool2d(x[i], output_size=(target_h, target_w))
        )
    return torch.stack(pooled, dim=0)

def upsample(x, target_h, target_w, mode="bilinear"):
    """
    interpolate:  (t, b, c, h, w) -> (t, b, c, target_h, target_w)
    mode:  'nearest', 'bilinear'
    """
    t, b, c, h_p, w_p = x.shape
    upsampled = []
    for i in range(t):
        upsampled.append(
            F.interpolate(x[i], size=(target_h, target_w), mode=mode, align_corners=False if mode=="bilinear" else None)
        )
    return torch.stack(upsampled, dim=0)

class GateModule(nn.Module):
    def __init__(self, dim, mode="channel", time_step=4, skip_ts=False, resolution=256):
        """
        Args:
            dim:           C_v（point-wise used）
            mode:         'point' or 'channel'
        """
        super().__init__()
        self.mode = mode
        if resolution ==256:
            spatial_size=16
        elif resolution ==384:
            spatial_size = 24
        else:
             raise ValueError("resolution only support 256 or 384.")      
        if mode == "point":
            self.mlp = nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim//2, kernel_size=1),
                mem_update(time_step=time_step, skip_ts=skip_ts),
                nn.Conv1d(in_channels=dim//2, out_channels=1, kernel_size=1)
            )
        else:  # channel-wise
            in_ch = spatial_size * spatial_size
            self.mlp = nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=1),
                mem_update(time_step=time_step, skip_ts=skip_ts),
                nn.Conv1d(in_channels=in_ch // 2, out_channels=1, kernel_size=1),

            )

    def forward(self, x):
        """
        Args:
            x - spike tensor : [T, B, C_v, H_s, W_s]
        """
        if self.mode == "point":
            return self.pixel_wise_gate(x)
        else:
            return self.channel_wise_gate(x)

    def pixel_wise_gate(self, x):
        T, B, C_v, H_s, W_s = x.shape
        x_perm = x.permute(1, 3, 4, 2, 0)                  # [B, H, W, C, T]
        x_reshaped = x_perm.reshape(B * H_s * W_s, C_v, T)  # [B*H*W, C, T]
        gate_scores = self.mlp(x_reshaped).squeeze(1)      # [B*H*W, T]
        gate_scores = gate_scores.view(B, H_s, W_s, T).permute(3, 0, 1, 2)  # [T, B, H, W]
        gate_weights = F.softmax(gate_scores, dim=0).unsqueeze(2)           # [T, B, 1, H, W]
        weighted = x * gate_weights  # [T, B, C, H, W]
        return weighted.sum(dim=0)  # [B, C, H, W]

    def channel_wise_gate(self, x):
        T, B, C_v, H_s, W_s = x.shape
        x_flat = x.view(T, B, C_v, H_s * W_s).permute(1, 2, 3, 0)   # [B, C, H*W, T]
        x_flat = x_flat.reshape(-1, H_s * W_s, T)                   # [B*C, H*W, T]
        gate_logits = self.mlp(x_flat).squeeze(1)                   # [B*C, T]
        gate_logits = gate_logits.view(B, C_v, T)                   # [B, C, T]
        gate_weights = F.softmax(gate_logits, dim=-1)               # [B, C, T]
        x_bcthw = x.permute(1, 2, 0, 3, 4)                           # [B, C, T, H, W]
        gate_weights_expand = gate_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
        return (x_bcthw * gate_weights_expand).sum(dim=2)            # [B, C, H, W]