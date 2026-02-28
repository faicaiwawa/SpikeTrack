from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

from lib.models.spiketrack.fuc import GateModule, downsample, upsample, FrozenBatchNorm2d, FrozenBatchNorm1d, \
    make_conv_layer
from lib.models.spiketrack.ni_lif import mem_update

resolution_to_patches = {256: 16, 384: 24}


class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = mem_update(time_step=4)
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            FrozenBatchNorm2d(med_channels)
        )

        self.spike2 = mem_update(time_step=4)
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            FrozenBatchNorm2d(med_channels)
        )
        self.spike3 = mem_update(time_step=4)
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            FrozenBatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.spike1(x)

        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike2(x)

        x = self.dwconv(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike3(x)

        x = self.pwconv2(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)
        self.mlp_ratio = mlp_ratio

        self.spike1 = mem_update(time_step=4)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = FrozenBatchNorm2d(dim * mlp_ratio)
        self.spike2 = mem_update(time_step=4)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = FrozenBatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x.flatten(0, 1))).reshape(T, B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, frozen=True, drop=0.0, layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)

        self.fc1_spike = mem_update(time_step=4)

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_spike = mem_update(time_step=4)
        if frozen:
            self.fc1_bn = FrozenBatchNorm1d(hidden_features)
            self.fc2_bn = FrozenBatchNorm1d(out_features)
        else:
            self.fc1_bn = nn.BatchNorm1d(hidden_features)
            self.fc2_bn = nn.BatchNorm1d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class MS_Attention_linear_3d(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
            lamda_ratio=1,
            frozen=True,
            resolution=384,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio
        self.head_spike = mem_update(time_step=4)
        self.q_spike = mem_update(time_step=4)
        self.k_spike = mem_update(time_step=4)
        self.v_spike = mem_update(time_step=4)
        self.resolution = resolution
        self.attn_spike = mem_update(time_step=4)
        norm_layer = FrozenBatchNorm2d if frozen else nn.BatchNorm2d

        self.q_conv = make_conv_layer(dim, dim, norm_layer)
        self.k_conv = make_conv_layer(dim, dim, norm_layer)
        self.v_conv = make_conv_layer(dim, int(dim * lamda_ratio), norm_layer)
        self.proj_conv = make_conv_layer(int(dim * lamda_ratio), dim, norm_layer)

        ''' '''
        if resolution not in resolution_to_patches:
            raise ValueError(f"Unsupported resolution: {resolution}")

        patches = resolution_to_patches[resolution]
        patch_size = patches * patches

        self.s_pos_embed = nn.Parameter(torch.zeros(1, dim, patch_size))
        self.t_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim, patch_size))
            for _ in range(4)
        ])

    def _apply_positional_encoding(self, x, branch, T=None):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.view(T, B, C, N)

        if branch == 'search':
            x = x + self.s_pos_embed
        elif branch == 'template':
            out_list = [
                x[t] + self.t_pos_embeds[min(t, 3)]
                for t in range(T)
            ]
            x = torch.stack(out_list, dim=0)
        else:
            raise ValueError(f"Unsupported branch: {branch}")

        x = x.view(T, B, C, H, W)  # ← 确保 reshape 回 5D
        return x

    def forward(self, x, branch):
        T, B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self._apply_positional_encoding(x, branch, T)
        x = self.head_spike(x)
        x = x.view(T * B, C, H, W)

        q = self.q_conv(x).reshape(T, B, C, N)
        k = self.k_conv(x).reshape(T, B, C, N)
        v = self.v_conv(x).reshape(T, B, C_v, N)


        def reshape_for_attention(tensor, num_channels):
            return (tensor.transpose(-1, -2)  # (T,B,N,C)
                    .reshape(T, B, N, self.num_heads, num_channels // self.num_heads)  # (T,B,N,H,C//H)
                    .permute(0, 1, 3, 2, 4)  # (T,B,H,N,C//H)
                    .contiguous())

        q = reshape_for_attention(self.q_spike(q), C)
        k = reshape_for_attention(self.k_spike(k), C)
        v = reshape_for_attention(self.v_spike(v), C_v)

        x = k.transpose(-2, -1) @ v  # (T,B,H,C//H,C_v//H)
        x = (q @ x) * self.scale * 2  # (T,B,H,N,C_v//H)

        x = (x.permute(0, 1, 3, 2, 4)  # (T,B,N,H,C_v//H)
             .reshape(T, B, N, C_v)
             .permute(0, 1, 3, 2)  # (T,B,C_v,N)
             .contiguous()
             .view(T, B, C_v, H, W))

        x = self.attn_spike(x)


        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        return x


class MS_Block_Spike_SepConv(nn.Module):
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
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            frozen=True,
            init_values=1e-6,
            resolution=384
    ):
        super().__init__()

        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)

        self.attn = MS_Attention_linear_3d(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
            frozen=frozen,
            resolution=resolution,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, frozen=frozen)

    def forward(self, x, branch):
        T, B, C, H, W = x.shape

        x = x + self.conv(x)

        x = x + self.attn(x, branch)

        x = x + self.mlp(x)

        return x


class MemoryRetrieval(nn.Module):
    def __init__(self, embed_dim, mlp_ratios, resolution, temp_num):
        super().__init__()
        self.retriever = Retriever(dim=embed_dim, resolution=resolution, temp_num=temp_num)
        self.mlp = MS_MLP(in_features=embed_dim, hidden_features=embed_dim * mlp_ratios, frozen=False)

    def forward(self, templates, search):
        search = search + self.retriever(templates, search)

        search = search + self.mlp(search)

        return search


class MS_DownSampling(nn.Module):
    def __init__(
            self,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,
            T=None,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = FrozenBatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = mem_update(time_step=4)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))

        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class Retriever(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
            lamda_ratio=4,
            resolution=384,
            temp_num=1
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio
        self.resolution = resolution
        self.head_spike = mem_update(time_step=4)

        norm_layer = nn.BatchNorm2d
        exp_dim = dim * lamda_ratio

        self.q_conv = make_conv_layer(dim, dim, norm_layer)
        self.k_conv = make_conv_layer(dim, dim, norm_layer)
        self.v_conv = make_conv_layer(dim, int(exp_dim), norm_layer)

        self.q_spike = mem_update(time_step=4)
        self.k_spike = mem_update(time_step=4)
        self.v_spike = mem_update(time_step=4)

        self.attn_spike = mem_update(time_step=4)
        self.proj_conv = make_conv_layer(exp_dim, dim, norm_layer)

        self.gate = GateModule(dim=exp_dim, mode="channel", time_step=4, resolution=resolution)
        self.gate_spike = mem_update(time_step=4)

        self.seg_responder_dw = nn.ModuleList([
            make_conv_layer(
                int(exp_dim), int(exp_dim), norm_layer,
                kernel_size=3, padding=1, groups=int(exp_dim)
            ) for _ in range(temp_num)
        ])

        self.seg_responder_pw = nn.ModuleList([
            make_conv_layer(
                int(exp_dim), int(dim), norm_layer
            ) for _ in range(temp_num)
        ])

        self.fuser = make_conv_layer(int(dim), int(dim), norm_layer)

        self.seg_pw_spike = mem_update(time_step=4)
        self.seg_dw_spike = mem_update(time_step=4)
        self.fuser_spike = mem_update(time_step=4)

        if resolution not in resolution_to_patches:
            raise ValueError(f"Unsupported resolution: {resolution}")

        patches = resolution_to_patches[resolution]
        patch_size = patches * patches

        self.s_pos_embed = nn.Parameter(torch.zeros(1, dim, patch_size))
        self.t_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim, patch_size))
            for _ in range(4)
        ])

    def forward(self, template, search):

        _H_s, _W_s = search.shape[-2:]
        template, search = self._preprocess_inputs(template, search)

        T, B, C = template.shape[:3]
        search = self._apply_positional_encoding(search, 'search')
        template = self._apply_positional_encoding(template, 'template', T=T)

        k_t, v_t, q_s, short_cut = self._compute_attention_features(template, search)

        q, k, v = self._reshape_for_attention(q_s, k_t, v_t)

        kv = k.transpose(-2, -1) @ v
        x = self._compute_attention_output(q, kv)

        x = self._recurrent_processing(x, kv, short_cut)

        x = self._postprocess_output(x).reshape(search.shape[0], B, C, search.shape[3], search.shape[4])

        return self._restore_resolution(x, _H_s, _W_s)

    def _apply_positional_encoding(self, x, branch, T=None):

        T, B, C, H, W = x.shape

        x = x.view(T, B, C, int(H * W))

        if branch == 'search':
            x = x + self.s_pos_embed

        elif branch == 'template':
            if T is None:
                raise ValueError("T must be provided for template branch")

            out_list = [
                x[t] + self.t_pos_embeds[min(t, 3)]
                for t in range(T)
            ]
            x = torch.stack(out_list, dim=0)

        x = x.view(T, B, C, H, W)

        return x

    def _preprocess_inputs(self, template, search):

        _, _, _, _H_s, _W_s = search.shape

        if self.resolution in resolution_to_patches:
            target_size = resolution_to_patches[self.resolution]
            if _H_s != target_size:
                search = downsample(search, target_size, target_size)
                template = downsample(template, target_size, target_size)

        return template, search

    def _compute_attention_features(self, template, search):

        T, B, C, H_t, W_t = template.shape
        T_s, _, _, H_s, W_s = search.shape
        C_v = int(C * self.lamda_ratio)

        template = self.head_spike(template)
        search = self.head_spike(search)

        k_t = self.k_conv(template.flatten(0, 1)).reshape(T, B, C, H_t, W_t)
        v_t = self.v_conv(template.flatten(0, 1)).reshape(T, B, C_v, H_t, W_t)
        q_s = self.q_conv(search.flatten(0, 1)).reshape(T_s, B, C, H_s, W_s)
        short_cut = q_s

        q_s = self.q_spike(q_s.flatten(3).squeeze(0).repeat(T, 1, 1, 1))
        v_t = self.v_spike(v_t).flatten(3)
        k_t = self.k_spike(k_t).flatten(3)

        return k_t, v_t, q_s, short_cut

    def _reshape_for_attention(self, q_s, k_t, v_t):

        T, B, C = q_s.shape[:3]
        N_s = q_s.shape[-1]
        N_t = k_t.shape[-1]
        C_v = v_t.shape[2]

        q = (q_s.transpose(-1, -2)
             .reshape(T, B, N_s, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())

        k = (k_t.transpose(-1, -2)
             .reshape(T, B, N_t, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())

        v = (v_t.transpose(-1, -2)
             .reshape(T, B, N_t, self.num_heads, C_v // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())

        return q, k, v

    def _compute_attention_output(self, q, kv):

        T, B, _, N_s, C_head = q.shape
        C_v = kv.shape[-1] * kv.shape[-3]  # num_heads * (C_v // num_heads)

        x = (q @ kv) * self.scale * 2
        return (x.permute(0, 1, 3, 2, 4)
                .reshape(T, B, N_s, C_v)
                .permute(0, 1, 3, 2).contiguous())

    def _recurrent_processing(self, x, kv, short_cut):

        T, B, C_v = x.shape[:3]
        H_s = W_s = int((x.shape[-1]) ** 0.5)
        recurrent_times = 1

        for i in range(recurrent_times):
            x = x.view(T, B, C_v, H_s, W_s)
            q = self.q_spike(x)

            q_list = []
            for t in range(T):
                if t < len(self.seg_responder_dw):
                    q_t = self.seg_responder_dw[t](q[t])
                    q_list.append(q_t)
            q = torch.stack(q_list, dim=0)
            q = self.seg_dw_spike(q)

            q_list = []
            for t in range(T):
                if t < len(self.seg_responder_pw):
                    q_t = self.seg_responder_pw[t](q[t])
                    q_list.append(q_t)
            q = torch.stack(q_list, dim=0)
            q = short_cut + q  # * self.layer_scale[i].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q = self.seg_pw_spike(q)

            q = self.fuser(q.flatten(0, 1)).reshape(T, B, -1, H_s, W_s).flatten(3)
            short_cut = q.reshape(T, B, -1, H_s, W_s)
            q = self.fuser_spike(q)

            C = q.shape[2]
            N_s = q.shape[-1]
            q = (q.transpose(-1, -2)
                 .reshape(T, B, N_s, self.num_heads, C // self.num_heads)
                 .permute(0, 1, 3, 2, 4).contiguous())

            x = self._compute_attention_output(q, kv)

        return x

    def _postprocess_output(self, x):

        T, B = x.shape[:2]
        H_s = W_s = int((x.shape[-1]) ** 0.5)
        C_v = x.shape[2]

        x = x.view(T, B, C_v, H_s, W_s)
        x = self.attn_spike(x)

        if T > 1:
            x = self.gate(x).unsqueeze(0)
            x = self.gate_spike(x)


        return self.proj_conv(x.flatten(0, 1))

    def _restore_resolution(self, x, _H_s, _W_s):

        current_H_s = x.shape[-2]

        if self.resolution in resolution_to_patches:
            target_size = resolution_to_patches[self.resolution]
            if _H_s != target_size and current_H_s == target_size:
                x = upsample(x, _H_s, _W_s)
        return x


class Spiking_vit_MetaFormer_Spike_SepConv(nn.Module):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=16,
            in_channels=2,
            num_classes=11,
            embed_dim=[64, 128, 256],
            num_heads=[1, 2, 4],
            mlp_ratios=[4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=[6, 8, 6],
            sr_ratios=[8, 4, 2],
            template_mode=None,
            resolution=None,
            temp_num=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,

        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    resolution=resolution,

                )
                for j in range(6)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,

        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    resolution=resolution,

                )
                for j in range(2)
            ]
        )
        self.apply(self._init_weights)

        '''  declared for spiketrack '''
        embed_dim_indices = [0, 1, 2, 2, 3, 3]

        self.template_mode = template_mode  # 1. stack, 2. siamese-like, 3. cap (concatenated and padding)

        self.mrm = nn.ModuleList([
            MemoryRetrieval(embed_dim=embed_dim[i], mlp_ratios=mlp_ratios, resolution=resolution, temp_num=temp_num)
            for i in embed_dim_indices
        ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        templates, search = self.get_T(x, self.template_mode)

        templates = self.downsample1_1(templates)
        search = self.downsample1_1(search)

        for blk in self.ConvBlock1_1:
            templates = blk(templates)
            search = blk(search)

        templates = self.downsample1_2(templates)
        search = self.downsample1_2(search)

        search = self.mrm[0](templates, search)

        for blk in self.ConvBlock1_2:
            templates = blk(templates)
            search = blk(search)

        templates = self.downsample2(templates)
        search = self.downsample2(search)

        search = self.mrm[1](templates, search)

        for blk in self.ConvBlock2_1:
            templates = blk(templates)
            search = blk(search)

        for blk in self.ConvBlock2_2:
            templates = blk(templates)
            search = blk(search)

        templates = self.downsample3(templates)
        search = self.downsample3(search)

        search = self.mrm[2](templates, search)


        for i, blk in enumerate(self.block3):
            templates = blk(templates, 'template')
            search = blk(search, 'search')

            if i == 2:
                search = self.mrm[3](templates, search)

        templates = self.downsample4(templates)
        search = self.downsample4(search)

        search = self.mrm[4](templates, search)

        for i, blk in enumerate(self.block4):
            templates = blk(templates, 'template')
            search = blk(search, 'search')

        search = self.mrm[5](templates, search)

        return search  # T,B,C,H,W

    def forward(self, x):

        x = self.forward_features(x)

        return x

    def get_T(self, x, template_mode):

        templates = x[0:-1]
        search = x[-1].unsqueeze(0)
        if len(templates) > 1:
            templates = torch.stack(templates, dim=0)
        else:
            templates = templates[0].unsqueeze(0)

        return templates, search


def Efficient_Spiking_Transformer_l(**kwargs):
    # 19.0M
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def Efficient_Spiking_Transformer_m(**kwargs):
    # 10.0M
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[48, 96, 192, 240],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def Efficient_Spiking_Transformer_s(**kwargs):
    # 5.1M
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[32, 64, 128, 192],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def Efficient_Spiking_Transformer_t(**kwargs):
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[24, 48, 96, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


def build_backbone(cfg):
    if cfg.MODEL.ENCODER.TYPE == 'Efficient_Spiking_Transformer_t':
        model = Efficient_Spiking_Transformer_t(template_mode=cfg.MODEL.TEMPLATE_MODE,
                                                resolution=cfg.DATA.SEARCH.SIZE,
                                                temp_num=cfg.DATA.TEMPLATE.NUMBER)
        pretrained_state = torch.load('.', map_location='cpu')['model']

    elif cfg.MODEL.ENCODER.TYPE == 'Efficient_Spiking_Transformer_s':
        model = Efficient_Spiking_Transformer_s(template_mode=cfg.MODEL.TEMPLATE_MODE,
                                                resolution=cfg.DATA.SEARCH.SIZE,
                                                temp_num=cfg.DATA.TEMPLATE.NUMBER)
        pretrained_state = torch.load('./pretrained_models/V3_5.1M_1x4.pth', map_location='cpu')['model']

    elif cfg.MODEL.ENCODER.TYPE == 'Efficient_Spiking_Transformer_m':
        model = Efficient_Spiking_Transformer_m(template_mode=cfg.MODEL.TEMPLATE_MODE,
                                                resolution=cfg.DATA.SEARCH.SIZE,
                                                temp_num=cfg.DATA.TEMPLATE.NUMBER)

    elif cfg.MODEL.ENCODER.TYPE == 'Efficient_Spiking_Transformer_l':
        model = Efficient_Spiking_Transformer_l(template_mode=cfg.MODEL.TEMPLATE_MODE,
                                                resolution=cfg.DATA.SEARCH.SIZE,
                                                temp_num=cfg.DATA.TEMPLATE.NUMBER)
        pretrained_state = torch.load('/pretrained_models/V3_19.0M_1x4.pth', map_location='cpu')['model']

    else:
        raise ValueError('Unknown model type: {}'.format(cfg.MODEL.TYPE))

    model_state = model.state_dict()
    filtered_state = {}

    for k, v in pretrained_state.items():
       if k in model_state and model_state[k].shape == v.shape:
           filtered_state[k] = v

    result = model.load_state_dict(filtered_state, strict=False)

    return model


