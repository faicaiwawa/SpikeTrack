
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from lib.models.spiketrack.ni_lif import mem_update

from lib.models.spiketrack.fuc import GateModule,downsample,upsample,FrozenBatchNorm2d,FrozenBatchNorm1d

decay = 0.25

resolution_to_patches = {256: 16, 384: 24}

def make_conv_layer(in_channels, out_channels, norm_layer,
                     kernel_size=1, stride=1, padding=0, groups=1, bias=False):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding=padding, groups=groups, bias=bias),
        norm_layer(out_channels)
    )

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
        # B, C, H, W = x.shape
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
            self, in_features, hidden_features=None, out_features=None, frozen = True, drop=0.0, layer=0,
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
            resolution = 384
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


        if resolution not in resolution_to_patches:
            raise ValueError(f"Unsupported resolution: {resolution}")

        patches = resolution_to_patches[resolution]
        patch_size = patches * patches

        self.s_pos_embed = nn.Parameter(torch.zeros(1, dim, patch_size))
        self.t_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim, patch_size))
            for _ in range(4)
        ])


    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self._apply_positional_encoding(x, 'template', T)

        x = self.head_spike(x)
        x = x.view(T * B, C, H, W)
        q = self.q_conv(x).reshape(T, B, C, H * W)
        k = self.k_conv(x).reshape(T, B, C, H * W)
        v = self.v_conv(x).reshape(T, B, C_v, H * W)

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

    def _apply_positional_encoding(self, x, branch, T=None):

        T, B, C, H, W = x.shape
        N = H * W
        x = x.view(T, B, C, N)
        if branch == 'search':
            x = x + self.s_pos_embed
        elif branch == 'template':
            if T is None:
                raise ValueError("T must be provided for template branch")

            out_list = [
                x[t] + self.t_pos_embeds[min(t, 3)]
                for t in range(T)
            ]
            x =  torch.stack(out_list, dim=0)

        x = x.view(T, B, C, H, W)
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
            resolution = 384
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


        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = mem_update(time_step=4, skip_ts=False)

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim*lamda_ratio) , 1, 1, bias=False),
                                    nn.BatchNorm2d(int(dim*lamda_ratio)))

        self.v_spike = mem_update(time_step=4, skip_ts=False)

        if resolution not in resolution_to_patches:
            raise ValueError(f"Unsupported resolution: {resolution}")

        patches = resolution_to_patches[resolution]
        patch_size = patches * patches

        self.t_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim, patch_size))
            for _ in range(4)
        ])
    def forward(self, template):

        template = template.clone()
        _, _, _, _H_t, _W_t = template.shape

        template = self._preprocess_inputs(template)

        T, B, C, H_t, W_t = template.shape
        N_t =H_t * W_t


        C_v = int(C * self.lamda_ratio)

        template = self._apply_positional_encoding(template, 'template', T=T)

        template = self.head_spike(template)

        k_t = self.k_conv(template.flatten(0, 1)).reshape(T, B, C, H_t, W_t)
        v_t = self.v_conv(template.flatten(0, 1)).reshape(T, B, C_v, H_t, W_t)

        v_t = self.v_spike(v_t).flatten(3)
        k_t = self.k_spike(k_t).flatten(3)


        k = (
            k_t.transpose(-1, -2)
            .reshape(T, B, N_t, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = (
            v_t.transpose(-1, -2)
            .reshape(T , B, N_t, self.num_heads, C_v // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        return x

    def _preprocess_inputs(self, template):

        _, _, _, _H_s, _W_s = template.shape

        if self.resolution in resolution_to_patches:
            target_size = resolution_to_patches[self.resolution]
            if _H_s != target_size:
                template = downsample(template, target_size, target_size)

        return template

    def _apply_positional_encoding(self, x, branch, T=None):

        T, B, C, H, W = x.shape

        x = x.view(T, B, C, int(H*W))

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
            resolution=resolution
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop,frozen=frozen)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x + self.conv(x)

        x= x + self.attn(x)

        x = x + self.mlp(x)

        return x

class MemoryRetrieval(nn.Module):
    def __init__(self, embed_dim, mlp_ratios,resolution):
        super().__init__()
        self.retriever = Retriever(dim=embed_dim,resolution=resolution)

    def forward(self, templates):
        x  = self.retriever(templates)

        return x

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
            time_step=None,
            resolution=None,
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
                    resolution = resolution,
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
                    resolution = resolution,
                )
                for j in range(2)
            ]
        )

        self.apply(self._init_weights)
        embed_dim_indices = [0, 1, 2, 2, 3, 3]
        self.template_mode = template_mode  # 1. stack, 2. siamese-like, 3. pac (padding and cat)
        # print('template_mode:', self.template_mode)
        self.mrm = nn.ModuleList([
            MemoryRetrieval(embed_dim=embed_dim[i], mlp_ratios=mlp_ratios, resolution=resolution)
            for i in embed_dim_indices
        ])
        #self.register_spike_rate_hooks()
        self.current_image_idx = 0
        self.spike_rate_dict_temp =  {}

    def register_spike_rate_hooks(self):
        for name, module in self.named_modules():
            if isinstance(module, mem_update):

                def hook_fn(module, input, output, name=name):

                    if output.shape[0] > 4:
                        output = output.transpose( 0, 2)

                    spatial_dims = tuple(range(1, output.ndim))
                    firing_rates = output.float().mean(dim=spatial_dims) * 4

                    firing_rates = firing_rates.cpu().tolist()

                    image_key = f'image_{self.current_image_idx}'
                    self.spike_rate_dict_temp.setdefault(name, {}).setdefault(image_key, []).extend(firing_rates)

                module.register_forward_hook(hook_fn)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        templates = self.downsample1_1(x)

        for blk in self.ConvBlock1_1:
            templates = blk(templates)

        templates = self.downsample1_2(templates)

        cross_block_0 = self.mrm[0](templates)

        for blk in self.ConvBlock1_2:
            templates = blk(templates)

        templates = self.downsample2(templates)

        cross_block_1 = self.mrm[1](templates)

        for blk in self.ConvBlock2_1:
            templates = blk(templates)


        for blk in self.ConvBlock2_2:
            templates = blk(templates)

        templates = self.downsample3(templates)

        cross_block_2 = self.mrm[2](templates)

        for i, blk in enumerate(self.block3):
            templates = blk(templates)

            if i == 2:
                cross_block_3 =  self.mrm[3](templates)

        templates = self.downsample4(templates)


        cross_block_4 =  self.mrm[4](templates)

        for i, blk in enumerate(self.block4):
            templates = blk(templates)

        cross_block_5 =  self.mrm[5](templates)

        cache_dict = {'cross_block_0': cross_block_0,

                      'cross_block_1': cross_block_1,

                      'cross_block_2': cross_block_2,

                      'cross_block_3': cross_block_3,

                      'cross_block_4': cross_block_4,

                      'cross_block_5': cross_block_5,
                      }

        self.reset_image_counter()
        
        if len(self.spike_rate_dict_temp) > 0:
            spike_rate_dict_temp = self.spike_rate_dict_temp
        else:
            spike_rate_dict_temp = None

        return cache_dict, spike_rate_dict_temp

    def reset_image_counter(self):
        self.current_image_idx += 1

    def forward(self, x):

        cache_dict, spike_rate_dict_temp = self.forward_features(x)

        return cache_dict, spike_rate_dict_temp

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


def build_backbone_temp(cfg):

    if cfg.MODEL.ENCODER.TYPE == 'Efficient_Spiking_Transformer_t':
        model = Efficient_Spiking_Transformer_t(template_mode=cfg.MODEL.TEMPLATE_MODE, resolution = cfg.DATA.SEARCH.SIZE)
    elif cfg.MODEL.ENCODER.TYPE == 'Efficient_Spiking_Transformer_s':
        model = Efficient_Spiking_Transformer_s(template_mode=cfg.MODEL.TEMPLATE_MODE,resolution = cfg.DATA.SEARCH.SIZE)
    elif cfg.MODEL.ENCODER.TYPE == 'Efficient_Spiking_Transformer_m':
        model = Efficient_Spiking_Transformer_m(template_mode=cfg.MODEL.TEMPLATE_MODE)
    elif cfg.MODEL.ENCODER.TYPE == 'Efficient_Spiking_Transformer_l':
        model = Efficient_Spiking_Transformer_l(template_mode=cfg.MODEL.TEMPLATE_MODE, resolution = cfg.DATA.SEARCH.SIZE)
    else:
        raise ValueError('Unknown model type: {}'.format(cfg.MODEL.TYPE))

    return model


