import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.spiketrack.ni_lif import mem_update
decay = 0.25

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
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            )




class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16,
                 conv_type='normal', freeze_bn=False, xavier_init=True):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        if conv_type == "normal":
            kernel_size = 3
            padding = 1
        elif conv_type == "small":
            kernel_size = 1
            padding = 0
        else:
            raise NotImplementedError('cfg.MODEL.DECODER.CONV_TYPE must be choosen from "normal" and "small".')
        # self.use_weight_mlp = True
        # if self.use_weight_mlp:
        #     self.weight_mlp = MS_MLP_Head(2)

        # corner predict
        #print("deocder inplanes: {}".format(inplanes))
        #print("deocder start channel : {}".format( channel))
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        self.ctr1_spike= mem_update(time_step=4)
        self.ctr2_spike = mem_update(time_step=4)
        self.ctr3_spike = mem_update(time_step=4)
        self.ctr4_spike = mem_update(time_step=4)

        self.size1_spike= mem_update(time_step=4)
        self.size2_spike = mem_update(time_step=4)
        self.size3_spike = mem_update(time_step=4)
        self.size4_spike = mem_update(time_step=4)

        self.offset1_spike= mem_update(time_step=4)
        self.offset2_spike = mem_update(time_step=4)
        self.offset3_spike = mem_update(time_step=4)
        self.offset4_spike = mem_update(time_step=4)
        self.spike = mem_update(time_step=4)
        if xavier_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):

        """ Forward pass with input x. """
        #x = x.squeeze(0)
        t,bs, _, _,_ = x.shape
        x = self.spike(x)
        score_map_ctr, size_map, offset_map = self.get_score_map(x) # x: torch.Size([b, c, h, w])
        score_map_ctr, size_map, offset_map = score_map_ctr.mean(0), size_map.mean(0), offset_map.mean(0)
        if gt_score_map is None:
            bbox, score = self.cal_bbox(score_map_ctr, size_map, offset_map, True)
        else:
            bbox, score = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map, True)

        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, 1, 4)
        out = {'pred_boxes': outputs_coord_new,
               'score_map': score_map_ctr,
               'size_map': size_map,
               'offset_map': offset_map}
        return out

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True) # score_map_ctr.flatten(1): torch.Size([32, 256]) idx: torch.Size([32, 1]) max_score: torch.Size([32, 1])
        idx_y = torch.div(idx, self.feat_sz, rounding_mode='floor')
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx) # size_map: torch.Size([32, 2, 16, 16])  size_map.flatten(2): torch.Size([32, 2, 256])
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        T, B, C, H, W = x.shape
        # ctr branch
        x_ctr1 = self.conv1_ctr(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_ctr1 = self.ctr1_spike(x_ctr1)
        x_ctr2 = self.conv2_ctr(x_ctr1.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_ctr2 = self.ctr2_spike(x_ctr2)
        x_ctr3 = self.conv3_ctr(x_ctr2.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_ctr3 = self.ctr3_spike(x_ctr3)
        x_ctr4 = self.conv4_ctr(x_ctr3.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_ctr4 = self.ctr4_spike(x_ctr4)
        score_map_ctr = self.conv5_ctr(x_ctr4.flatten(0, 1)).reshape(T, B, -1, H, W)

        # offset branch
        x_offset1 = self.conv1_offset(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_offset1 = self.offset1_spike(x_offset1)
        x_offset2 = self.conv2_offset(x_offset1.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_offset2 = self.offset2_spike(x_offset2)
        x_offset3 = self.conv3_offset(x_offset2.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_offset3 = self.offset3_spike(x_offset3)
        x_offset4 = self.conv4_offset(x_offset3.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_offset4 = self.offset4_spike(x_offset4)
        score_map_offset = self.conv5_offset(x_offset4.flatten(0, 1)).reshape(T, B, -1, H, W)

        # size branch
        x_size1 = self.conv1_size(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_size1 = self.size1_spike(x_size1)
        x_size2 = self.conv2_size(x_size1.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_size2 = self.size2_spike(x_size2)
        x_size3 = self.conv3_size(x_size2.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_size3 = self.size3_spike(x_size3)
        x_size4 = self.conv4_size(x_size3.flatten(0, 1)).reshape(T, B, -1, H, W)
        x_size4 = self.size4_spike(x_size4)
        score_map_size = self.conv5_size(x_size4.flatten(0, 1)).reshape(T, B, -1, H, W)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

class MLPPredictor(nn.Module):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(MLPPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        self.num_layers = 3
        h = [channel] * (self.num_layers - 1)
        self.layers_cls = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([inplanes] + h, h + [1]))
        self.layers_reg = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([inplanes] + h, h + [4]))


    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), offset_map)

        return score_map, bbox, offset_map

    def cal_bbox(self, score_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map.flatten(1), dim=1, keepdim=True)
        idx_y = torch.div(idx, self.feat_sz, rounding_mode='floor')
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 4, 1) # torch.Size([32, 4, 1])
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        # offset: (l,t,r,b)

        # x1, y1, x2, y2
        bbox = torch.cat([idx_x.to(torch.float) / self.feat_sz - offset[:, :1], # the offset should not divide the self.feat_sz, since I use the sigmoid to limit it in (0,1)
                          idx_y.to(torch.float) / self.feat_sz - offset[:, 1:2],
                          idx_x.to(torch.float) / self.feat_sz + offset[:, 2:3],
                          idx_y.to(torch.float) / self.feat_sz + offset[:, 3:4],
                          ], dim=1)
        bbox = box_xyxy_to_cxcywh(bbox)
        if return_score:
            return bbox, max_score
        return bbox

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        x_cls = x
        for i, layer in enumerate(self.layers_cls):
            x_cls = F.relu(layer(x_cls)) if i < self.num_layers - 1 else layer(x_cls)
        x_cls = x_cls.permute(0,2,1).reshape(-1,1,self.feat_sz,self.feat_sz)

        x_reg = x
        for i, layer in enumerate(self.layers_reg):
            x_reg = F.relu(layer(x_reg)) if i < self.num_layers - 1 else layer(x_reg)
        x_reg = x_reg.permute(0, 2, 1).reshape(-1, 4, self.feat_sz, self.feat_sz)

        return _sigmoid(x_cls), _sigmoid(x_reg)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_decoder(cfg, num_channels_enc):
    stride = cfg.MODEL.ENCODER.STRIDE
    if cfg.MODEL.DECODER.TYPE == "MLP":
        in_channel = num_channels_enc
        hidden_dim = cfg.MODEL.DECODER.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        mlp_head = MLPPredictor(inplanes=in_channel, channel=hidden_dim,
                                feat_sz=feat_sz, stride=stride)
        return mlp_head
    elif "CORNER" in cfg.MODEL.DECODER.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.DECODER.TYPE == "CENTER":
        in_channel = num_channels_enc
        out_channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        #print("head channel: %d" % out_channel)
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        conv_type = 'normal'
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride,
                                      conv_type=conv_type,
                                      xavier_init=True)
        return center_head

    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
