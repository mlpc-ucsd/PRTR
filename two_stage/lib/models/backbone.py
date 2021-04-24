import logging
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from models.hrnet import HighResolutionNet
from models.positional_encoding import build_position_encoding

logger = logging.getLogger(__name__)


class FrozenBatchNorm2d(nn.Module):
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
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)

    def forward(self, x):
        xs = self.body(x)
        res = []
        for _, x in xs.items():
            res.append(x)
        return res


class ResNetBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 pretrained: bool,
                 dilation: bool = False):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
        logger.info(f'=> Loading backbone, pretrained: {pretrained}')
        assert name in ['resnet50',
                        'resnet101'], "Number of channels is hard-coded"
        num_channels = 2048
        super().__init__(backbone, train_backbone, return_interm_layers)
        self.num_channels = num_channels
        if return_interm_layers:
            self.num_channels = [512, 1024, 2048]
        else:
            self.num_channels = [2048]


class HRNetBackbone(nn.Module):
    def __init__(self, cfg, return_interm_layers: bool, pretrained: bool = False):
        super().__init__()
        if return_interm_layers:
            raise NotImplementedError(
                "HRNet backbone does not support return interm layers")
        else:
            self.num_channels = [2048]
        self.body = HighResolutionNet(cfg)
        if pretrained:
            self.body.init_weights(cfg.MODEL.PRETRAINED)

    def forward(self, x):
        y = self.body(x)
        return y


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: Tensor):
        out = self[0](tensor_list)
        pos = []

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(cfg, pretrained):
    extra = cfg.MODEL.EXTRA
    num_layers = extra.NUM_LAYERS
    if type(num_layers) == str:
        name = num_layers
    else:
        name = f'resnet{num_layers}'
    position_embedding = build_position_encoding(
        extra.HIDDEN_DIM, extra.POS_EMBED_METHOD)
    return_interm_layers = hasattr(
        extra, 'NUM_FEATURE_LEVELS') and extra.NUM_FEATURE_LEVELS > 1
    if name.startswith('resnet'):
        backbone = ResNetBackbone(
            name, train_backbone=True, return_interm_layers=return_interm_layers, pretrained=pretrained, dilation=extra.DILATION)
    elif name == 'hrnet':
        backbone = HRNetBackbone(
            cfg, pretrained=pretrained, return_interm_layers=return_interm_layers)
    else:
        raise NotImplementedError(f'Unsupported backbone type: {name}')
    model = Joiner(backbone, position_embedding)
    return model
