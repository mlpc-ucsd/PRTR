from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.transformer import build_transformer
from models.backbone import build_backbone

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PoseTransformer(nn.Module):

    def __init__(self, cfg, backbone, transformer, **kwargs):
        super(PoseTransformer, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.num_queries = extra.NUM_QUERIES
        self.num_classes = cfg.MODEL.NUM_JOINTS
        self.transformer = transformer
        self.backbone = backbone
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.kpt_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels[0], hidden_dim, 1)

        self.aux_loss = extra.AUX_LOSS

    def forward(self, x):
        src, pos = self.backbone(x)
        hs = self.transformer(self.input_proj(src[-1]), None,
                              self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.kpt_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1],
               'pred_coords': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_coords': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def get_pose_net(cfg, is_train, **kwargs):
    extra = cfg.MODEL.EXTRA

    transformer = build_transformer(hidden_dim=extra.HIDDEN_DIM, dropout=extra.DROPOUT, nheads=extra.NHEADS, dim_feedforward=extra.DIM_FEEDFORWARD,
                                    enc_layers=extra.ENC_LAYERS, dec_layers=extra.DEC_LAYERS, pre_norm=extra.PRE_NORM)
    pretrained = is_train and cfg.MODEL.INIT_WEIGHTS
    backbone = build_backbone(cfg, pretrained)
    model = PoseTransformer(cfg, backbone, transformer, **kwargs)

    return model
