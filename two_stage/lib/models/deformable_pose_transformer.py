from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from models.deformable_transformer import build_deformable_transformer, inverse_sigmoid
from models.backbone import build_backbone
from models.pose_transformer import MLP

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeformablePoseTransformer(nn.Module):

    def __init__(self, cfg, backbone, transformer, **kwargs):
        super(DeformablePoseTransformer, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.num_queries = extra.NUM_QUERIES
        self.num_classes = cfg.MODEL.NUM_JOINTS
        self.num_feature_levels = extra.NUM_FEATURE_LEVELS
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.kpt_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim * 2)

        if self.num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes + 1) * bias_value
        nn.init.constant_(self.kpt_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.kpt_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.kpt_embed = nn.ModuleList([self.kpt_embed for _ in range(num_pred)])
        self.aux_loss = extra.AUX_LOSS

    def forward(self, x):
        features, pos = self.backbone(x)

        srcs = []
        for l, feat in enumerate(features):
            srcs.append(self.input_proj[l](feat))
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                pos_l = self.backbone[1](src).to(src.dtype)
                srcs.append(src)
                pos.append(pos_l)
        masks = [src.new_zeros(src.shape[0], src.shape[2], src.shape[3], dtype=torch.bool) for src in srcs]

        hs, init_reference, inter_references = self.transformer(srcs, masks, pos, self.query_embed.weight)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.kpt_embed[lvl](hs[lvl])
            tmp += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

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

    transformer = build_deformable_transformer(hidden_dim=extra.HIDDEN_DIM, dropout=extra.DROPOUT, nheads=extra.NHEADS, dim_feedforward=extra.DIM_FEEDFORWARD,
                                               enc_layers=extra.ENC_LAYERS, dec_layers=extra.DEC_LAYERS, num_feature_levels=extra.NUM_FEATURE_LEVELS,
                                               enc_n_points=extra.ENC_N_POINTS, dec_n_points=extra.DEC_N_POINTS)

    pretrained = is_train and cfg.MODEL.INIT_WEIGHTS
    backbone = build_backbone(cfg, pretrained)
    model = DeformablePoseTransformer(cfg, backbone, transformer, **kwargs)

    return model
