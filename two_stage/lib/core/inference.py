# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.transforms import transform_preds, fliplr_joints


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def get_final_preds_match(config, outputs, center, scale, flip_pairs=None):
    pred_logits = outputs['pred_logits'].detach().cpu()
    pred_coords = outputs['pred_coords'].detach().cpu()

    num_joints = pred_logits.shape[-1] - 1

    if config.TEST.INCLUDE_BG_LOGIT:
        prob = F.softmax(pred_logits, dim=-1)[..., :-1]
    else:
        prob = F.softmax(pred_logits[..., :-1], dim=-1)

    score_holder = []
    coord_holder = []
    orig_coord = []
    for b, C in enumerate(prob):
        _, query_ind = linear_sum_assignment(-C.transpose(0, 1)) # Cost Matrix: [17, N]
        score = prob[b, query_ind, list(np.arange(num_joints))][..., None].numpy()
        pred_raw = pred_coords[b, query_ind].numpy()
        if flip_pairs is not None:
            pred_raw, score = fliplr_joints(pred_raw, score, 1, flip_pairs, pixel_align=False, is_vis_logit=True)
        # scale to the whole patch
        pred_raw *= np.array(config.MODEL.IMAGE_SIZE)
        # transform back w.r.t. the entire img
        pred = transform_preds(pred_raw, center[b], scale[b], config.MODEL.IMAGE_SIZE)
        orig_coord.append(pred_raw)
        score_holder.append(score)
        coord_holder.append(pred)
    
    matched_score = np.stack(score_holder)
    matched_coord = np.stack(coord_holder)

    return matched_coord, matched_score, np.stack(orig_coord)