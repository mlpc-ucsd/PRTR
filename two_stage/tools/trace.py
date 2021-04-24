from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count

import _init_paths
from config import cfg
from config import update_config
from utils.utils import create_logger
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Trace keypoints network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()
    return args

def get_model_stats(model, cfg, mode):
    assert mode in ['flop', 'activation']
    if mode == "flop":
        model_stats_fun = flop_count
    else:
        model_stats_fun = activation_count

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    inputs = (torch.rand(1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]).cuda(), )
    count_dict, *_ = model_stats_fun(model, inputs)
    count = sum(count_dict.values())
    model.train(model_mode)
    return count


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'trace')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    ).cuda()

    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info(
        "Flops: {:,} G".format(
            get_model_stats(model, cfg, "flop")
        )
    )
    logger.info(
        "Activations: {:,} M".format(
            get_model_stats(model, cfg, "activation")
        )
    )

if __name__ == '__main__':
    main()
