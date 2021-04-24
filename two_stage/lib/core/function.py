# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
from torch.utils import data

from core.evaluate import accuracy
from core.inference import get_final_preds, get_final_preds_match
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    metrics_dict = {}

    def add_to_metrics(name, val, cnt):
        if name not in metrics_dict.keys():
            metrics_dict[name] = AverageMeter()
        metrics_dict[name].update(val, cnt)

    # switch to train mode
    model.train()
    criterion.train()

    image_size = np.array(config.MODEL.IMAGE_SIZE)
    max_norm = config.TRAIN.CLIP_MAX_NORM

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        pred = None

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
            output = outputs[-1]
        elif isinstance(outputs, dict):
            output = outputs
            loss_dict, pred = criterion(outputs, target, target_weight)
            pred *= image_size
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                       for k in loss_dict.keys() if k in weight_dict)

            bs = input.size(0)
            for k, v in loss_dict.items():
                add_to_metrics(f'{k}_unscaled', v.item(), bs)
                if k in weight_dict:
                    add_to_metrics(k, (v * weight_dict[k]).item(), bs)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, speed=input.size(0)/batch_time.val, data_time=data_time
                )
            msg_extra = 'Metrics: '
            lst = list(metrics_dict.items())
            for idx, (k, meter) in enumerate(lst):
                msg_extra += f'{k}: {meter.val:.5f} ({meter.avg:.5f})'
                if idx + 1 < len(lst):
                    msg_extra += ', '
            msg += msg_extra
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred, output, prefix)
            for k, meter in metrics_dict.items():
                writer.add_scalar(k, meter.val, global_steps)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    metrics_dict = {}

    def add_to_metrics(name, val, cnt):
        if name not in metrics_dict.keys():
            metrics_dict[name] = AverageMeter()
        metrics_dict[name].update(val, cnt)

    # switch to evaluate mode
    model.eval()
    criterion.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    image_size = np.array(config.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        end = time.time()

        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            num_images = input.size(0)
            # compute output
            outputs = model(input)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            output = outputs

            loss_dict, _ = criterion(outputs, target, target_weight)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys() if k in weight_dict)

            bs = input.size(0)
            for k, v in loss_dict.items():
                add_to_metrics(f'{k}_unscaled', v.item(), bs)
                if k in weight_dict:
                    add_to_metrics(k, (v * weight_dict[k]).item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals, pred = get_final_preds_match(
                config, output, c, s
            )
            if config.TEST.FLIP_TEST:
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)
                preds_flipped, maxvals_flipped, pred_flipped = get_final_preds_match(config, outputs_flipped, c, s, val_dataset.flip_pairs)
                preds = (preds + preds_flipped) / 2
                maxvals = (maxvals + maxvals_flipped) / 2
                pred = (pred + pred_flipped) / 2

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time)
                msg_extra = 'Metrics: '
                lst = list(metrics_dict.items())
                for _, (k, meter) in enumerate(lst):
                    msg_extra += f'{k}: {meter.val:.5f} ({meter.avg:.5f})'
                    if _ + 1 < len(lst):
                        msg_extra += ', '
                msg += msg_extra
                logger.info(msg)

                prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred, output,
                                prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1
            for k, meter in metrics_dict.items():
                writer.add_scalar(f'valid/{k}', meter.val, global_steps)

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 20:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
