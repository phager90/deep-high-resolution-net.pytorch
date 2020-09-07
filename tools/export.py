# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

#import dataset
import models

def valid_tensor(s):
    msg = "Not a valid resolution: '{0}' [CxHxW].".format(s)
    try:
        q = s.split('x')
        if len(q) != 3:
            raise argparse.ArgumentTypeError(msg)
        return [int(v) for v in q]
    except ValueError:
        raise argparse.ArgumentTypeError(msg)

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
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
    parser.add_argument('-r', '--ONNX_resolution', default="3x384x288", type=valid_tensor,
                    help='ONNX input resolution (default: 3x384x288 [coco])')
    parser.add_argument('-o', '--outfile', default='./out.onnx',
                    help='output file path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=False, )

    #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # EXPORT
        # Export ONNX file
    input_names = [ "input:0" ]  # this are our standardized in/out nameing (required for runtime)
    output_names = [ "output:0" ]
    dummy_input = torch.randn([1]+args.ONNX_resolution)
    ONNX_path = args.outfile
    # Exporting -- CAFFE2 compatible
    # requires operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    # https://github.com/pytorch/pytorch/issues/41848
    # for CAFFE2 backend (old exports mode...)
    #torch.onnx.export(model, dummy_input, ONNX_path, input_names=input_names, output_names=output_names, 
    #    keep_initializers_as_inputs=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    # Exporting -- ONNX runtime compatible
    #   keep_initializers_as_inputs=True -> is required for onnx optimizer...
    torch.onnx.export(model, dummy_input, ONNX_path, input_names=input_names, output_names=output_names,
        keep_initializers_as_inputs=True, opset_version=11)

if __name__ == '__main__':
    main()
