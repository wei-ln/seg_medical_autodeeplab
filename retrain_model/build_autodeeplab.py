import numpy as np
import torch.nn as nn
import torch.distributed as dist

from operations import NaiveBN
from retrain_model.aspp import ASPP
from retrain_model.decoder import Decoder
from retrain_model.new_model import get_default_arch, newModel, network_layer_to_space


class Retrain_Autodeeplab(nn.Module):
    def __init__(self, args):
        super(Retrain_Autodeeplab, self).__init__()
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}#不同分辨率层之间的channel翻倍数，分辨率减半channel翻倍
        BatchNorm2d = NaiveBN
        if (not args.dist and args.use_ABN) or (args.dist and args.use_ABN and dist.get_rank() == 0):
            print("=> use ABN!")
            #args.net_arch : network_path.npy [1 2 2 2 1 1 1 1]
            #args.cell-arch: genotype.npy: [10,2]的矩阵
        if args.net_arch is not None and args.cell_arch is not None:
            net_arch, cell_arch = np.load(args.net_arch), np.load(args.cell_arch)
            network_arch = network_layer_to_space(net_arch)
            network_path = net_arch
        else:
            network_arch, cell_arch, network_path = get_default_arch()
        self.encoder = newModel(network_arch, cell_arch,net_arch, args.num_classes, 8, args.filter_multiplier, BatchNorm=BatchNorm2d, args=args)
        self.aspp = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[-1]],
                         256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        self.decoder = Decoder(args.num_classes, filter_multiplier=args.filter_multiplier * args.block_multiplier,
                               args=args, last_level=network_path[-1])

    def forward(self, x):
        encoder_output = self.encoder(x)
        # high_level_feature = self.aspp(encoder_output)#不改变分辨率
        # decoder_output = self.decoder(high_level_feature, low_level_feature)
        return nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(encoder_output)

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params

