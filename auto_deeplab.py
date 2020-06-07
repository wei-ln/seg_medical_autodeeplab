import torch.nn as nn
import cell_level_search
from genotypes import PRIMITIVES
import torch.nn.functional as F
from operations import *
from decoding_formulas import Decoder


class AutoDeeplab(nn.Module):
    def __init__(self, num_classes, num_layers, criterion=None, filter_multiplier=8, block_multiplier=5, step=5, cell=cell_level_search.Cell):
        super(AutoDeeplab, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
        self._initialize_alphas_betas()
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)

        self.stem0 = nn.Sequential(
            nn.Conv2d(1, half_f_initial * self._block_multiplier, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )#改变channel数目，并且减小分辨率
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_f_initial * self._block_multiplier, half_f_initial * self._block_multiplier, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )#什么都不改变，只是学习一下
        self.stem2 = nn.Sequential(
            nn.Conv2d(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(f_initial * self._block_multiplier),
            nn.ReLU()
        )#减半分辨率，channel翻倍

        # intitial_fm = C_initial
        for i in range(self._num_layers):

            if i == 0:#第零层的cell，也就是[4,1],[8,1]
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier)
                cell2 = cell(self._step, self._block_multiplier, -1,
                             f_initial, None, None,
                             self._filter_multiplier * 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None,#same来自[8,1]，channel数要翻倍
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None,
                             self._filter_multiplier * 4)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, None, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.aspp_4 = nn.Sequential(
            ASPP(self._filter_multiplier * self._block_multiplier, self._num_classes, 24, 24)  # 96 / 4 as in the paper
        )
        self.aspp_8 = nn.Sequential(
            ASPP(self._filter_multiplier * 2 * self._block_multiplier, self._num_classes, 12, 12)  # 96 / 8
        )
        self.aspp_16 = nn.Sequential(
            ASPP(self._filter_multiplier * 4 * self._block_multiplier, self._num_classes, 6, 6)  # 96 / 16
        )
        self.aspp_32 = nn.Sequential(
            ASPP(self._filter_multiplier * 8 * self._block_multiplier, self._num_classes, 3, 3)  # 96 / 32
        )

    def forward(self, x):
        # TODO: GET RID OF THESE LISTS, we dont need to keep everything.
        # TODO: Is this the reason for the memory issue ?
        self.level_4 = []
        self.level_8 = []
        self.level_16 = []
        self.level_32 = []
        temp = self.stem0(x)
        temp = self.stem1(temp)
        self.level_4.append(self.stem2(temp))#走到了图中layer0，分辨率4这个node
        #paper中fig1中如何表述每个node [分辨率,层数]，比如右下角那个白圈就是[4,0]
        count = 0
        normalized_betas = torch.randn(self._num_layers, 4, 3).cuda()#初始化beta，每层4个node，默认每个node三个输出右上edge，右edge和右下edge
        # Softmax on alphas and betas
        if torch.cuda.device_count() > 1:

            img_device = torch.device('cuda', x.get_device())
            normalized_alphas = F.softmax(self.alphas.to(device=img_device), dim=-1)#计算cell内部如何连接，采用什么op，确定了一个结构出来了已经

            # normalized_betas[layer][ith node][0 : ➚, 1: ➙, 2 : ➘]
            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    #node[4,0]只有右下两条边，所以是[1：]
                    #为什么需要乘2/3,只有两条边平衡一下。
                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    #第一层有两个节点，第一行语句是node[4，1]，第二行是[8,1]
                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                    #这个同理，对照图中的[4,2],[8,2],[16,2]
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:1].to(device=img_device), dim=-1) * (2/3)

        else:#因为device数目不同，所以需要写两套来应对这个，这里适用于只有一个device
            normalized_alphas = F.softmax(self.alphas, dim=-1)

            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:2], dim=-1) * (2/3)


        for layer in range(self._num_layers):

            if layer == 0:#也就是[4,1]和 [8,1]这两个no
                level4_new, = self.cells[count](None, None, self.level_4[-1], None, normalized_alphas)
                count += 1
                level8_new, = self.cells[count](None, self.level_4[-1], None, None, normalized_alphas)
                count += 1
                #self.cells返回三个输出，分别用的是L-1层得左上，左，左下node和L-2层(左左节点)得输入得到得输出，用β权重组合起来得到新的embedding
                level4_new = normalized_betas[layer][0][1] * level4_new
                level8_new = normalized_betas[layer][0][2] * level8_new
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)

            elif layer == 1:
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2 = self.cells[count](None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               None,
                                                               normalized_alphas)
                count += 1
                # level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][2] * level8_new_2
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2
                #这里写错了，是[0,2]和 [1,1]
                level16_new, = self.cells[count](None,
                                                 self.level_8[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level16_new = normalized_betas[layer][1][2] * level16_new
                count += 1

                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)

            elif layer == 2:
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2 = self.cells[count](None,
                                                                 self.level_8[-1],
                                                                 self.level_16[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2

                level32_new, = self.cells[count](None,
                                                 self.level_16[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level32_new = normalized_betas[layer][2][2] * level32_new
                count += 1

                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                self.level_32.append(level32_new)

            elif layer == 3:
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](self.level_16[-2],
                                                                                self.level_8[-1],
                                                                                self.level_16[-1],
                                                                                self.level_32[-1],
                                                                                normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][
                    0] * level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](None,#没有左左node
                                                                 self.level_16[-1],
                                                                 self.level_32[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2

                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                self.level_32.append(level32_new)

            else:
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas)
                count += 1

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](self.level_16[-2],
                                                                                self.level_8[-1],
                                                                                self.level_16[-1],
                                                                                self.level_32[-1],
                                                                                normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][
                    0] * level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](self.level_32[-2],
                                                                 self.level_16[-1],
                                                                 self.level_32[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2

                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                self.level_32.append(level32_new)

            self.level_4 = self.level_4[-2:]
            self.level_8 = self.level_8[-2:]
            self.level_16 = self.level_16[-2:]
            self.level_32 = self.level_32[-2:]

        aspp_result_4 = self.aspp_4(self.level_4[-1])
        aspp_result_8 = self.aspp_8(self.level_8[-1])
        aspp_result_16 = self.aspp_16(self.level_16[-1])
        aspp_result_32 = self.aspp_32(self.level_32[-1])

        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_4 = upsample(aspp_result_4)
        aspp_result_8 = upsample(aspp_result_8)
        aspp_result_16 = upsample(aspp_result_16)
        aspp_result_32 = upsample(aspp_result_32)

        sum_feature_map = aspp_result_4 + aspp_result_8 + aspp_result_16 + aspp_result_32

        return sum_feature_map

    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        #类似darts中的mixop，算一个一个block有多少条连接node的边，
        num_ops = len(PRIMITIVES)
        # alphas = torch.tensor(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)
        betas = (1e-3 * torch.randn(self._num_layers, 4, 3)).clone().detach().requires_grad_(True)

        self._arch_parameters = [
            alphas,
            betas,
        ]
        self._arch_param_names = [
            'alphas',
            'betas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]

    # def decode_viterbi(self):
    #     decoder = Decoder(self.bottom_betas, self.betas8, self.betas16, self.top_betas)
    #     return decoder.viterbi_decode()

    # def decode_dfs(self):
    #     decoder = Decoder(self.bottom_betas, self.betas8, self.betas16, self.top_betas)
    #     return decoder.dfs_decode()

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)


def main():
    model = AutoDeeplab(7, 12, None)
    x = torch.tensor(torch.ones(4, 3, 224, 224))
    resultdfs = model.decode_dfs()
    resultviterbi = model.decode_viterbi()[0]

    print(resultviterbi)
    print(model.genotype())


if __name__ == '__main__':
    main()

