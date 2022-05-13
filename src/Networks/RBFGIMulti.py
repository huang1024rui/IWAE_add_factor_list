import math
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from Modules.Distribution import Gaussian2D
from Modules.Interpolation.SpatialTransformer import SpatialTransformer
from Modules.Loss import (LOSSDICT, JacobianDeterminantLoss,
                          RBFBendingEnergyLossA)
from torch import tensor

from .BaseNetwork import GenerativeRegistrationNetwork
from .ConvBlock import ConvBlock, ConvLayer


def Sample(mu, log_var):
    '''
    此处是取q(z|F,M)的值，
    :param mu: 均值，通过Encoder得到[32，164，2]
    :param log_var: 方差的log，通过Encoder得到[32，164，2]
    :return:
    采样值一个点，得到的维度是[32，164，2]
    '''
    # torch.manual_seed(0)
    eps = torch.randn(mu.size(), device=mu.device)

    std = torch.exp(0.5 * log_var)

    z = mu + std * eps
    return z


class RBFGISharedEncoderA(nn.Module):
    def __init__(self,
                 index=2,
                 dims=[16, 32, 32, 32, 32],
                 num_layers=[1, 1, 1, 1, 1],
                 local_dims=[16, 32, 32, 32, 32],
                 local_num_layers=[1, 1, 1, 1, 1]):
        super(RBFGISharedEncoderA, self).__init__()
        self.index = index
        if index >= 1:
            self.cb0 = ConvBlock(num_layers[0], dims[0], 2)  # G 128

            self.dconv0 = ConvLayer(2, local_dims[0], 3)  # L 64
            self.dcb0 = ConvBlock(local_num_layers[0], local_dims[0],
                                  local_dims[0] + dims[0])

        if index >= 2:
            self.do0 = ConvLayer(dims[0], dims[1], 3, 2, 1)  # G 64
            self.cb1 = ConvBlock(num_layers[1], dims[1], dims[1])

            self.ddo0 = ConvLayer(local_dims[0], local_dims[1], 3, 2,
                                  1)  # L 32
            self.dcb1_0 = ConvLayer(local_dims[1] + dims[1],
                                    local_dims[1],
                                    padding=0)  # L 30
            self.dcb1_1 = ConvLayer(local_dims[1], local_dims[1],
                                    padding=0)  # L 28

        if index >= 3:
            self.do1 = ConvLayer(dims[1], dims[2], 3, 2, 1)  # G 32
            self.cb2 = ConvBlock(num_layers[2], dims[2], dims[2])

            self.ddo1 = ConvLayer(local_dims[1], local_dims[2], 3, 2, 1)  #L 14
            self.dcb2 = ConvBlock(local_num_layers[2], local_dims[2],
                                  local_dims[2] + dims[2])

        if index >= 4:
            self.do2 = ConvLayer(dims[2], dims[3], 3, 2, 1)  #G 16
            self.cb3 = ConvBlock(num_layers[3], dims[3], dims[3])

            self.ddo2 = ConvLayer(local_dims[2], local_dims[3], 3,
                                  padding=0)  #L 12
            self.dcb3 = ConvBlock(local_num_layers[3], local_dims[3],
                                  local_dims[3] + dims[3])

    def forward(self, src, tgt):
        x_global = torch.cat((src, tgt), 1)
        x_local = x_global[:, :, 32:96, 32:96]

        if self.index >= 1:
            x_global = self.cb0(x_global)
            #
            dx0 = self.dconv0(x_local)
            dx0 = torch.cat([dx0, x_global[:, :, 32:96, 32:96]], 1)
            x_local = self.dcb0(dx0)

        if self.index >= 2:
            x_global = self.cb1(self.do0(x_global))
            #
            dx1 = self.ddo0(x_local)
            dx1 = torch.cat([dx1, x_global[:, :, 16:48, 16:48]], 1)
            dx1 = self.dcb1_0(dx1)  # 30
            x_local = self.dcb1_1(dx1)  # 28

        if self.index >= 3:
            x_global = self.cb2(self.do1(x_global))
            #
            dx2_0 = self.ddo1(x_local)
            dx2_1 = torch.nn.functional.interpolate(x_global[:, :, 8:24, 8:24],
                                                    [14, 14],
                                                    mode='bilinear')
            x_local = self.dcb2(torch.cat([dx2_0, dx2_1], 1))

        if self.index >= 4:
            x_global = self.cb3(self.do2(x_global))
            #
            dx3_0 = self.ddo2(x_local)
            dx3_1 = torch.nn.functional.interpolate(x_global[:, :, 4:12, 4:12],
                                                    [12, 12],
                                                    mode='bilinear')
            x_local = self.dcb3(torch.cat([dx3_0, dx3_1], 1))

        return x_global, x_local


class RBFGIUnSharedEncoderA(nn.Module):
    def __init__(self,
                 index=2,
                 dims=[16, 32, 32, 32, 32],
                 num_layers=[1, 1, 1, 1, 1],
                 local_dims=[16, 32, 32, 32, 32],
                 local_num_layers=[1, 1, 1, 1, 1]):
        super(RBFGIUnSharedEncoderA, self).__init__()
        self.index = index
        if index <= 2:
            self.do0 = ConvLayer(dims[0], dims[1], 3, 2, 1)  # G 64
            self.cb1 = ConvBlock(num_layers[1], dims[1], dims[1])

            self.ddo0 = ConvLayer(local_dims[0], local_dims[1], 3, 2,
                                  1)  # L 32
            self.dcb1_0 = ConvLayer(local_dims[1] + dims[1],
                                    local_dims[1],
                                    padding=0)  # L 30
            self.dcb1_1 = ConvLayer(local_dims[1], local_dims[1],
                                    padding=0)  # L 28

        if index <= 3:
            self.do1 = ConvLayer(dims[1], dims[2], 3, 2, 1)  # G 32
            self.cb2 = ConvBlock(num_layers[2], dims[2], dims[2])

            self.ddo1 = ConvLayer(local_dims[1], local_dims[2], 3, 2, 1)  #L 14
            self.dcb2 = ConvBlock(local_num_layers[2], local_dims[2],
                                  local_dims[2] + dims[2])

        if index <= 4:
            self.do2 = ConvLayer(dims[2], dims[3], 3, 2, 1)  #G 16
            self.cb3 = ConvBlock(num_layers[3], dims[3], dims[3])

            self.ddo2 = ConvLayer(local_dims[2], local_dims[3], 3,
                                  padding=0)  #L 12
            self.dcb3 = ConvBlock(local_num_layers[3], local_dims[3],
                                  local_dims[3] + dims[3])

        self.do3 = ConvLayer(dims[3], dims[4], 3, 2, 1)  # 8
        self.cb4 = ConvBlock(num_layers[4], dims[4], dims[4])

        self.ddo3 = ConvLayer(local_dims[3], local_dims[4], 3, padding=0)  # 10
        self.dcb4 = ConvBlock(local_num_layers[4], local_dims[4],
                              local_dims[4] + dims[4])

        # global
        self.reduce1 = ConvLayer(dims[4], 8, 3, 1, 1)  # 8
        self.qsalpha = Gaussian2D(8, 2, 3, 1, 1)
        # local
        self.reduce2 = ConvLayer(local_dims[4], 8, 3, 1, 1)  # 10
        self.qdalpha = Gaussian2D(8, 2, 3, 1, 1)

    def forward(self, x_global, x_local):
        '''
        Encoder 网络，输出mu和log_var
        :param x_global: 全局网路，64维度
        :param x_local: 局部网络。100维度
        :return: 输出mu和log_var,{list:3},每个维度为{[32,2,128,128]}
        '''
        if self.index <= 2:
            x_global = self.cb1(self.do0(x_global))
            #
            dx1 = self.ddo0(x_local)
            dx1 = torch.cat([dx1, x_global[:, :, 16:48, 16:48]], 1)
            dx1 = self.dcb1_0(dx1)  # 30
            x_local = self.dcb1_1(dx1)  # 28

        if self.index <= 3:
            x_global = self.cb2(self.do1(x_global))
            #
            dx2_0 = self.ddo1(x_local)
            dx2_1 = torch.nn.functional.interpolate(x_global[:, :, 8:24, 8:24],
                                                    [14, 14],
                                                    mode='bilinear')
            x_local = self.dcb2(torch.cat([dx2_0, dx2_1], 1))

        if self.index <= 4:
            x_global = self.cb3(self.do2(x_global))
            #
            dx3_0 = self.ddo2(x_local)
            dx3_1 = torch.nn.functional.interpolate(x_global[:, :, 4:12, 4:12],
                                                    [12, 12],
                                                    mode='bilinear')
            x_local = self.dcb3(torch.cat([dx3_0, dx3_1], 1))
        x_global = self.cb4(self.do3(x_global))

        dx4_0 = self.ddo3(x_local)
        dx4_1 = torch.nn.functional.interpolate(x_global[:, :, 2:6, 2:6],
                                                [10, 10],
                                                mode='bilinear')
        x_local = self.dcb4(torch.cat([dx4_0, dx4_1], 1))

        # global alpha
        re = self.reduce1(x_global)
        pqsalpha_mu, pqsalpha_var = self.qsalpha(re)
        pqsalpha_mu = torch.flatten(pqsalpha_mu, start_dim=2)
        pqsalpha_var = torch.flatten(pqsalpha_var, start_dim=2)
        # local alpha
        dre = self.reduce2(x_local)
        pqdalpha_mu, pqdalpha_var = self.qdalpha(dre)
        pqdalpha_mu = torch.flatten(pqdalpha_mu, start_dim=2)
        pqdalpha_var = torch.flatten(pqdalpha_var, start_dim=2)

        mu = torch.cat([pqsalpha_mu, pqdalpha_mu], 2).permute(0, 2, 1)

        var = torch.cat([pqsalpha_var, pqdalpha_var], 2).permute(0, 2, 1)

        return mu, var


class RBFGIMutilRadiusEncoderA(nn.Module):
    def __init__(self, shared_param, unshared_param, c_nums=3):
        super(RBFGIMutilRadiusEncoderA, self).__init__()
        self.shared_encoder = RBFGISharedEncoderA(**shared_param)

        self.unshared_encoder_list = nn.ModuleList(
            [RBFGIUnSharedEncoderA(**unshared_param) for _ in range(c_nums)])

    def forward(self, src, tgt):
        '''
        返回Encoder的mu 和 var, 其中前64是globle, 后100个是center, 2表示x和y方向。
        :param src: 源图像
        :param tgt: 目标图像
        :return:
        3个C上每个控制点系数\alpha的 mu 和 var, mu_list和var_list: {list:3}每个list:torch.Size([32, 164, 2])
        '''
        shared_feature = self.shared_encoder(src, tgt)
        mu_list, var_list = [], []
        for unshared_encoder in self.unshared_encoder_list:
            res = unshared_encoder(*shared_feature)
            mu_list.append(res[0])
            var_list.append(res[1])

        return mu_list, var_list


class RBFGIMutilAdaptiveDecoder(nn.Module):
    def __init__(self, img_size, c_list, int_steps=None):
        super(RBFGIMutilAdaptiveDecoder, self).__init__()

        self.scale = nn.parameter.Parameter(data=torch.tensor(
            [1, 1], dtype=torch.float), requires_grad=True)

        global_cp_loc_grid = [
            torch.linspace(s + (e - s) / 16, e - (e - s) / 16, 8)
            for s, e in ((0, 8), (0, 8))
        ]
        global_cp_loc_grid = torch.meshgrid(global_cp_loc_grid)
        global_cp_loc = torch.stack(global_cp_loc_grid, 2)[:, :, [1, 0]]
        global_cp_loc = torch.flatten(global_cp_loc, start_dim=0,
                                      end_dim=1).float()
        self.register_buffer('global_cp_loc', global_cp_loc)

        local_cp_loc_grid = [
            torch.linspace(s + (s - e) / 20, e - (s - e) / 20, 10)
            for s, e in ((2, 6), (2, 6))
        ]
        local_cp_loc_grid = torch.meshgrid(local_cp_loc_grid)
        local_cp_loc = torch.stack(local_cp_loc_grid, 2)[:, :, [1, 0]]
        local_cp_loc = torch.flatten(local_cp_loc, start_dim=0,
                                     end_dim=1).float()
        self.register_buffer('local_cp_loc', local_cp_loc)
        self.img_size = img_size
        self.c_list = c_list

        # a location mesh of output
        loc_vectors = [torch.linspace(0.0, 8.0, i_s) for i_s in self.img_size]
        loc = torch.meshgrid(loc_vectors)
        loc = torch.stack(loc, 2)
        loc = loc[:, :, [1, 0]].float().unsqueeze(2)
        # repeating for calculate the distance of contorl cpoints
        loc_tile = loc.repeat(1, 1,
                              local_cp_loc.size()[0] + global_cp_loc.size()[0],
                              1)
        self.register_buffer('loc_tile', loc_tile)
        self.int_steps = int_steps

        self.first_test = True

        if self.int_steps:
            self.flow_transformer = SpatialTransformer(self.img_size)

    def cp_gen(self):
        local_cp_loc = (self.local_cp_loc - 4) * self.scale + 4
        cpoint_pos = torch.cat([self.global_cp_loc, local_cp_loc], 0)
        return cpoint_pos

    def getWeight(self, c):
        '''
        输出每个点对应的控制点参数 \phi(||o - p_i|| / r)
        :param c: 径向基半径
        :return:
            weight:torch.Size([1,128,128,164,1])
        '''
        local_cp_loc = (self.local_cp_loc - 4) * self.scale + 4
        # cpoint_pos全局的坐标点
        cpoint_pos = torch.cat([self.global_cp_loc, local_cp_loc], 0)

        # a location mesh of control points,控制点的位置网
        cp_loc = cpoint_pos.unsqueeze(0).unsqueeze(0)
        # cp_loc_tile: torch.Szie{[128, 128, 164, 2]}
        cp_loc_tile = cp_loc.repeat(*self.img_size, 1, 1)

        # calculate r
        dist = torch.norm(self.loc_tile - cp_loc_tile, dim=3) / c
        # add mask for r < 1
        mask = dist < 1
        # weight if r<1 weight=(1-r)^4*(4r+1)
        #        else   weight=0
        # Todo: reduce weight size
        weight = torch.pow(1 - dist, 4) * (4 * dist + 1)
        weight = weight * mask.float()
        # weight:torch.Size([1,128,128,164,1])
        weight = weight.unsqueeze(0).unsqueeze(4)
        # print('calculate weight')

        return weight

    def interpolate(self, alpha, c):
        '''
        输入alpha 和 c, 产生weight为：\phi(||o - p_i|| / r)， 维度为：torch.Size([1,128,128,164,1])；
        :param alpha:对应C上每个控制点系数 mu 和 var, mu_list和var_list: {list:3}每个list:torch.Size([32, 164, 2])
        :param c: 径向基函数的半径
        :return:
            phi:
        '''
        weight = self.getWeight(c) #torch.Size([1,128,128,164,1])

        alpha = alpha.unsqueeze(1).unsqueeze(1) #torch.Szie([32, 1, 1, 164, 2])
        phi = torch.sum(weight * alpha, 3) #torch.Size([32, 128, 128, 2])
        phi = phi.permute(0, 3, 1, 2) #torch.Size([32, 2, 128, 128])
        return phi

    def interpolateForTest(self, all_alpha):
        if self.first_test:
            self.first_test = False
            weight_list = [self.getWeight(c).unsqueeze(1) for c in self.c_list]
            self.weight = torch.cat(weight_list, 1)
            # print('weight done')

        all_alpha = all_alpha.unsqueeze(2).unsqueeze(2)
        phi = torch.sum(self.weight * all_alpha, 4)
        return torch.sum(phi, 1).permute(0, 3, 1, 2)

    def diffeomorphic(self, flow):
        v = flow / (2**self.int_steps)
        for _ in range(self.int_steps):
            v1 = self.flow_transformer(v, v)
            v = v + v1
        return v

    def forward(self, IWAE_k, mu_list, var_list):
        '''
        输入：均值和方差，3个C的控制半径(每个C有32张图，164个控制点，2个位移方向)；
        输出：10个随机采样点的采样概率(alpha_list)和插值(phi_list)；
            采样概率（alpha_list）包括：3个紧支撑半径，每个半径中包括：32张图，164个控制点，2个位移方向；
            插值图（phi_list）包括：32张图，每个图有2个位移方向（x和y）,图像大小为128&128.
        :param src: 输入图像
        :param mu_list: mu_list={list:3}每个list:torch.Size([32, 164, 2])
        :param var_list: var_list={list:3}每个list:torch.Size([32, 164, 2])
        :return:
            alpha_list：10个采样值(list：10)，每个采样值包括3个紧支撑半径(list:3)，每个半径中的维度：torch.Size{[32,164,2]}
            phi_list: 10个采样值(list：10)，每个维度：torch.Size{[32,2,128，128]}
            self.scale:[1.5, 2, 2.5]
        '''
        phi_list = []
        alpha_list = []
        for i in range(IWAE_k):
            phi_list_one = []
            alpha_list_one = []
            for c, mu, var in zip(self.c_list, mu_list, var_list):
                alpha = Sample(mu, var)
                phi_list_one.append(self.interpolate(alpha, c)) # alpha对应的空间形变场
                alpha_list_one.append(alpha)

            phi = reduce(torch.add, phi_list_one)

            if self.int_steps:
                phi = self.diffeomorphic(phi)

            phi_list.append(phi)
            alpha_list.append(alpha_list_one)

        return alpha_list, phi_list, self.scale

    def test(self, src, mu_list, var_list):
        phi_list = []
        for c, mu, var in zip(self.c_list, mu_list, var_list):
            alpha = mu
            phi_list.append(self.interpolate(alpha, c))

        phi = reduce(torch.add, phi_list)

        # all_alpha = torch.cat([mu.unsqueeze(1) for mu in mu_list], 1)
        # phi = self.interpolateForTest(all_alpha)

        if self.int_steps:
            phi = self.diffeomorphic(phi)

        return phi, self.scale


class RBFGIMutilRadiusAAdaptive(GenerativeRegistrationNetwork):
    def __init__(self,
                 encoder_param,
                 c_list=[1.5, 2, 2.5],
                 i_size=[128, 128],
                 factor_list=[60000],
                 similarity_loss='LCC',
                 similarity_loss_param={},
                 int_steps=None):
        super(RBFGIMutilRadiusAAdaptive, self).__init__(i_size)
        self.encoder = RBFGIMutilRadiusEncoderA(**encoder_param)
        self.decoder = RBFGIMutilAdaptiveDecoder(i_size, c_list, int_steps)
        self.bending_energy_cal = RBFBendingEnergyLossA()
        self.similarity_loss = LOSSDICT[similarity_loss](
            **similarity_loss_param)
        self.jacobian_loss = JacobianDeterminantLoss()
        self.factor_list = factor_list
        self.c_list = c_list

        # generate a name
        name = str(similarity_loss) + '--'
        for k in similarity_loss_param:
            name += '-' + str(similarity_loss_param[k])
        name += '--'
        for i in self.factor_list:
            name += str(i)
        self.name = name
        if int_steps:
            self.name += '-diff'

    def forward(self, src, tgt, IWAE_k):
        '''
        输入：src和tgt,
        处理：alpha_list为encoder的出mu和var;{tuple:2}, 每个tuple包括{list：3}, 每个list：torch.Size([32, 164, 2])
            IWAE_alpha：控制点分布；k个采样值(list:k)，每个采样值包括3个紧支撑半径(list:3)，每个半径中的维度：torch.Size{[32,164,2]}[batch_size, contr_point, dirction]
            IWAE_phi: 形变场；k个采样值(list:k)，每个维度：torch.Size{[32,2,128，128]}[batch_size, dirction, img_H, img_W]
            IWAE_w_src: decoder 产生的图片, 10个采样值，{list:10}, 每个list: torch.Size([32, 1, 128, 128])[batch_size, dirction, img_H, img_W]
        :param src: torch.Size([32, 1, 128, 128])[batch_size, 1, src_img_H, src_img_W]
        :param tgt: torch.Size([32, 1, 128, 128])[batch_size, 1, tgt_img_H, tgt_img_W]
        :return:
        '''
        alpha_list = self.encoder(src, tgt)
        IWAE_alpha, IWAE_phi, _ = self.decoder(IWAE_k, *alpha_list)
        IWAE_w_src = []
        for i in range(IWAE_k):
            w_src = self.transformer(src, IWAE_phi[i])
            IWAE_w_src.append(w_src)
        return IWAE_alpha, IWAE_phi, IWAE_w_src, alpha_list

    def test(self, src, tgt):
        alpha_list = self.encoder(src, tgt)
        phi, _ = self.decoder.test(src, *alpha_list)

        w_src = self.transformer(src, phi)
        return phi, w_src, alpha_list

    def objective(self, src, tgt, IWAE_k):
        IWAE_Z_Sample, phi, IWAE_w_src, alpha_list = self(src, tgt, IWAE_k)# src_s 需要用IWAE_w_src的一个维度代替；flow需要用IWAE_phi的某一个维度代替

        IWAE_log_p_FCMz = []
        for i in range(IWAE_k):
            similarity_loss = self.similarity_loss(IWAE_w_src[i], tgt)
            IWAE_log_p_FCMz.append(similarity_loss.unsqueeze(0))

        # 3个C的2个维度的6维张量，2各维度分别是x方向和y方向
        bending_energy_list = [
            self.bending_energy_cal(mu, self.decoder.cp_gen(), c)
            for mu, c in zip(alpha_list[0], self.c_list)
        ]

        # 计算损失函数
        simi_loss = self.IWAE_Similarity_Loss(IWAE_log_p_FCMz, IWAE_Z_Sample,  IWAE_k, alpha_list)

        smooth_term = reduce(torch.add, bending_energy_list)

        # 计算KL散度
        sigma = torch.cat(alpha_list[1], 1)
        sigma_term = torch.sum(torch.exp(sigma) - sigma, dim=[1, 2])
        KL_loss = sigma_term / 2 + smooth_term

        L1_loss = torch.sum(torch.abs(torch.cat(alpha_list[0], 1)), dim=[1, 2])

        # 由于只取一次encoder,所以其中一个就能算出jacobian_loss。
        jacobian_loss = self.jacobian_loss(phi[0])

        return {
            'simil_loss':
            simi_loss,
            'KL_loss':
            KL_loss,
            'jacob_loss':
            jacobian_loss,
            'L1_loss':
            L1_loss,
            'loss':
            self.factor_list[0] * simi_loss + KL_loss +
            self.factor_list[1] * jacobian_loss +
            self.factor_list[2] * L1_loss,
            'ELBO':
            self.factor_list[0] * simi_loss + KL_loss
        }

    def IWAE_Similarity_Loss(self, IWAE_log_p_FCMz, IWAE_Z_Sample, IWAE_k, IWAE_q_zCFM_mu_sigma):

        IWAE_log_p_Z_sub_Q_zCFM = self.IWAE_calculate_P_z_sub_Q_zCFM(IWAE_Z_Sample, IWAE_q_zCFM_mu_sigma, IWAE_k)
        IWAE_log_p_Z_sub_Q_zCFM = torch.cat(IWAE_log_p_Z_sub_Q_zCFM, dim=0).permute(1, 0) # torch.Size([32, IWAE])
        IWAE_log_p_FCMz = torch.cat(IWAE_log_p_FCMz, dim=0).permute(1, 0) # torch.Size([32, IWAE])
        w_eps_log = -1.0 * self.factor_list[0] * IWAE_log_p_FCMz + IWAE_log_p_Z_sub_Q_zCFM # torch.Size([32, IWAE])
        tgt_w_eps_log = w_eps_log.unsqueeze(2).repeat(1, 1, IWAE_k)
        src_w_eps_log = w_eps_log.unsqueeze(1).repeat(1, IWAE_k, 1)
        w_eps_log_count = tgt_w_eps_log - src_w_eps_log
        w_eps_log_filter_count = w_eps_log_count < 0.1

        IWAE_log_p_FCMz_filter_count = torch.all(w_eps_log_filter_count, dim=2) != 0
        IWAE_Loss_count = IWAE_log_p_FCMz_filter_count.float() * IWAE_log_p_FCMz
        IWAE_Loss = torch.sum(IWAE_Loss_count, dim=1)
        return IWAE_Loss

    def IWAE_calculate_P_z_sub_Q_zCFM(self, IWAE_Z_Sample, IWAE_q_zCFM_mu_sigma, IWAE_k):
        ## TODO: 1. calculate  IWAE_q_zCFM_mu, IWAE_q_zCFM_sigma, IWAE_q_zCFM_sigma_one_flatten_matrix
        IWAE_q_zCFM_mu = IWAE_q_zCFM_mu_sigma[0]
        IWAE_q_zCFM_sigma = IWAE_q_zCFM_mu_sigma[1]
        # IWAE_q_zCFM_mu_one, IWAE_q_zCFM_sigma_one: torch.Size([1, 32, 164])
        IWAE_q_zCFM_mu_one = torch.cat(IWAE_q_zCFM_mu, dim=2).permute(0, 2, 1)
        IWAE_q_zCFM_sigma_one = torch.exp(0.5 * torch.cat(IWAE_q_zCFM_sigma, dim=2).permute(0, 2, 1))
        # 拉平
        IWAE_q_zCFM_mu_one_flatten = torch.cat(
            (IWAE_q_zCFM_mu_one[0][0], IWAE_q_zCFM_mu_one[0][2], IWAE_q_zCFM_mu_one[0][4],
             IWAE_q_zCFM_mu_one[0][1], IWAE_q_zCFM_mu_one[0][3], IWAE_q_zCFM_mu_one[0][5])).unsqueeze(0)
        IWAE_q_zCFM_sigma_one_flatten = torch.exp(0.5*torch.cat(
            (IWAE_q_zCFM_sigma_one[0][0], IWAE_q_zCFM_sigma_one[0][2], IWAE_q_zCFM_sigma_one[0][4],
             IWAE_q_zCFM_sigma_one[0][1], IWAE_q_zCFM_sigma_one[0][3], IWAE_q_zCFM_sigma_one[0][5])))

        IWAE_p_Z_sub_Q_zCFM_sum = []
        # TODO: 处理采样点z
        for k in range(IWAE_k):
            # torch.Size([6, 164, 32]),,6对应着x_{1.5}, y_{1.5}, x_{2}, y_{2}, x_{2.5}, y_{2.5}
            IWAE_Z_Sample_k_one = torch.cat((IWAE_Z_Sample[k][0], IWAE_Z_Sample[k][1], IWAE_Z_Sample[k][2]), 2).permute(2, 0, 1)
            #  torch.Size([6, 164, 32]),,6对应着x_{1.5}, x_{2}, x_{2.5}, y_{1.5}, y_{2}, y_{2.5}
            IWAE_Z_Sample_k_xxx_yyy = torch.cat((IWAE_Z_Sample_k_one[0], IWAE_Z_Sample_k_one[2],
                                                 IWAE_Z_Sample_k_one[4], IWAE_Z_Sample_k_one[1],
                                                 IWAE_Z_Sample_k_one[3], IWAE_Z_Sample_k_one[5]), dim=1).unsqueeze(1)
            ## 4. 对于p_z处理z数据
            # # TODO: 用现有分布公式计算log_q_zCFM
            dist_log_q_zCFM = torch.distributions.Normal(IWAE_q_zCFM_mu_one_flatten, IWAE_q_zCFM_sigma_one_flatten.unsqueeze(0))
            log_q_zCFM = dist_log_q_zCFM.log_prob(IWAE_Z_Sample_k_xxx_yyy)
            IWAE_log_q_zCFM_sum = torch.squeeze(torch.sum(log_q_zCFM, dim=2))
            ## TODO:尝试用现有公式计算log_p_z
            c_list = [1.5, 2, 2.5]
            IWAE_log_p_z = []
            for i in range(3):
                weight = self.get_P_z_SigamP(c_list[i]).to(IWAE_log_q_zCFM_sum.device)
                ## TODO：分开轴来计算
                IWAE_P_z_mu = torch.zeros(len(weight)).unsqueeze(0).to(IWAE_log_q_zCFM_sum.device)
                dist_P_z_one =torch.distributions.MultivariateNormal(IWAE_P_z_mu, weight)
                #x
                IWAE_P_z_x = dist_P_z_one.log_prob(IWAE_Z_Sample_k_one[2*i].unsqueeze(1))
                IWAE_log_p_z.append(IWAE_P_z_x)
                # y
                IWAE_P_z_y = dist_P_z_one.log_prob(IWAE_Z_Sample_k_one[2*i+1].unsqueeze(1))
                IWAE_log_p_z.append(IWAE_P_z_y)
            IWAE_log_p_z_sum = torch.cat(IWAE_log_p_z, dim=1)
            IWAE_log_p_z_sum_batch = torch.sum(IWAE_log_p_z_sum, dim=1)

            IWAE_p_Z_sub_Q_zCFM = (IWAE_log_p_z_sum_batch - IWAE_log_q_zCFM_sum).unsqueeze(0)
            IWAE_p_Z_sub_Q_zCFM_sum.append(IWAE_p_Z_sub_Q_zCFM)
        return IWAE_p_Z_sub_Q_zCFM_sum


    def get_P_z_SigamP(self, r):
        # TODO: global_cp_loc_grid
        global_cp_loc_grid = [
            torch.linspace(s + (e - s) / 16, e - (e - s) / 16, 8)
            for s, e in ((0, 8), (0, 8))
        ]
        global_cp_loc_grid = torch.meshgrid(global_cp_loc_grid)
        global_cp_loc = torch.stack(global_cp_loc_grid, 2)[:, :, [1, 0]]
        global_cp_loc = torch.flatten(global_cp_loc, start_dim=0,
                                      end_dim=1).float()
        # TODO: local_cp_loc_grid
        local_cp_loc_grid = [
            torch.linspace(s + (s - e) / 20, e - (s - e) / 20, 10)
            for s, e in ((2, 6), (2, 6))
        ]
        local_cp_loc_grid = torch.meshgrid(local_cp_loc_grid)
        local_cp_loc = torch.stack(local_cp_loc_grid, 2)[:, :, [1, 0]]
        local_cp_loc = torch.flatten(local_cp_loc, start_dim=0,
                                     end_dim=1).float()

        # TODO: local_cp_loc扩展并拼接
        local_cp_loc = (local_cp_loc - 4) * torch.tensor([1, 1], dtype=torch.float) + 4
        cpoint_pos = torch.cat([global_cp_loc, local_cp_loc], 0)

        # TODO: 计算预置矩阵
        num_cp = cpoint_pos.size()[0]
        scppos = cpoint_pos.unsqueeze(1).repeat(1, num_cp, 1)
        despos = cpoint_pos.unsqueeze(0).repeat(num_cp, 1, 1)
        dis = torch.norm(scppos - despos, dim=2) / r
        filter_dis = dis < 1
        weight = torch.pow(1 - dis, 4) * (4 * dis + 1)
        weight_1 = (weight * filter_dis.float())
        return weight_1


    def uncertainty(self, src, tgt, K):
        _, _, alpha_list = self(src, tgt)
        d = []
        for _ in range(K):
            phi_xy = self.decoder(src,
                                  *alpha_list)[0].unsqueeze(1)  # B 1 2 H W
            phi_r = torch.norm(phi_xy, dim=2, keepdim=True)  # B 1 1 H W
            phi_theta = torch.abs(
                torch.atan2(phi_xy[:, :, 1, :, :],
                            phi_xy[:, :,
                                   0, :, :])).unsqueeze(2) * 180  # B 1 1 H W
            phi_polar = torch.cat([phi_r, phi_theta], dim=2)  # B 1 2 H W
            phi = torch.cat([phi_xy, phi_polar], dim=2)  # B 1 4 H W
            d.append(phi)

        d = torch.cat(d, dim=1)  # B K 4 H W
        d_expect = torch.mean(d, dim=1)  # B 4 H W
        dd_expect = torch.mean(d * d, dim=1)  # B 4 H W
        var1 = 1 + dd_expect - d_expect * d_expect
        # var = torch.mean(torch.pow(d - d_expect, 2), dim=1)
        # return var
        var2 = torch.mean(torch.pow(d - torch.mean(d, dim=1, keepdim=True), 2),
                          dim=1)
        return torch.cat([var1, var2], dim=1), d_expect


