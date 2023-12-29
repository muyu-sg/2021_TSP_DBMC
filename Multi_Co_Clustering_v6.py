'''
@Author: Shide Du
@Date: 2020-02-03 09:10:04
@Email: shidedums@163.com
@Description:
'''
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as FF
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from torch import nn
from tqdm import tqdm
import time
from DataStardardization import zero_score
from LoadF import loadFW, loadGW, loadMFW, loadMGW
from clusteringPerformance1 import similarity_function, StatisticClustering
from loadMatData import loadData
from ranger import Ranger



class Bi_sparse_co_clustering(nn.Module):
    def __init__(self, block_num, lr, epoch, X0, X1, X2, X3, X4, X5, F0, F1, F2, F3, F4, F5, G, FW, GW0, GW1, GW2, GW3, GW4, GW5, gnd, para, alpha):
        super(Bi_sparse_co_clustering, self).__init__()
        self.lr = lr
        self.block_num = block_num
        self.epoch = epoch
        self.alpha = alpha
    ##############################################################################################################
        ##  Init data
        #   Training data X_v
        self.X0_train = torch.from_numpy(X0).double()#
        self.X1_train = torch.from_numpy(X1).double()  #
        self.X2_train = torch.from_numpy(X2).double()  #
        self.X3_train = torch.from_numpy(X3).double()  #
        self.X4_train = torch.from_numpy(X4).double()  #
        self.X5_train = torch.from_numpy(X5).double()  #
        #   Init F_v
        self.F0_init = torch.from_numpy(F0).double()  #
        self.F1_init = torch.from_numpy(F1).double()  #
        self.F2_init = torch.from_numpy(F2).double()  #
        self.F3_init = torch.from_numpy(F3).double()  #
        self.F4_init = torch.from_numpy(F4).double()  #
        self.F5_init = torch.from_numpy(F5).double()  #
        #   L_wv
        self.G_init = torch.from_numpy(G).double()  #
        self.FW_init = torch.from_numpy(FW).double()
        self.GW0_init = torch.from_numpy(GW0).double()
        self.GW1_init = torch.from_numpy(GW1).double()
        self.GW2_init = torch.from_numpy(GW2).double()
        self.GW3_init = torch.from_numpy(GW3).double()
        self.GW4_init = torch.from_numpy(GW4).double()
        self.GW5_init = torch.from_numpy(GW5).double()

        #self.W = torch.from_numpy(W).float()

        self.gnd = gnd
        self.X0_train = zero_score(self.X0_train).double()
        self.X1_train = zero_score(self.X1_train).double()
        self.X2_train = zero_score(self.X2_train).double()
        self.X3_train = zero_score(self.X3_train).double()
        self.X4_train = zero_score(self.X4_train).double()
        self.X5_train = zero_score(self.X5_train).double()
    ##############################################################################################################
        ##  Init Xdim
        self.input0_dim = self.X0_train.shape[0]
        self.input1_dim = self.X1_train.shape[0]
        self.input2_dim = self.X2_train.shape[0]
        self.input3_dim = self.X3_train.shape[0]
        self.input4_dim = self.X4_train.shape[0]
        self.input5_dim = self.X5_train.shape[0]

        self.output0_dim = self.X0_train.shape[1]
        self.output1_dim = self.X1_train.shape[1]
        self.output2_dim = self.X2_train.shape[1]
        self.output3_dim = self.X3_train.shape[1]
        self.output4_dim = self.X4_train.shape[1]
        self.output5_dim = self.X5_train.shape[1]
        #  init Fdim
        self.output_dim01 = self.F0_init.shape[1]
        self.output_dim11 = self.F1_init.shape[1]
        self.output_dim21 = self.F2_init.shape[1]
        self.output_dim31 = self.F3_init.shape[1]
        self.output_dim41 = self.F4_init.shape[1]
        self.output_dim51 = self.F5_init.shape[1]
        #  init Gdim
        self.output_dim2 = self.G_init.shape[1]
    ##############################################################################################################

        ##  learnable layer and soft_threshold valse theta
        #   U2
        self.S01 = nn.Linear(self.output_dim01, self.output_dim01, bias=False).double()#U2
        self.S010 = nn.Linear(self.output_dim01, self.output_dim01, bias=False).double()
        self.S11 = nn.Linear(self.output_dim11, self.output_dim11, bias=False).double()
        self.S111 = nn.Linear(self.output_dim11, self.output_dim11, bias=False).double()
        self.S21 = nn.Linear(self.output_dim21, self.output_dim21, bias=False).double()
        self.S212 = nn.Linear(self.output_dim21, self.output_dim21, bias=False).double()
        self.S31 = nn.Linear(self.output_dim31, self.output_dim31, bias=False).double()
        self.S313 = nn.Linear(self.output_dim31, self.output_dim31, bias=False).double()
        self.S41 = nn.Linear(self.output_dim41, self.output_dim41, bias=False).double()
        self.S414 = nn.Linear(self.output_dim41, self.output_dim41, bias=False).double()
        self.S51 = nn.Linear(self.output_dim51, self.output_dim51, bias=False).double()
        self.S515 = nn.Linear(self.output_dim51, self.output_dim51, bias=False).double()

        #  U1
        self.S02 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()#U1
        self.S020 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()  # U1
        self.S12 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S121 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S22 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S222 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S32 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S323 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S42 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S424 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S52 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()
        self.S525 = nn.Linear(self.output_dim2, self.output_dim2, bias=False).double()

        self.theta1 = nn.Parameter(torch.DoubleTensor([para]), requires_grad=True)
        self.theta2 = nn.Parameter(torch.DoubleTensor([para]), requires_grad=True)
        self.theta3 = nn.Parameter(torch.DoubleTensor([para]), requires_grad=True)
        self.theta4 = nn.Parameter(torch.DoubleTensor([para]), requires_grad=True)
        self.theta5 = nn.Parameter(torch.DoubleTensor([para]), requires_grad=True)
        self.theta6 = nn.Parameter(torch.DoubleTensor([para]), requires_grad=True)
        self.theta7 = nn.Parameter(torch.DoubleTensor([para]), requires_grad=True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    ##############################################################################################################
    #  soft_threshold function
    def soft_threshold1(self, u):
        tmp_u = torch.abs(u)
        h = torch.abs(tmp_u - self.theta1).mul(torch.sign(u))
        return h

    def soft_threshold2(self, u):
        tmp_u = torch.abs(u)
        h = torch.abs(tmp_u - self.theta2).mul(torch.sign(u))
        return h

    def soft_threshold3(self, u):
        tmp_u = torch.abs(u)
        h = torch.abs(tmp_u - self.theta3).mul(torch.sign(u))
        return h

    def soft_threshold4(self, u):
        tmp_u = torch.abs(u)
        h = torch.abs(tmp_u - self.theta4).mul(torch.sign(u))
        return h

    def soft_threshold5(self, u):
        tmp_u = torch.abs(u)
        h = torch.abs(tmp_u - self.theta5).mul(torch.sign(u))
        return h

    def soft_threshold6(self, u):
        tmp_u = torch.abs(u)
        h = torch.abs(tmp_u - self.theta6).mul(torch.sign(u))
        return h

    def soft_threshold7(self, u):
        tmp_u = torch.abs(u)
        h = torch.abs(tmp_u - self.theta7).mul(torch.sign(u))
        return h


    def soft_threshold8(self, u):
        return FF.relu(u - self.theta1) - FF.relu(-1.0 * u - self.theta1)

    def soft_threshold81(self, u):
        return FF.relu(u - self.theta1) - FF.relu(-1.0 * u - self.theta1)

    def soft_threshold82(self, u):
        return FF.relu(u - self.theta2) - FF.relu(-1.0 * u - self.theta2)

    def soft_threshold83(self, u):
        return FF.relu(u - self.theta3) - FF.relu(-1.0 * u - self.theta3)

    def soft_threshold84(self, u):
        return FF.relu(u - self.theta4) - FF.relu(-1.0 * u - self.theta4)

    def soft_threshold85(self, u):
        return FF.relu(u - self.theta5) - FF.relu(-1.0 * u - self.theta5)

    def soft_threshold86(self, u):
        return FF.relu(u - self.theta6) - FF.relu(-1.0 * u - self.theta6)

    def soft_threshold87(self, u):
        return FF.relu(u - self.theta7) - FF.relu(-1.0 * u - self.theta7)



    ##############################################################################################################
    ##  Init Batchlayer
    def para_init(self):
        self.bn_input0_0 = nn.BatchNorm1d(self.output0_dim, momentum=0.5).double()
        self.bn_input1_0 = nn.BatchNorm1d(self.output1_dim, momentum=0.5).double()
        self.bn_input2_0 = nn.BatchNorm1d(self.output2_dim, momentum=0.5).double()
        self.bn_input3_0 = nn.BatchNorm1d(self.output3_dim, momentum=0.5).double()
        self.bn_input4_0 = nn.BatchNorm1d(self.output4_dim, momentum=0.5).double()
        self.bn_input5_0 = nn.BatchNorm1d(self.output5_dim, momentum=0.5).double()

        self.bn_input_01 = nn.BatchNorm1d(self.output_dim01, momentum=0.5).double()
        self.bn_input_11 = nn.BatchNorm1d(self.output_dim11, momentum=0.5).double()
        self.bn_input_21 = nn.BatchNorm1d(self.output_dim21, momentum=0.5).double()
        self.bn_input_31 = nn.BatchNorm1d(self.output_dim31, momentum=0.5).double()
        self.bn_input_41 = nn.BatchNorm1d(self.output_dim41, momentum=0.5).double()
        self.bn_input_51 = nn.BatchNorm1d(self.output_dim51, momentum=0.5).double()

        self.bn_input_2 = nn.BatchNorm1d(self.output_dim2, momentum=0.5).double()

        self.dropout1 = nn.Dropout(0.5)

    ##############################################################################################################
    ##  Loss function
    def my_loss(self, x, pred_x):
        return torch.norm(x - pred_x, 2) ** 2

    def my_loss1(self, x, pred_x):
        criterion = nn.MSELoss()
        return criterion(x, pred_x)

    def my_loss2(self, x, pred_x):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(x, pred_x)


    ##############################################################################################################
    #  Struction of net
    def forward(self, x0, x1, x2, x3, x4, x5):
        G = []
        S = []
        F0 = []
        F1 = []
        F2 = []
        F3 = []
        F4 = []
        F5 = []

        G_std = []
        F0_std = []
        F1_std = []
        F2_std = []
        F3_std = []
        F4_std = []
        F5_std = []
    ##############################################################################################################
        x0 = self.bn_input0_0(x0)
        x1 = self.bn_input1_0(x1)
        x2 = self.bn_input2_0(x2)
        x3 = self.bn_input3_0(x3)
        x4 = self.bn_input4_0(x4)
        x5 = self.bn_input5_0(x5)
    ##############################################################################################################
        #   Graph Laplacian Matrix FW
        #FD = torch.diag(self.FW_init.sum(1))
        #FL = FD - self.FW_init
        # Fm = self.FW_init.size(0)
        # FI = torch.eye(Fm)
        # FK = torch.inverse(FI + 1 * FL)
        # FH = FI - torch.ones(Fm, Fm) / Fm
        # FKc = FH.matmul(FK).matmul(FH)

        #   Graph Laplacian Matrix GW
        GD0 = torch.diag(self.GW0_init.sum(1)).double()
        #GI0 = torch.eye(GD0.size(0), GD0.size(1)).double()
        GL0 = GD0 - self.GW0_init#((GD0)**(-0.5)).matmul(self.GW0_init).matmul((GD0)**(-0.5))

        GD1 = torch.diag(self.GW1_init.sum(1)).double()
        #GI1 = torch.eye(GD1.size(0), GD1.size(1)).double()
        GL1 = GD1 - self.GW1_init

        GD2 = torch.diag(self.GW2_init.sum(1)).double()
        #GI2 = torch.eye(GD2.size(0), GD2.size(1)).double()
        GL2 = GD2 - self.GW2_init

        GD3 = torch.diag(self.GW3_init.sum(1)).double()
        # GI3 = torch.eye(GD3.size(0), GD3.size(1)).double()
        GL3 = GD3 - self.GW3_init

        GD4 = torch.diag(self.GW4_init.sum(1)).double()
        # GI4 = torch.eye(GD4.size(0), GD4.size(1)).double()
        GL4 = GD4 - self.GW4_init

        GD5 = torch.diag(self.GW5_init.sum(1)).double()
        #GI5 = torch.eye(GD5.size(0), GD5.size(1)).double()
        GL5 = GD5 - self.GW5_init


        # Gm = self.GW_init.size(0)
        # GI = torch.eye(Gm)
        # GK = torch.inverse(GI + 1 * GL)
        # GH = GI - torch.ones(Gm, Gm)/Gm
        # GKc = GH.matmul(GK).matmul(GH)
    ##############################################################################################################
        ## Init S
        tmp30 = torch.inverse(self.G_init.t().matmul(self.G_init))
        tmp40 = torch.inverse(self.F0_init.t().matmul(self.F0_init))
        tmp50 = self.G_init.t().matmul(x0.matmul(self.F0_init))
        self.S0_init = tmp30.matmul(tmp50.matmul(tmp40))

        tmp31 = torch.inverse(self.G_init.t().matmul(self.G_init))
        tmp41 = torch.inverse(self.F1_init.t().matmul(self.F1_init))
        tmp51 = self.G_init.t().matmul(x1.matmul(self.F1_init))
        self.S1_init = tmp31.matmul(tmp51.matmul(tmp41))

        tmp32 = torch.inverse(self.G_init.t().matmul(self.G_init))
        tmp42 = torch.inverse(self.F2_init.t().matmul(self.F2_init))
        tmp52 = self.G_init.t().matmul(x2.matmul(self.F2_init))
        self.S2_init = tmp32.matmul(tmp52.matmul(tmp42))

        tmp33 = torch.inverse(self.G_init.t().matmul(self.G_init))
        tmp43 = torch.inverse(self.F3_init.t().matmul(self.F3_init))
        tmp53 = self.G_init.t().matmul(x3.matmul(self.F3_init))
        self.S3_init = tmp33.matmul(tmp53.matmul(tmp43))

        tmp34 = torch.inverse(self.G_init.t().matmul(self.G_init))
        tmp44 = torch.inverse(self.F4_init.t().matmul(self.F4_init))
        tmp54 = self.G_init.t().matmul(x4.matmul(self.F4_init))
        self.S4_init = tmp34.matmul(tmp54.matmul(tmp44))

        tmp35 = torch.inverse(self.G_init.t().matmul(self.G_init))
        tmp45 = torch.inverse(self.F5_init.t().matmul(self.F5_init))
        tmp55 = self.G_init.t().matmul(x5.matmul(self.F5_init))
        self.S5_init = tmp35.matmul(tmp55.matmul(tmp45))

        self.S_init = (self.S0_init + self.S1_init + self.S2_init + self.S3_init + self.S4_init + self.S5_init) / 6
    ##############################################################################################################


        ## Init F_v
        L1 = torch.norm((self.G_init).t().matmul(self.G_init))
        F0.append(self.soft_threshold1(x0.t().matmul(self.G_init.matmul(self.S_init)) / L1))  #1440*1024 * 1024*20 * 20*20=1440*20
        F0_std.append(self.bn_input_01(F0[-1]))

        F1.append(self.soft_threshold2(x1.t().matmul(self.G_init.matmul(self.S_init)) / L1))  # 1440*1024 * 1024*20 * 20*20=1440*20
        F1_std.append(self.bn_input_11(F1[-1]))

        F2.append(self.soft_threshold3(x2.t().matmul(self.G_init.matmul(self.S_init)) / L1))  # 1440*1024 * 1024*20 * 20*20=1440*20
        F2_std.append(self.bn_input_21(F2[-1]))

        F3.append(self.soft_threshold4(x3.t().matmul(self.G_init.matmul(self.S_init)) / L1))  # 1440*1024 * 1024*20 * 20*20=1440*20
        F3_std.append(self.bn_input_31(F3[-1]))

        F4.append(self.soft_threshold5(x4.t().matmul(self.G_init.matmul(self.S_init)) / L1))  # 1440*1024 * 1024*20 * 20*20=1440*20
        F4_std.append(self.bn_input_41(F4[-1]))

        F5.append(self.soft_threshold6(x5.t().matmul(self.G_init.matmul(self.S_init)) / L1))  # 1440*1024 * 1024*20 * 20*20=1440*20
        F5_std.append(self.bn_input_51(F5[-1]))
    ##############################################################################################################
        ## Init G
        L20 = torch.norm(F0_std[-1].t().matmul(F0_std[-1]))
        L21 = torch.norm(F1_std[-1].t().matmul(F1_std[-1]))
        L22 = torch.norm(F2_std[-1].t().matmul(F2_std[-1]))
        L23 = torch.norm(F3_std[-1].t().matmul(F3_std[-1]))
        L24 = torch.norm(F4_std[-1].t().matmul(F4_std[-1]))
        L25 = torch.norm(F5_std[-1].t().matmul(F5_std[-1]))
        temporary1 = (x0.matmul(F0_std[-1].matmul(self.S_init.t()))/L20)
        temporary2 = (x1.matmul(F1_std[-1].matmul(self.S_init.t()))/L21)
        temporary3 = (x2.matmul(F2_std[-1].matmul(self.S_init.t()))/L22)
        temporary4 = (x3.matmul(F3_std[-1].matmul(self.S_init.t()))/L23)
        temporary5 = (x4.matmul(F4_std[-1].matmul(self.S_init.t()))/L24)
        temporary6 = (x5.matmul(F5_std[-1].matmul(self.S_init.t()))/L25)

        temporary1 = self.soft_threshold87(temporary1)
        #temporary1 = (1 - (1 - alpha) / torch.norm(temporary1, 2)) * temporary1
        temporary2 = self.soft_threshold87(temporary2)
        #temporary2 = (1 - (1 - alpha) / torch.norm(temporary2, 2)) * temporary2
        temporary3 = self.soft_threshold87(temporary3)
        #temporary3 = (1 - (1 - alpha) / torch.norm(temporary3, 2)) * temporary3
        temporary4 = self.soft_threshold87(temporary4)
        #temporary4 = (1 - (1 - alpha) / torch.norm(temporary4, 2)) * temporary4
        temporary5 = self.soft_threshold87(temporary5)
        #temporary5 = (1 - (1 - alpha) / torch.norm(temporary5, 2)) * temporary5
        temporary6 = self.soft_threshold87(temporary6)
        #temporary6 = (1 - (1 - alpha) / torch.norm(temporary6, 2)) * temporary6
        temp = (temporary1 + temporary2 + temporary3 + temporary4 + temporary5 + temporary6)/6
        G.append(temp)

        G_std.append(self.bn_input_2(G[-1]))
    ##############################################################################################################
        # Start
        tmp30 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
        tmp40 = torch.inverse(F0_std[-1].t().matmul(F0_std[-1]))
        tmp50 = G_std[-1].t().matmul(x0.matmul(F0_std[-1]))
        tmp60 = tmp30.matmul(tmp50.matmul(tmp40))

        tmp31 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
        tmp41 = torch.inverse(F1_std[-1].t().matmul(F1_std[-1]))
        tmp51 = G_std[-1].t().matmul(x1.matmul(F1_std[-1]))
        tmp61 = tmp31.matmul(tmp51.matmul(tmp41))

        tmp32 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
        tmp42 = torch.inverse(F2_std[-1].t().matmul(F2_std[-1]))
        tmp52 = G_std[-1].t().matmul(x2.matmul(F2_std[-1]))
        tmp62 = tmp32.matmul(tmp52.matmul(tmp42))

        tmp33 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
        tmp43 = torch.inverse(F3_std[-1].t().matmul(F3_std[-1]))
        tmp53 = G_std[-1].t().matmul(x3.matmul(F3_std[-1]))
        tmp63 = tmp33.matmul(tmp53.matmul(tmp43))

        tmp34 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
        tmp44 = torch.inverse(F4_std[-1].t().matmul(F4_std[-1]))
        tmp54 = G_std[-1].t().matmul(x4.matmul(F4_std[-1]))
        tmp64 = tmp34.matmul(tmp54.matmul(tmp44))

        tmp35 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
        tmp45 = torch.inverse(F5_std[-1].t().matmul(F5_std[-1]))
        tmp55 = G_std[-1].t().matmul(x5.matmul(F5_std[-1]))
        tmp65 = tmp35.matmul(tmp55.matmul(tmp45))

        tmp6 = (tmp60 + tmp61 + tmp62 + tmp63 + tmp64 + tmp65) / 6
        S.append(tmp6)
    ##############################################################################################################
        for t in range(0, self.block_num):
        ##############################################################################################################
            # update F
            L1 = torch.norm((G_std[-1]).t().matmul(G_std[-1]))
            tmp10 = self.S01(F0_std[-1]) + x0.t().matmul(G_std[-1].matmul(S[-1])) / L1 - ((F0_std[-1].matmul(F0_std[-1].t().matmul(F0_std[-1])) - F0_std[-1]) * (alpha) / L1)
            F0.append(self.soft_threshold81(tmp10))
            F0_std.append(self.bn_input_01(F0[-1]))

            tmp11 = self.S11(F1_std[-1]) + x1.t().matmul(G_std[-1].matmul(S[-1])) / L1 - ((F1_std[-1].matmul(F1_std[-1].t().matmul(F1_std[-1])) - F1_std[-1]) * (alpha) / L1)
            F1.append(self.soft_threshold82(tmp11))
            F1_std.append(self.bn_input_11(F1[-1]))

            tmp21 = self.S21(F2_std[-1]) + x2.t().matmul(G_std[-1].matmul(S[-1])) / L1 - ((F2_std[-1].matmul(F2_std[-1].t().matmul(F2_std[-1])) - F2_std[-1]) * (alpha) / L1)
            F2.append(self.soft_threshold83(tmp21))
            F2_std.append(self.bn_input_21(F2[-1]))

            tmp31 = self.S31(F3_std[-1]) + x3.t().matmul(G_std[-1].matmul(S[-1])) / L1 - ((F3_std[-1].matmul(F3_std[-1].t().matmul(F3_std[-1])) - F3_std[-1]) * (alpha) / L1)
            F3.append(self.soft_threshold84(tmp31))
            F3_std.append(self.bn_input_31(F3[-1]))

            tmp41 = self.S41(F4_std[-1]) + x4.t().matmul(G_std[-1].matmul(S[-1])) / L1 - ((F4_std[-1].matmul(F4_std[-1].t().matmul(F4_std[-1])) - F4_std[-1]) * (alpha) / L1)
            F4.append(self.soft_threshold85(tmp41))
            F4_std.append(self.bn_input_41(F4[-1]))

            tmp51 = self.S51(F5_std[-1]) + x5.t().matmul(G_std[-1].matmul(S[-1])) / L1 - ((F5_std[-1].matmul(F5_std[-1].t().matmul(F5_std[-1])) - F5_std[-1]) * (alpha) / L1)
            F5.append(self.soft_threshold86(tmp51))
            F5_std.append(self.bn_input_51(F5[-1]))
        ##############################################################################################################
            # update G
            L20 = torch.norm(F0_std[-1].t().matmul(F0_std[-1]))
            L21 = torch.norm(F1_std[-1].t().matmul(F1_std[-1]))
            L22 = torch.norm(F2_std[-1].t().matmul(F2_std[-1]))
            L23 = torch.norm(F3_std[-1].t().matmul(F3_std[-1]))
            L24 = torch.norm(F4_std[-1].t().matmul(F4_std[-1]))
            L25 = torch.norm(F5_std[-1].t().matmul(F5_std[-1]))
            temporary1 =   ((self.S02(G_std[-1]) + x0.matmul(F0_std[-1].matmul(self.S_init.t())) / L20  - ((GL0) * (self.alpha)/L20).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1]))-G_std[-1])*(alpha) / L20))
                          + (self.S12(G_std[-1]) + x1.matmul(F1_std[-1].matmul(self.S_init.t())) / L21  - ((GL1) * (self.alpha)/L21).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1]))-G_std[-1])*(alpha) / L21))
                          + (self.S22(G_std[-1]) + x2.matmul(F2_std[-1].matmul(self.S_init.t())) / L22  - ((GL2) * (self.alpha)/L22).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1]))-G_std[-1])*(alpha) / L22))
                          + (self.S32(G_std[-1]) + x3.matmul(F3_std[-1].matmul(self.S_init.t())) / L23  - ((GL3) * (self.alpha)/L23).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1]))-G_std[-1])*(alpha) / L23))
                          + (self.S42(G_std[-1]) + x4.matmul(F4_std[-1].matmul(self.S_init.t())) / L24  - ((GL4) * (self.alpha)/L24).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1]))-G_std[-1])*(alpha) / L24))
                          + (self.S52(G_std[-1]) + x5.matmul(F5_std[-1].matmul(self.S_init.t())) / L25  - ((GL5) * (self.alpha)/L25).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1]))-G_std[-1])*(alpha) / L25))) / 6
            #temporary1 = (self.S02(G_std[-1]) + x0.matmul(F0_std[-1].matmul(self.S_init.t())) / L20 - ((GL0) * (self.alpha) / L20).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1])) - G_std[-1]) * (alpha) / L20))
            #temporary2 = (self.S12(G_std[-1]) + x1.matmul(F1_std[-1].matmul(self.S_init.t())) / L21 - ((GL1) * (self.alpha) / L21).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1])) - G_std[-1]) * (alpha) / L21))
            #temporary3 = (self.S22(G_std[-1]) + x2.matmul(F2_std[-1].matmul(self.S_init.t())) / L22 - ((GL2) * (self.alpha) / L22).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1])) - G_std[-1]) * (alpha) / L22))
            #temporary4 = (self.S32(G_std[-1]) + x3.matmul(F3_std[-1].matmul(self.S_init.t())) / L23 - ((GL3) * (self.alpha) / L23).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1])) - G_std[-1]) * (alpha) / L23))
            #temporary5 = (self.S42(G_std[-1]) + x4.matmul(F4_std[-1].matmul(self.S_init.t())) / L24 - ((GL4) * (self.alpha) / L24).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1])) - G_std[-1]) * (alpha) / L24))
            #temporary6 = (self.S52(G_std[-1]) + x5.matmul(F5_std[-1].matmul(self.S_init.t())) / L25 - ((GL5) * (self.alpha) / L25).matmul(G_std[-1]) - ((G_std[-1].matmul(G_std[-1].t().matmul(G_std[-1])) - G_std[-1]) * (alpha) / L25))
            #temporary1 = self.dropout1(temporary1)
            temporary1 = self.soft_threshold87(temporary1)
            #temporary1 = max((1 - (1 - alpha) / torch.norm(temporary1, 2)) * temporary1, 0)
            #temporary2 = self.soft_threshold87(temporary2)
            #temporary2 = max((1 - (1 - alpha) / torch.norm(temporary2, 2)) * temporary2, 0)
            #temporary3 = self.soft_threshold87(temporary3)
            #temporary3 = max((1 - (1 - alpha) / torch.norm(temporary3, 2)) * temporary3, 0)
            #temporary4 = self.soft_threshold87(temporary4)
            #temporary4 = max((1 - (1 - alpha) / torch.norm(temporary4, 2)) * temporary4, 0)
            #temporary5 = self.soft_threshold87(temporary5)
            #temporary5 = max((1 - (1 - alpha) / torch.norm(temporary5, 2)) * temporary5, 0)
            #temporary6 = self.soft_threshold87(temporary6)
            #temporary6 = max((1 - (1 - alpha) / torch.norm(temporary6, 2)) * temporary6, 0)
            #temp = temporary1 + temporary2 + temporary3 + temporary4 + temporary5 + temporary6
            G.append(temporary1)
            G_std.append(self.bn_input_2(G[-1]))
        ##############################################################################################################
            #  update S
            tmp30 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
            tmp40 = torch.inverse(F0_std[-1].t().matmul(F0_std[-1]))
            tmp50 = G_std[-1].t().matmul(x0.matmul(F0_std[-1]))
            tmp60 = tmp30.matmul(tmp50.matmul(tmp40))

            tmp31 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
            tmp41 = torch.inverse(F1_std[-1].t().matmul(F1_std[-1]))
            tmp51 = G_std[-1].t().matmul(x1.matmul(F1_std[-1]))
            tmp61 = tmp31.matmul(tmp51.matmul(tmp41))

            tmp32 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
            tmp42 = torch.inverse(F2_std[-1].t().matmul(F2_std[-1]))
            tmp52 = G_std[-1].t().matmul(x2.matmul(F2_std[-1]))
            tmp62 = tmp32.matmul(tmp52.matmul(tmp42))

            tmp33 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
            tmp43 = torch.inverse(F3_std[-1].t().matmul(F3_std[-1]))
            tmp53 = G_std[-1].t().matmul(x3.matmul(F3_std[-1]))
            tmp63 = tmp33.matmul(tmp53.matmul(tmp43))

            tmp34 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
            tmp44 = torch.inverse(F4_std[-1].t().matmul(F4_std[-1]))
            tmp54 = G_std[-1].t().matmul(x4.matmul(F4_std[-1]))
            tmp64 = tmp34.matmul(tmp54.matmul(tmp44))

            tmp35 = torch.inverse(G_std[-1].t().matmul(G_std[-1]))
            tmp45 = torch.inverse(F5_std[-1].t().matmul(F5_std[-1]))
            tmp55 = G_std[-1].t().matmul(x5.matmul(F5_std[-1]))
            tmp65 = tmp35.matmul(tmp55.matmul(tmp45))

            tmp6 = (tmp60 + tmp61 + tmp62 + tmp63 + tmp64 + tmp65) / 6
            S.append(tmp6)
        ##############################################################################################################
        return G_std, S, F0_std, F1_std, F2_std, F3_std, F4_std, F5_std

    def train(self):
        self.loss_list = []
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.0)
        #optimizer = Ranger(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=15, verbose=True,
                                                               min_lr=1e-8)  # learning rate decay
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=500, gamma = 0.7)  # learning rate decay
        with tqdm(total=self.epoch, desc="training") as pbar:
            for epoch in range(self.epoch):
                G, S, F0, F1, F2, F3, F4, F5 = self(self.X0_train, self.X1_train, self.X2_train, self.X3_train, self.X4_train, self.X5_train)
                # print(torch.mean(z),torch.mean(w))
                x0_ = G[-1].mm(S[-1].mm(F0[-1].t()))
                x1_ = G[-1].mm(S[-1].mm(F1[-1].t()))
                x2_ = G[-1].mm(S[-1].mm(F2[-1].t()))
                x3_ = G[-1].mm(S[-1].mm(F3[-1].t()))
                x4_ = G[-1].mm(S[-1].mm(F4[-1].t()))
                x5_ = G[-1].mm(S[-1].mm(F5[-1].t()))
                loss = self.my_loss(x0_, self.X0_train) + self.my_loss(x1_, self.X1_train) \
                       + self.my_loss(x2_, self.X2_train) + self.my_loss(x3_, self.X3_train) + self.my_loss(x4_,self.X4_train) + self.my_loss(x5_,self.X5_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = loss.cpu().detach().numpy()
                scheduler.step(loss)
                self.loss_list.append(train_loss/1000000)
                if (optimizer.param_groups[0]['lr'] <= 2e-7):
                    print("early stopped")
                    break
                pbar.set_postfix({'loss': '{0:1.5f}'.format(train_loss)})
                pbar.update(1)

    def compute_error(self):
        G, S, F0, F1, F2, F3, F4, F5 = self(self.X0_train, self.X1_train, self.X2_train, self.X3_train, self.X4_train, self.X5_train)
        # print(torch.mean(z),torch.mean(w))
        x0_ = G[-1].mm(S[-1].mm(F0[-1].t()))
        x1_ = G[-1].mm(S[-1].mm(F1[-1].t()))
        x2_ = G[-1].mm(S[-1].mm(F2[-1].t()))
        x3_ = G[-1].mm(S[-1].mm(F3[-1].t()))
        x4_ = G[-1].mm(S[-1].mm(F4[-1].t()))
        x5_ = G[-1].mm(S[-1].mm(F5[-1].t()))

        err0 = x0_ - self.X0_train
        err0 = torch.pow(err0, 2)
        err0 = torch.sum(err0, 1)
        err0 = torch.mean(err0)
        print("test error on x0:", err0.cpu().detach().numpy())

        err1 = x1_ - self.X1_train
        err1 = torch.pow(err1, 2)
        err1 = torch.sum(err1, 1)
        err1 = torch.mean(err1)
        print("test error on x1:", err1.cpu().detach().numpy())

        err2 = x2_ - self.X2_train
        err2 = torch.pow(err2, 2)
        err2 = torch.sum(err2, 1)
        err2 = torch.mean(err2)
        print("test error on x2:", err2.cpu().detach().numpy())

        err3 = x3_ - self.X3_train
        err3 = torch.pow(err3, 2)
        err3 = torch.sum(err3, 1)
        err3 = torch.mean(err3)
        print("test error on x3:", err3.cpu().detach().numpy())

        err4 = x4_ - self.X4_train
        err4 = torch.pow(err4, 2)
        err4 = torch.sum(err4, 1)
        err4 = torch.mean(err4)
        print("test error on x4:", err4.cpu().detach().numpy())

        err5 = x5_ - self.X5_train
        err5 = torch.pow(err5, 2)
        err5 = torch.sum(err5, 1)
        err5 = torch.mean(err5)
        print("test error on x5:", err5.cpu().detach().numpy())



    def get_ans(self):
        G, S, F0, F1, F2, F3, F4, F5 = self(self.X0_train, self.X1_train, self.X2_train, self.X3_train, self.X4_train, self.X5_train)
        return G[-1], S[-1], F0[-1], F1[-1], F2[-1], F3[-1], F4[-1], F5[-1]

    def plot_loss(self):
        dx = [t for t in range(0, self.epoch)]

        plt.ylabel('Objective value', size=15)
        plt.xlabel('Number of iterations', size=15)
        plt.xlim((1, 40))
        plt.plot(dx, self.loss_list, linewidth=2, c='b', marker='o', mfc='w')  # 在当前对象进行绘图,c为颜色,linewidth为线的宽度
        plt.ticklabel_format(useOffset=False, style='plain')
        #plt.ylim((0, 50000000))

        plt.grid(color="k", linestyle="--", linewidth=0.5)
        plt.savefig("training_loss6.jpg")
        plt.show()


    def clustering(self, z, gnd, k):
        [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(z, gnd, k)
        print("ACC, NMI, Purity, ARI, Fscore, Precision, Recall : ", ACC, NMI, Purity, ARI, Fscore, Precision, Recall)

    def spectral_clustering(self, points, k):
        W = similarity_function(points)
        Dn = np.diag(1 / np.power(np.sum(W, axis=1), -0.5))
        L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
        eigvals, eigvecs = LA.eig(L)
        eigvecs = eigvecs.astype(float)
        indices = np.argsort(eigvals)[:k]
        k_smallest_eigenvectors = normalize(eigvecs[:, indices])

        [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(k_smallest_eigenvectors, gnd, k)
        print("ACC, NMI, Purity, ARI : ", ACC, NMI, Purity, ARI)


if __name__ == "__main__":
    data_dir = "./_multiview datasets" #数据集
    datasetW_dir = "./datasetW" #所有视角的相似性矩阵
    dataGFW_dir = "./dataGFW" #所有视角相似性矩阵的初始化
    datasetGFW_dir = "./datasetGFW" #单个视角的初始化矩阵
    #data_dir = "./data"
    #dataGF_dir = "./dataGF"
    datasetMGF_dir = "./datasetMGF" #单个视角的拉普拉斯矩阵



for i in range(1, 2):
    ### Load data
    #features, gnd = loadData(os.path.join(data_dir, "warpPIE10P.mat"))
    #F = loadF(os.path.join(dataGF_dir, "warpPIE10PF.mat"))
    #G = loadG(os.path.join(dataGF_dir, "warpPIE10PG.mat"))
    #FW = loadFW(os.path.join(dataGFW_dir, "warpPIE10PFW.mat"))
    #GW = loadGW(os.path.join(dataGFW_dir, "warpPIE10PGW.mat"))

    start = time.time()
    features, gnd = loadData(os.path.join(data_dir, "Caltech101-7.mat"))#Caltech101-7, Caltech101-20, HW, NUS-WIDE, Youtube.
    feature0 = features[0][0]
    feature1 = features[0][1]
    feature2 = features[0][2]
    feature3 = features[0][3]
    feature4 = features[0][4]
    feature5 = features[0][5]
    # print("The size of data matrix is: ", feature1.shape)

    #W = loadSIM(os.path.join(datasetW_dir, "ALOIW.mat"))#所有视角的相似性矩阵和

    #G_ = loadG(os.path.join(dataGFW_dir, "ALOIWG.mat"))#所有视角的相似性矩阵和的G

    #W2 = np.dot(G_, (G_.T))

    #W = W1+W2


    F = loadFW(os.path.join(datasetGFW_dir, "Caltech1017WF.mat"))#单个视角的初始化F
    F0 = F[0][0]
    F1 = F[0][1]
    F2 = F[0][2]
    F3 = F[0][3]
    F4 = F[0][4]
    F5 = F[0][5]

    #F_init = (F0 + F1 + F2 + F3)/4 NUSWIDEWG

    G = loadGW(os.path.join(datasetGFW_dir, "Caltech1017WG.mat"))#单个视角初始化矩阵G
    G0 = G[0][0]
    G1 = G[0][1]
    G2 = G[0][2]
    G3 = G[0][3]
    G4 = G[0][4]
    G5 = G[0][5]
    G_w = (G0 + G1 + G2 + G3 + G4 + G5) / 6




    #[ACC, NMI, ARI] = StatisticClustering(W, gnd)
    #print("baseline performance: ")
    #print("ACC, NMI, ARI: ", ACC, NMI, ARI)

    MF = loadMFW(os.path.join(datasetMGF_dir, "Caltech1017MFW.mat"))#单个视角拉普拉斯矩阵LF
    MF0 = MF[0][0]
    MF1 = MF[0][1]
    MF2 = MF[0][2]
    MF3 = MF[0][3]
    MF4 = MF[0][4]
    MF5 = MF[0][5]

    MG = loadMGW(os.path.join(datasetMGF_dir, "Caltech1017MGW.mat"))#单个视角拉普拉斯矩阵LG
    MG0 = MG[0][0]
    MG1 = MG[0][1]
    MG2 = MG[0][2]
    MG3 = MG[0][3]
    MG4 = MG[0][4]
    MG5 = MG[0][5]

    nc = np.unique(gnd).shape[0]
    layer = 5 # 1-9 high
    lr = 1e-4  # 1e-2---4
    epoch = 20  # 10 20 30 40 50
    #if (4<=i<15):
    #    alpha = 1 * 10 ** (-i)
    #    print(i)
    #elif (i==5):
    #    alpha = 1 * 10 ** (0)
    #    print(i)
    #elif (6<=i<=9):
    #    alpha = 1 * 10 ** (i%5)
    #    print(i)
    para = 9e-5  # gamma and delta. If cpp, up.
    alpha = 1e-30  # and beta. If nan, down.

    model0 = Bi_sparse_co_clustering(layer, lr, epoch, feature0/1.0, feature1/1.0, feature2/1.0, feature3/1.0, feature4/1.0, feature5/1.0, F0, F1, F2, F3, F4, F5, G_w, MF0, MG0, MG1, MG2, MG3, MG4, MG5, gnd, para, alpha)
    model0.para_init()
    model0.train()
    for param_tensor in model0.state_dict():
        print(param_tensor, '\t', model0.state_dict()[param_tensor])

    model0.compute_error()
    G00, S00, F000, F100, F200, F300, F400, F500 = model0.get_ans()
    #WW = (G.mm(S).mm(F.t())).cpu().detach().numpy()
    #model.spectral_clustering(WW, nc)
    #model.clustering(G.cpu().detach().numpy(), gnd)
    #G = G.cpu().detach().numpy()
    #io.savemat(dataGFW_dir, {'G': G})
    # model.plot_loss()


    G0 = torch.from_numpy(G0).double()
    G1 = torch.from_numpy(G1).double()
    G2 = torch.from_numpy(G2).double()
    G3 = torch.from_numpy(G3).double()
    G4 = torch.from_numpy(G4).double()
    G5 = torch.from_numpy(G5).double()
    G_s = (G0 + G1 + G2 + G3 + G4 + G5) / 6

    [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(G_s, gnd, nc)
    print("baseline performance: ")
    print("ACC, NMI, Purity, ARI, Fscore, Precision, Recall : ", ACC, NMI, Purity, ARI, Fscore, Precision, Recall)
    print("Our performance1: ")
    model0.clustering(G00.cpu().detach().numpy(), gnd, nc)
    model0.plot_loss()

    end = time.time()
    # endtime = datetime.datetime.now()
    print('The time is:', (end - start))
    #####使用ranger, drogout, 去掉除号