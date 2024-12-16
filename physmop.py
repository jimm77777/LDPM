import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
from models.siMLPe_mlp import build_mlps
import config
import constants


import torch
import torch.nn as nn
import numpy as np

class Fusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Fusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, motion_pred_data, motion_pred_physics, motion_feats, t):
        # 将数据驱动、物理驱动和运动特征的结果结合
        motion_fusion_feats = torch.cat([motion_pred_data, motion_pred_physics, motion_feats], dim=-1)

        if t.dim() == 1:
            t = t.unsqueeze(1)  # 在维度 1 处增加一个维度

        if t.size(1) != motion_fusion_feats.size(1):
            t = torch.nn.functional.interpolate(t.unsqueeze(-1), size=motion_fusion_feats.size(1), mode='nearest').squeeze(-1)
            t = t.unsqueeze(-1)  # 再次扩展至三维

        motion_fusion_feats = self.fc(motion_fusion_feats) + t
        return motion_fusion_feats


class Regression(nn.Module):
    def __init__(self, input_dim, output_dim, physics=True, data=True):
        super(Regression, self).__init__()
        self.physics = physics
        self.data = data

        # 数据驱动分支
        self.motion_fc_in = nn.Linear(input_dim, output_dim)
        self.motion_fc_out = nn.Linear(output_dim, output_dim)

        # 物理驱动分支
        self.physics_fc_in = nn.Linear(input_dim, output_dim)
        self.physics_fc_out = nn.Linear(output_dim, output_dim)

        # 融合模块
        self.fusion_net = Fusion(output_dim * 3, output_dim)

    def physics_forward(self, motion_feats):
        # 简化的物理学预测模型
        x = self.physics_fc_in(motion_feats)
        x = torch.relu(x)
        pred_physics = self.physics_fc_out(x)
        return pred_physics

    def data_forward(self, motion_feats):
        # 数据驱动的预测模型
        x = self.motion_fc_in(motion_feats)
        x = torch.relu(x)
        pred_data = self.motion_fc_out(x)
        return pred_data

    def forward(self, motion_input, t):
        # 数据驱动结果
        if self.data:
            motion_pred_data = self.data_forward(motion_input)
        else:
            motion_pred_data = torch.zeros_like(motion_input)

        # 物理驱动结果
        if self.physics:
            motion_pred_physics = self.physics_forward(motion_input)
        else:
            motion_pred_physics = torch.zeros_like(motion_input)

        # 将两者结合
        motion_fusion = self.fusion_net(motion_pred_data, motion_pred_physics, motion_input, t)
        return motion_fusion


class PhysMoP(nn.Module):
    def __init__(self, input_dim, output_dim, hist_length, physics=True, data=True, fusion=False, device='cpu'):
        super(PhysMoP, self).__init__()

        self.device = torch.device(device)
        self.hist_length = hist_length

        self.input_dim = input_dim  # 添加输入维度
        self.output_dim = output_dim  # 添加输出维度
        
        self.fusion = fusion  # 添加这个参数
        self.regressor = Regression(input_dim, output_dim, physics, data)

    def forward_dynamics(self, motion_input, t):
        return self.regressor(motion_input, t)


# 模拟输入和参数配置
input_dim = 64
output_dim = 64
hist_length = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 PhysMoP 实例
phys_mop = PhysMoP(input_dim=input_dim, output_dim=output_dim, hist_length=hist_length).to(device)

# 模拟输入数据 (batch_size, hist_length, input_dim)
motion_input = torch.randn(32, hist_length, input_dim).to(device)
time_steps = torch.arange(0, 1, step=1.0 / motion_input.size(0)).to(device)

# 前向计算
motion_pred = phys_mop.forward_dynamics(motion_input, time_steps)
print(motion_pred.shape)  # 输出的预测形状
