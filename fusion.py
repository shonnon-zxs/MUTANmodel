import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError


class MLBFusion(AbstractFusion):

    def __init__(self, opt):
        super(MLBFusion, self).__init__(opt)
        # Modules
        if 'dim_v' in self.opt:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_h'])
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if 'dim_q' in self.opt:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_h'])
        else:
            print('Warning fusion.py: no question embedding before fusion')
        
    def forward(self, input_v, input_q):
        # visual (cnn features)
        if 'dim_v' in self.opt:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v
        # question (rnn features)
        if 'dim_q' in self.opt:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q
        # hadamard product
        x_mm = torch.mul(x_q, x_v)
        return x_mm


class MutanFusion(AbstractFusion):
    # Mutan主要将V，q融合。
    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__(opt)
        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        # Modules
        if self.visual_embedding:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_hv'])  # 2048-->310
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if self.question_embedding:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_hq'])  # 2400-->310
        else:
            print('Warning fusion.py: no question embedding before fusion')
        
        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.opt['dim_hv'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(self.opt['dim_hq'], self.opt['dim_mm'])  # Linear(310, 510)
            for i in range(self.opt['R'])])

    def forward(self, input_v, input_q):
        # input_v和input_q的维度都是2, (batchsize, d_v) ,统一输入图像和问题的维度
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError
        batch_size = input_v.size(0)

        # 分别处理图像和问题嵌入
        # dropout-->linear(d_v2048/d_q2400--》310)-->tanh
        if self.visual_embedding:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                    x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v

        if self.question_embedding:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                    x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q

        # 秩R的约束，（论文中）Z表示成R个Zr的总和（Z会投影到预测空间y上）。
        # 处理后的图像和问题，使用了对应位的相乘，
        # 使用堆叠求和方式进行相加，最终得到的x_mm相当于文章的Z
        x_mm = []
        for i in range(self.opt['R']): # 用for循环对R个映射独立的进行映射，存储到x_mm

            # 分别处理，图像和问题嵌入
            # dropout-->linear(310--》510)-->tanh

            x_hv = F.dropout(x_v, p=self.opt['dropout_hv'], training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)  # linear后大小变510，
            if 'activation_hv' in self.opt: # tanh
                x_hv = getattr(F, self.opt['activation_hv'])(x_hv)

            x_hq = F.dropout(x_q, p=self.opt['dropout_hq'], training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            if 'activation_hq' in self.opt:
                x_hq = getattr(F, self.opt['activation_hq'])(x_hq)

            #
            x_mm.append(torch.mul(x_hq, x_hv))  # 使用mul（）对应位相乘进行融合，这样融合之后大小不变，但是有R个

        # x_mm([batchsize，510],,,,R个,,,[batchsize，510]),
        x_mm = torch.stack(x_mm, dim=1)  # R个，，在维度1堆起来，
        x_mm = x_mm.sum(1).view(batch_size, self.opt['dim_mm'])  # dim1求和，恢复原来大小（batchsize,510）

        if 'activation_mm' in self.opt:
            x_mm = getattr(F, self.opt['activation_mm'])(x_mm)   # activation_mm = softmax

        # 这就是模型的输出，output，用来预测答案。
        return x_mm

# 输入输出都是三维的，用在第一次融合。
class MutanFusion2d(MutanFusion):

    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion2d, self).__init__(opt,
                                            visual_embedding,
                                            question_embedding)

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 3:  # 输入都是三维的（，，）
            raise ValueError
        batch_size = input_v.size(0)
        weight_height = input_v.size(1)
        dim_hv = input_v.size(2)
        dim_hq = input_q.size(2)
        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_q.is_contiguous():
            input_q = input_q.contiguous()
        # 三维的（，，）变成二维的（，）
        x_v = input_v.view(batch_size * weight_height, self.opt['dim_hv'])
        x_q = input_q.view(batch_size * weight_height, self.opt['dim_hq'])
        # MutanFusion
        x_mm = super().forward(x_v, x_q)
        # 再变成三维的
        x_mm = x_mm.view(batch_size, weight_height, self.opt['dim_mm'])
        return x_mm

