import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class BaseModel(torch.nn.Module):
    def __init__(self, n_views, layer_dims, n_classes, drop_rate=0.5, args=None):
        super(BaseModel, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.sigma = args.sigma

        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.online_decoder = nn.ModuleList([FCNDecoder(dim_layer=layer_dims[i][::-1], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])
        self._initialize_target_encoders()

        self.cl = ContrastiveLoss(args.con_temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

        self.cluster_centers = None

    def forward(self, *args, **kwargs):
        return self.forward_impl(*args, **kwargs)

    def _initialize_target_encoders(self):
        for online_encoder, target_encoder in zip(self.online_encoder, self.target_encoder):
            for param_q, param_k in zip(online_encoder.parameters(), target_encoder.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(
                self.online_encoder[i].parameters(), self.target_encoder[i].parameters()
            ):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

class NoisyModel(BaseModel):
    def forward_impl(self, data_ori, data_copy, warm_up, singular_thresh):
        z = [self.online_encoder[i](data_copy[i]) for i in range(self.n_views)]
        x_r = [self.online_decoder[1-i](z[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data_copy[i]) for i in range(self.n_views)]

        l_rec = (F.mse_loss(data_ori[0], x_r[1], reduction='mean') + F.mse_loss(data_ori[1], x_r[0], reduction='mean')) / 2

        l_intra = (self.cl.forward(z[0], z_t[0], None) + self.cl.forward(z[1], z_t[1], None)) / 2
        l_inter = (self.cl.forward(p[0], z_t[1], None) + self.cl.forward(p[1], z_t[0], None)) / 2

        loss = {'l_rec': l_rec, 'l_intra': l_intra, 'l_inter': l_inter}
        return loss

    @torch.no_grad()
    def extract_feature(self, data, mask):
        """
        简化版特征提取（无缺失场景）
        """
        # 编码
        z = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        
        # 投影对齐
        z = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        
        # 归一化
        z = [F.normalize(z[i], dim=-1) for i in range(self.n_views)]
        
        return z

class FCN(nn.Module):
    def __init__(
        self,
        dim_layer=None,
        norm_layer=None,
        act_layer=None,
        drop_out=0.0,
        norm_last_layer=True,
    ):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)

class FCNDecoder(nn.Module):
    def __init__(
        self,
        dim_layer=None,
        norm_layer=None,
        act_layer=None,
        drop_out=0.0,
        norm_last_layer=False
    ):
        super(FCNDecoder, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []

        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), act_layer(), nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = F.normalize(x_q)
        x_k = F.normalize(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.sum() / N
        # loss = nll_loss.mean()
        return loss