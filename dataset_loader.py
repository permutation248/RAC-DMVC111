import os.path
import copy
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp
import torch
from torch.utils.data import Dataset

def load_mat(args):
    """
    加载数据：保持 Input(data_raw) 未归一化，Target(data_target) 归一化
    """
    data_raw = []
    label_y = None
    
    if args.dataset == "Scene15":
        mat = sio.loadmat(os.path.join(args.data_path, 'Scene15.mat'))
        data_raw.append(mat['X1'].astype('float32')) 
        data_raw.append(mat['X2'].astype('float32'))
        label_y = np.squeeze(mat['Y'])
        
    elif args.dataset == "LandUse21":
        mat = sio.loadmat(os.path.join(args.data_path, "LandUse_21.mat"))
        from scipy import sparse
        data_raw.append(sparse.csr_matrix(mat["X"][0, 1]).A)
        data_raw.append(sparse.csr_matrix(mat["X"][0, 2]).A)
        label_y = np.squeeze(mat["Y"]).astype("int")
        
    elif args.dataset == "Reuters":
        mat = sio.loadmat(os.path.join(args.data_path, "Reuters_dim10.mat"))
        data_raw.append(np.vstack((mat["x_train"][0], mat["x_test"][0])))
        data_raw.append(np.vstack((mat["x_train"][1], mat["x_test"][1])))
        label_y = np.squeeze(np.hstack((mat["y_train"], mat["y_test"])))
        
    elif args.dataset == "Caltech101":
        mat = sio.loadmat(os.path.join(args.data_path, "Caltech101.mat"))
        X = mat["X"][0]
        data_raw.append(X[0].T)
        data_raw.append(X[1].T)
        label_y = np.squeeze(mat["gt"]) - 1
        
    elif args.dataset == "NUSWIDE":
        mat = sio.loadmat(os.path.join(args.data_path, "nuswide_deep_2_view.mat"))
        data_raw.append(mat["Img"])
        data_raw.append(mat["Txt"])
        label_y = np.squeeze(mat["label"].T)
    else:
        raise KeyError(f"Unknown Dataset {args.dataset}")

    # 【关键】必须设置 n_sample，否则 evaluate 会报错
    args.n_sample = data_raw[0].shape[0]

    # 1. Input: 复制一份原始数据 (未归一化)
    data_input = [d.copy() for d in data_raw]

    # 2. Target: 对原始引用进行归一化
    data_target = data_raw 
    if args.data_norm == "standard":
        for i in range(len(data_target)):
            data_target[i] = skp.scale(data_target[i])
    elif args.data_norm == "l2-norm":
        for i in range(len(data_target)):
            data_target[i] = skp.normalize(data_target[i])
    elif args.data_norm == "min-max":
        for i in range(len(data_target)):
            data_target[i] = skp.minmax_scale(data_target[i])

    return data_target, data_input, label_y

def load_dataset(args):
    data_target, data_input, targets = load_mat(args)
    dataset = PureMultiViewDataset(data_target, data_input, targets)
    return dataset

class PureMultiViewDataset(Dataset):
    def __init__(self, data_target, data_input, label_y):
        super(PureMultiViewDataset, self).__init__()
        self.data_target = data_target
        self.data_input = data_input
        self.targets = label_y - np.min(label_y)
        self.n_views = len(data_target)
        self.mask = torch.ones(self.n_views).bool()

    def __len__(self):
        return self.data_target[0].shape[0]

    def __getitem__(self, idx):
        target_views = []
        input_views = []
        for i in range(self.n_views):
            target_views.append(torch.tensor(self.data_target[i][idx].astype('float32')))
            input_views.append(torch.tensor(self.data_input[i][idx].astype('float32')))
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return idx, target_views, input_views, self.mask, label, []

# 【新增】简化版的 Sampler，只保留随机种子控制逻辑
class SimpleSampler:
    def __init__(self, dataset, seed: int = 0) -> None:
        self.num_samples = len(dataset)
        self.epoch = 0
        self.seed = seed
        self.indices = torch.arange(self.num_samples)

    def __iter__(self):
        # 这里的逻辑是复现结果的关键：每个 epoch 重新手动设置种子
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.num_samples, generator=g).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch