import numpy as np
import torch
import torch.nn.functional as F
import utils
from sklearn.cluster import KMeans
from utils import MetricLogger, SmoothedValue, adjust_learning_config, AverageMeter

def train_one_epoch(model, data_loader_train, optimizer, device, epoch, args=None):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    model.train(True)
    
    # 仅保留这三个基础 Loss 的记录器
    Recon_losses = AverageMeter()
    Intra_losses = AverageMeter()
    Inter_losses = AverageMeter()

    # 注意：这里移除了 all_samples 和 labels 列表，因为没有蒸馏就不需要收集它们了

    for data_iter_step, (ids, samples_ori, samples_noise, mask, label, noisy_idx) in enumerate(data_loader_train):
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        lr = adjust_learning_config(optimizer, smooth_epoch, args)
        mmt = args.momentum

        for i in range(args.n_views):
            samples_ori[i] = samples_ori[i].to(device, non_blocking=True)
            samples_noise[i] = samples_noise[i].to(device, non_blocking=True)
            # 移除了 all_samples[i].append(...)

        with torch.autocast("cuda", enabled=False):
            # 前向传播
            loss_dict = model(samples_ori, samples_noise, epoch < args.start_rectify_epoch, args.singular_thresh)

        rec_loss = loss_dict['l_rec']
        intra_loss = loss_dict['l_intra']
        inter_loss = loss_dict['l_inter']

        # 总 Loss：重构 + 视图内对比 + 视图间对比 (无蒸馏)
        total_loss = rec_loss + intra_loss + inter_loss

        Recon_losses.update(rec_loss.item(), len(ids))
        Intra_losses.update(intra_loss.item(), len(ids))
        Inter_losses.update(inter_loss.item(), len(ids))

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 动量更新 Target Encoder
        model.update_target_branch(mmt)

    log = {
        'Epoch': epoch,
        'lr': lr, 
        'Rec_losses': Recon_losses.avg, 
        'Intra_loss': Intra_losses.avg, 
        'Inter_loss': Inter_losses.avg
    }
    
    if args.print_this_epoch:
        print(log)

    # 移除了 model.update_cluster_centers(...)
    return log


def evaluate(epoch, model, data_loader_test, device, args=None):
    model.eval()
    with torch.no_grad():
        features_all = torch.zeros(args.n_views, args.n_sample, args.embed_dim).to(device)
        labels_all = torch.zeros(args.n_sample, dtype=torch.long).to(device)
        
        for indexs, samples_ori, samples_noise, mask, labels, noisy_idx in data_loader_test:
            for i in range(args.n_views):
                samples_noise[i] = samples_noise[i].to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            
            # 提取特征
            features = model.extract_feature(samples_noise, mask)

            for i in range(args.n_views):
                features_all[i][indexs] = features[i]

            labels_all[indexs] = labels

        # 特征融合 (拼接 -> 归一化)
        features_cat = features_all.permute(1, 0, 2).reshape(args.n_sample, -1)
        features_cat = F.normalize(features_cat, dim=-1).cpu().numpy()
        
        # 执行 K-Means 聚类
        kmeans = KMeans(n_clusters=args.n_classes, random_state=0).fit(features_cat)

        nmi, ari, f, acc = utils.evaluate(np.asarray(labels_all.cpu()), kmeans.labels_)
        result = {"acc": acc, "nmi": nmi, "ari": ari, "f": f}

        # 可视化 t-SNE (仅在首尾 epoch)
        if epoch == 0 or epoch == 99:
            utils.tsne_plot(features_cat, labels_all.cpu().numpy(), args.output_dir + f'/{epoch}.pdf')
            
    return result