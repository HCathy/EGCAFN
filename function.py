from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import torch
import math

import random



def choose_top(image, cornor_index, x, y, patch, b, n_top):
    sort = image.reshape(patch * patch, b)
    sort = torch.from_numpy(sort).type(torch.FloatTensor)
    pos = (x - cornor_index[0]) * patch + (y - cornor_index[1])
    Q = torch.sum(torch.pow(sort[pos] - sort, 2), dim=1)
    _, indices = Q.topk(k=n_top, dim=0, largest=False, sorted=True)
    return indices


def gain_neighborhood_pixel(pca_image, point, i, patch, W, H, n_gcn):
    x = point[i, 0]
    y = point[i, 1]
    m = int((patch - 1) / 2)
    _, _, b = pca_image.shape
    if x <= m:
        if y <= m:
            temp_image = pca_image[0:patch, 0:patch, :]
            cornor_index = [0, 0]
        if y >= (H - m):
            temp_image = pca_image[0:patch, H - patch:H, :]
            cornor_index = [0, H - patch]
        if y > m and y < H - m:
            temp_image = pca_image[0:patch, y - m:y + m + 1, :]
            cornor_index = [0, y - m]
    if x >= (W - m):
        if y <= m:
            temp_image = pca_image[W - patch:W, 0:patch, :]
            cornor_index = [W - patch, 0]
        if y >= (H - m):
            temp_image = pca_image[W - patch:W, H - patch:H, :]
            cornor_index = [W - patch, H - patch]
        if y > m and y < H - m:
            temp_image = pca_image[W - patch:W, y - m:y + m + 1, :]
            cornor_index = [W - patch, y - m]
    if x > m and x < W - m:
        if y <= m:
            temp_image = pca_image[x - m:x + m + 1, 0:patch, :]
            cornor_index = [x - m, 0]
        if y >= (H - m):
            temp_image = pca_image[x - m:x + m + 1, H - patch:H, :]
            cornor_index = [x - m, H - patch]
        if y > m and y < H - m:
            temp_image = pca_image[x - m:x + m + 1, y - m:y + m + 1, :]
            cornor_index = [x - m, y - m]
    index = choose_top(temp_image, cornor_index, x, y, patch, b, n_gcn)
    return temp_image, cornor_index, index


# 汇总训练数据和测试数据
def train_and_test_data(pca_image, band, train_point, test_point, true_point, patch, w, h, n_gcn):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    corner_train = np.zeros((train_point.shape[0], 2), dtype=int)
    corner_test = np.zeros((test_point.shape[0], 2), dtype=int)
    corner_true = np.zeros((true_point.shape[0], 2), dtype=int)
    indexs_train = torch.zeros((train_point.shape[0], n_gcn), dtype=int).cuda()
    indexs_test = torch.zeros((test_point.shape[0], n_gcn), dtype=int).cuda()
    indexs_ture = torch.zeros((true_point.shape[0], n_gcn), dtype=int).cuda()
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :], corner_train[i, :], indexs_train[i] = gain_neighborhood_pixel(pca_image, train_point, i,
                                                                                           patch, w, h, n_gcn)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :], corner_test[j, :], indexs_test[j] = gain_neighborhood_pixel(pca_image, test_point, j, patch,
                                                                                        w, h, n_gcn)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :], corner_true[k, :], indexs_ture[k] = gain_neighborhood_pixel(pca_image, true_point, k, patch,
                                                                                        w, h, n_gcn)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")

    return x_train, x_test, x_true, corner_train, corner_test, corner_true, indexs_train, indexs_test, indexs_ture


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
# --------用于计算模型输出与目标值之间的准确率--------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()

def train_epoch(gcn_net, train_loader, criterion, optimizer, indexs_train):
    objs = AvgrageMeter()           # 用于计算损失的平均值
    top1 = AvgrageMeter()           # 用于计算top-1准确率的平均值
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (A, batch_data, batch_target) in enumerate(train_loader):
        batch_A = A.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()

        gcn_pred = gcn_net(batch_data, batch_A, indexs_train)
        loss = criterion(gcn_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(gcn_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


# -------------------------------------------------------------------------------
# validate model
def valid_epoch(gcn_net, valid_loader, criterion, indexs_test):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (A, batch_data, batch_target) in enumerate(valid_loader):
        batch_A = A.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        gcn_pred = gcn_net(batch_data, batch_A, indexs_test)
        loss = criterion(gcn_pred, batch_target)

        prec1, t, p = accuracy(gcn_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre

# -------------------------------------------------------------------------------
# test model
def test_epoch(gcn_net, test_loader, criterion, indexs_test):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (A, batch_data, batch_target) in enumerate(test_loader):
        batch_A = A.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        gcn_pred = gcn_net(batch_data, batch_A, indexs_test)

        loss = criterion(gcn_pred, batch_target)

        prec1, t, p = accuracy(gcn_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def applyPCA(X, numComponents=50):
    newX = X.reshape(-1, X.shape[-1])
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    return newX.reshape(X.shape[:-1] + (numComponents,))



def GET_A2(temp_image, input2, corner, patches, l=3, sigma=1.0):

    input2 = input2.cuda()
    N, h, w, _ = temp_image.shape
    A = np.zeros((N, w * h, w * h), dtype=np.float32)

    k = min(15, w * h - 1)

    for i in range(N):
        x_l = int(corner[i, 0])
        x_r = int(corner[i, 0] + patches)
        y_l = int(corner[i, 1])
        y_r = int(corner[i, 1] + patches)

        patch_feat = input2[x_l:x_r, y_l:y_r, :]
        H, W, C = patch_feat.shape

        feat_flat = patch_feat.reshape(H * W, C)
        dist = torch.cdist(feat_flat, feat_flat)
        sim = torch.exp(-dist ** 2 / (2 * sigma ** 2))

        topk_val, topk_idx = torch.topk(sim, k=k + 1, dim=1)

        adj = torch.zeros_like(sim)
        adj.scatter_(1, topk_idx, topk_val)

        D = adj.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-5))
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        A[i] = adj_norm.cpu().numpy()

    return torch.from_numpy(A).float().cuda()

def pdists_corner(A, sigma=10):
    height, width, band = A.shape
    A = A.reshape(height * width, band)
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    D = torch.exp(-res / (sigma ** 2))
    return D

def normalize(input):
    mean = np.mean(input, axis=(0,1))
    std = np.std(input, axis=(0,1))
    return (input - mean) / (std + 1e-6)

def sampling(proportion, ground_truth, CLASSES_NUM):
    train = {}
    test = {}
    train_num = []
    test_num = []
    labels_loc = {}
    for i in range(CLASSES_NUM):
        indexes = np.argwhere(ground_truth == (i + 1))
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            # nb_val = max(int((1 - proportion) * len(indexes)), 3)
            # print(nb_val)
            if indexes.shape[0] <= 60:
                nb_val = 15
            else:
                nb_val = 30
        else:
            nb_val = 0
        train_num.append(nb_val)
        test_num.append(len(indexes) - nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = train[0]
    test_indexes = test[0]
    for i in range(CLASSES_NUM - 1):
        train_indexes = np.concatenate((train_indexes, train[i + 1]), axis=0)
        test_indexes = np.concatenate((test_indexes, test[i + 1]), axis=0)
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes, train_num, test_num

def get_label(indices, gt_hsi):
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label = gt_hsi[dim_0, dim_1]
    return label


def get_data(dataset):
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(dataset)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
    CLASSES_NUM = max(gt)
    train_indices, test_indices, train_num, test_num = sampling(VALIDATION_SPLIT, gt_hsi, CLASSES_NUM)
    _, total_indices, _, total_num = sampling(1, gt_hsi, CLASSES_NUM)
    y_train = get_label(train_indices, gt_hsi) - 1
    y_test = get_label(test_indices, gt_hsi) - 1
    y_true = get_label(total_indices, gt_hsi) - 1
    return data_hsi,gt_hsi, CLASSES_NUM, train_indices, test_indices, total_indices, y_train, y_test, y_true


def metrics(OA, AA, Kappa, class_acc):
    return {
        'OA': OA,
        'AA': AA,
        'Kappa': Kappa,
        'class_acc': class_acc
    }

def output_metric(target, prediction):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    OA = accuracy_score(target, prediction)
    cm = confusion_matrix(target, prediction)
    AA = np.mean(cm.diagonal() / cm.sum(axis=1))
    Kappa = cohen_kappa_score(target, prediction)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    return OA, AA, Kappa, class_acc

def show_results(results, aggregated=False, dataset_name=None):
    if aggregated:
        oas = [result['OA'] for result in results]
        aas = [result['AA'] for result in results]
        kappas = [result['Kappa'] for result in results]
        class_accs = [result['class_acc'] for result in results]

        mean_oa = np.mean(oas)
        std_oa = np.std(oas)
        mean_aa = np.mean(aas)
        std_aa = np.std(aas)
        mean_kappa = np.mean(kappas)
        std_kappa = np.std(kappas)
        mean_class_acc = np.mean(class_accs, axis=0)
        # 标准差
        std_class_acc = np.std(class_accs, axis=0)
        # 方差
        # var_class_acc = np.var(class_accs, axis=0)
        # var_class_acc = np.var(class_accs, axis=0)
        print("Final Results:")
        print("OA: {:.4f} ± {:.4f}".format(mean_oa, std_oa))
        print("AA: {:.4f} ± {:.4f}".format(mean_aa, std_aa))
        print("Kappa: {:.4f} ± {:.4f}".format(mean_kappa, std_kappa))
        print("Class Accuracies:")
        for i, (mean_acc, std_acc) in enumerate(zip(mean_class_acc, std_class_acc)):
            print("Class {}: {:.4f} ± {:.4f}".format(i + 1, mean_acc, std_acc))

        if dataset_name:
            file_path = f"{dataset_name}_results.txt"
            with open(file_path, 'w') as f:
                f.write("Final Results:\n")
                f.write("OA: {:.4f} ± {:.4f}\n".format(mean_oa, std_oa))
                f.write("AA: {:.4f} ± {:.4f}\n".format(mean_aa, std_aa))
                f.write("Kappa: {:.4f} ± {:.4f}\n".format(mean_kappa, std_kappa))
                f.write("Class Accuracies:\n")
                for i, (mean_acc, std_acc) in enumerate(zip(mean_class_acc, std_class_acc)):
                    f.write("Class {}: {:.4f} ± {:.4f}\n".format(i + 1, mean_acc, std_acc))
    else:
        print("Single Run Results:")
        print("OA: {:.4f}".format(results['OA']))
        print("AA: {:.4f}".format(results['AA']))
        print("Kappa: {:.4f}".format(results['Kappa']))
        print("Class Accuracies:")
        for i, acc in enumerate(results['class_acc']):
            print("Class {}: {:.4f}".format(i + 1, acc))

        if dataset_name:
            file_path = f"{dataset_name}_single_run_results.txt"
            with open(file_path, 'w') as f:
                f.write("Single Run Results:\n")
                f.write("OA: {:.4f}\n".format(results['OA']))
                f.write("AA: {:.4f}\n".format(results['AA']))
                f.write("Kappa: {:.4f}\n".format(results['Kappa']))
                f.write("Class Accuracies:\n")
                for i, acc in enumerate(results['class_acc']):
                    f.write("Class {}: {:.4f}\n".format(i + 1, acc))


################get data######################################################################################################################
# 0.2 训练
def load_dataset(Dataset):
    if Dataset == 'Indian':
        mat_data = sio.loadmat('Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PaviaU':
        uPavia = sio.loadmat('PaviaU.mat')
        gt_uPavia = sio.loadmat('PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Salinas':
        SV = sio.loadmat('Salinas_corrected.mat')
        gt_SV = sio.loadmat('Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)


    if Dataset == 'Botswana':
        mat_data = sio.loadmat('Botswana.mat')
        mat_gt = sio.loadmat('Botswana_gt.mat')
        data_hsi = mat_data['Botswana']
        gt_hsi = mat_gt['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.9975
        # 计算训练集大小
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

