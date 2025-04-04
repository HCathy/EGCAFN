import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
from EGCAFN import EGCAFN
from function import show_results, normalize, get_data, GET_A2, train_and_test_data, train_epoch, valid_epoch, \
    applyPCA

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'Salinas','KSC','Botswana'], default='Salinas', help='dataset to use')
parser.add_argument("--num_run", type=int, default=1)
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--patches', type=int, default=9, help='number of patches')
parser.add_argument('--n_gcn', type=int, default=15, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=50, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# 设置随机种子，保证结果可复现
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

# 准备数据
# input, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true = get_data(args.dataset)
input,gt_label, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true = get_data(args.dataset)
input = applyPCA(input, numComponents=args.pca_band)
input_normalize = normalize(input)
height, width, band = input_normalize.shape
print("height={0}, width={1}, band={2}".format(height, width, band))



# 获取训练和测试数据
x_train_band, x_test_band, x_true_band, corner_train, corner_test, corner_true, indexs_train, indexs_test, indexs_true = train_and_test_data(
    input_normalize, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, w=height, h=width,
    n_gcn=args.n_gcn)

input2 = torch.from_numpy(input_normalize).type(torch.FloatTensor)
A_train = GET_A2(x_train_band, input2, corner=corner_train, patches=args.patches, l=3, sigma=10)
x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
Label_train = Data.TensorDataset(A_train, x_train, y_train)

A_test = GET_A2(x_test_band, input2, corner=corner_test, patches=args.patches, l=3, sigma=10)
x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
Label_test = Data.TensorDataset(A_test, x_test, y_test)

x_true = torch.from_numpy(x_true_band).type(torch.FloatTensor)
y_true = torch.from_numpy(y_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(x_true, y_true)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)

results = []


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

# 保存预测结果为 JPG 图片
for run in range(args.num_run):
    print(f"Run {run + 1}/{args.num_run}")

    # 初始化模型
    best_OA2 = 0.0
    best_AA_mean2 = 0.0
    best_Kappa2 = 0.0

    gcn_net = EGCAFN(height, width, band, num_classes, dim=64)
    gcn_net = gcn_net.cuda()

    # 损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    # 优化器
    optimizer = torch.optim.Adam(gcn_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)

    print("Start training")
    tic = time.time()
    train_time = 0
    total_valid_time = 0
    for epoch in range(args.epoches):
        scheduler.step()
        gcn_net.train()
        train_start = time.time()
        train_acc, train_obj, tar_t, pre_t = train_epoch(gcn_net, label_train_loader, criterion, optimizer,
                                                         indexs_train)
        train_end = time.time()
        train_sum = train_end - train_start
        train_time += train_sum
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(epoch + 1, train_obj, train_acc))

        if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1) and epoch >= args.epoches * 0.6:
            gcn_net.eval()
            valid_start = time.time()  # 记录验证开始时间
            tar_v, pre_v = valid_epoch(gcn_net, label_test_loader, criterion, indexs_test)
            valid_end = time.time()  # 记录验证结束时间

            valid_time = valid_end - valid_start  # 计算本次验证时间
            total_valid_time += valid_time  # 累加到总验证时间

            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            if OA2 >= best_OA2 and AA_mean2 >= best_AA_mean2 and Kappa2 >= best_Kappa2:
                best_OA2 = OA2
                best_AA_mean2 = AA_mean2
                best_Kappa2 = Kappa2
                run_results = metrics(best_OA2, best_AA_mean2, best_Kappa2, AA2)


    print('训练时间：', train_time)
    toc = time.time()
    total_time = toc - tic
    print('总时间：', total_time)

if args.num_run > 1:
    show_results(results, aggregated=True, dataset_name=args.dataset)
