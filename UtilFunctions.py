import math
import os
import shutil
import time
import cv2
import numpy
import torch
import datetime
import numpy as np
from torch.utils import data

import my_net
from draw_box_utils import draw_box
import datasets
import matplotlib.pyplot as plt
import torchvision
from torch import nn
from torchvision import transforms

from ObjectDetect0412 import ODAB
from torch.utils.tensorboard import SummaryWriter


class Timer:  # @save
    """记录多次运行时间。"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


class Accumulator:  # @save
    """在`n`个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:  # @save
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(15, 8)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        # use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

    def show(self):
        plt.show()


# 定义计算准确性、敏感性
def evaluate(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    tp = y_hat[(y_hat == 0) & (y == 0)]
    fp = y_hat[(y_hat == 0) & (y == 1)]
    tn = y_hat[(y_hat == 1) & (y == 1)]
    fn = y_hat[(y_hat == 1) & (y == 0)]
    return tp.numel(), fp.numel(), tn.numel(), fn.numel()


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def sensitivity(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float(y_hat[y == 0].sum())


def false_positive(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        a, b = float(y[y_hat == 0].sum()), y_hat[y_hat == 0].numel()
    return float(y[y_hat == 0].sum())


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / (metric[1]+0.0001)


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net.forward_once(X), y), y.numel())
    return metric[0] / metric[1]


def evaluate_sensitivity_gpu(net, data_iter, lst, device=None):
    """使用GPU计算模型在数据集上的敏感度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    for i, (X, y) in enumerate(data_iter):
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        metric.add(sensitivity(y_hat, y), y[y == 0].numel())
        y_temp = y_hat.argmax(axis=1)
        if len(lst) > 0:
            for j in range(0, y.numel()):
                if y[j] != y_temp[j]:
                    lst[j][1] += 1
    return 1 - metric[0] / metric[1]


def evaluate_false_positive_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的FP。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    for i, (X, y) in enumerate(data_iter):
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        y_temp = y_hat.argmax(axis=1)
        metric.add(false_positive(y_hat, y), y_temp[y_temp == 0].numel())

    return metric[0] / (metric[1]+0.0001)


def evaluations(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的混淆矩阵，返回tp, fp, tn, fn, y.numel()。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(5)
    for i, (X, y) in enumerate(data_iter):
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        y_temp = y_hat.argmax(axis=1)
        tp = y_temp[(y_temp == 0) & (y == 0)]
        fp = y_temp[(y_temp == 0) & (y == 1)]
        tn = y_temp[(y_temp == 1) & (y == 1)]
        fn = y_temp[(y_temp == 1) & (y == 0)]
        metric.add(tp.numel(), fp.numel(), tn.numel(), fn.numel(), y.numel())
    return metric


def evaluationsWithAugument(net, data_iter, device=None):  # @save
    """
    TODO:使用增强技术推理
    """
    """使用GPU计算模型在数据集上的混淆矩阵，返回tp, fp, tn, fn, y.numel()。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(5)
    for i, (X, y) in enumerate(data_iter):
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        y_temp = y_hat.argmax(axis=1)
        tp = y_temp[(y_temp == 0) & (y == 0)]
        fp = y_temp[(y_temp == 0) & (y == 1)]
        tn = y_temp[(y_temp == 1) & (y == 1)]
        fn = y_temp[(y_temp == 1) & (y == 0)]
        metric.add(tp.numel(), fp.numel(), tn.numel(), fn.numel(), y.numel())
    return metric


# 设置训练设备
def try_gpu(i=1):  # @save
    """如果gpu(i)存在，则返回cudai，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# 使用matplotlib绘制，loss、train_acc、test_acc
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def save_wrong_set(lst, save=True):
    """统计训练时出现错误的数据集"""
    if save:
        a = 0
        for i in range(0, len(lst)):
            if lst[a][1] < 5:
                del lst[a]
            else:
                a += 1
        print(lst)
        for ele in lst:
            element, indx = ele[0], ele[1]
            elements = element.split('/')
            new_name = '/home/gsw/Desktop/test_wrong_set/' + elements[-2] + '_' + str(indx) + '_' + elements[-1]
            shutil.copy(element, new_name)


def compute_loss(p, targets, anchors, l_box=5, l_obj=50):  # predictions, targets, weight of box loss/object loss
    device = p[0].device
    lbox = torch.zeros(1, device=device)  # Tensor(0) 预测的box的损失
    lobj = torch.zeros(1, device=device)  # Tensor(0) 预测的目标损失
    tcls, tboxs, indices, anchors_vec = build_targets(p, targets, anchors)  # targets
    # anchors_vec = anchors[0] / torch.Tensor([[32, 8]])  # 转换为相对尺寸
    # anchors_vec = anchors_vec.to(device)

    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device), reduction=red)

    # per output
    for i in range(3):
        tbox = tboxs[i].to(device)
        b, a, gj, gi = indices[i]  # image_idx, anchor_idx, grid_y, grid_x
        tobj = torch.zeros_like(p[i][..., 0], device=device)  # target obj

        nb = b.shape[0]  # number of positive samples
        if nb:
            # 对应匹配到正样本的预测信息
            ps = p[i][b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = ps[:, :2].sigmoid()
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors_vec[i].cuda()
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox, x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # Obj
            tobj[b, a, gj, gi] = 1  # giou ratio
            lobj += BCEobj(p[i][..., 4], tobj)  # obj loss

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
        else:
            lbox += 0
            lobj += BCEobj(p[i][..., 4], tobj).cuda()

    # 乘上每种损失的对应权重
    lbox *= l_box
    lobj *= l_obj

    # loss = lbox + lobj
    return {"box_loss": lbox,
            "obj_loss": lobj}


def build_targets(p, targets, anchor, iou_t=0.30):
    # Build targets for compute_loss(), input targets(image_idx,class,x,y,w,h)
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

    for i in range(3):
        # 注意anchor_vec是anchors缩放到对应特征层上的尺度
        anchors = torch.Tensor(anchor[i])
        # 图像缩放的尺度
        scale_w = 512 / p[i].shape[3]
        scale_h = 64 / p[i].shape[2]
        # p[i].shape: [batch_size, 3, grid_h, grid_w, num_params]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        na = 1  # number of anchors
        # [3] -> [3, 1] -> [3, nt]
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        a, t = [], targets
        if nt:  # 如果存在target的话
            # 通过计算anchor模板与所有target的wh_iou来匹配正样本
            # j: [3, nt] , iou_t = 0.20 , 计算的是每个anchor（1个）和每个target框（groundtruth框）的粗略的iou
            j = wh_iou(anchors, t[:, 4:6]) > iou_t  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            # t.repeat(na, 1, 1): [nt, 6] -> [3, nt, 6]
            # 获取正样本对应的anchor模板与target信息
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

        # Define
        # long等于to(torch.int64), 数值向下取整
        b, c = t[:, :2].long().T  # image_idx, class
        gxy = t[:, 2:4]  # grid xy
        gxy[:, 0] /= scale_w
        gxy[:, 1] /= scale_h
        gwh = t[:, 4:6]  # grid wh
        gwh[:, 0] /= scale_w
        gwh[:, 1] /= scale_h
        gij = gxy.long()  # 匹配targets所在的grid cell左上角坐标
        gi, gj = gij.T  # grid xy indices  gi->x,gj->y
        anchors_vec = anchors / torch.tensor([scale_w, scale_h])

        # Append
        # gain[3]: grid_h, gain[2]: grid_w
        # image_idx, anchor_idx, grid indices(y, x)
        indices.append((b, a, gj, gi))
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # gt box相对anchor的x,y偏移量以及w,h
        anch.append(anchors_vec[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def get_targets(labels):
    """
    返回所有图像中的真实目标的box：
    targets[image_idx,obj,x,y,w,h]
    其中obj为1，表示为检测的对象，x和y为box的中心点坐标，w和h为box的宽和高
    x、y、w、h都为绝对值
    """
    targets = numpy.zeros((1, 5))  # (image_idx,class,x,y,w,h)
    image_idx = 0
    for path in labels:
        try:
            os.path.exists(path)
            with open(path, 'r') as f:
                if os.path.getsize(path):
                    l = np.array([x.lstrip('[').rstrip(']').split(',') for x in f.read().splitlines()], dtype=np.float32)  # l(l_x, u_y, w, h, area)
                    l[:, :2] += l[:, 2:4]/2  # 将l_x, u_y转换为中心点的坐标
                    l[:, 4] = image_idx  # 将第五列的数据设置为图片张数
                    info = l[:, [4, 0, 1, 2, 3]]
                    targets = np.concatenate([targets, info], axis=0)
                image_idx += 1

        except Exception as e:
            print("labels file not found!{},{}".format(path, e))
    targets = np.insert(targets, 1, values=1, axis=1)
    targets = np.delete(targets, 0, axis=0)

    return torch.from_numpy(targets)


def get_targets_COCO(targets, n):
    """
    返回所有图像中的真实目标的box：
    targets[image_idx,obj,x,y,w,h]
    其中obj为1，表示为检测的对象，x和y为box的中心点坐标，w和h为box的宽和高
    x、y、w、h都为绝对值
    """
    image_idx = 0
    tg = numpy.zeros((1, 5))  # (image_idx,class,x,y,w,h)
    i = 0
    for target in targets:
        target = target[:n[i], :]
        i += 1
        target[:, 0] = image_idx
        tg = np.concatenate([tg, target], axis=0)
        image_idx += 1

    tg = np.insert(tg, 1, values=1, axis=1)
    tg = np.delete(tg, 0, axis=0)
    tg[:, 2:] = xyxy2xywh(tg[:, 2:])

    return torch.from_numpy(tg)


def getOneTarget(labels):
    """
    返回所有图像中的真实目标的box：
    targets[image_idx,obj,x,y,w,h]
    其中obj为1，表示为检测的对象，x和y为box的中心点坐标，w和h为box的宽和高
    x、y、w、h都为绝对值
    """
    targets = []
    path = labels
    if os.path.exists(path):
        with open(path, 'r') as f:
            targets = np.array([x.lstrip('[').rstrip(']').split(' ') for x in f.read().splitlines()], dtype=np.float32)
    return targets


def non_max_suppression_for_single(prediction, conf_thres=0.7, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=10):
    """
    Performs  Non-Maximum Suppression on inference results
    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 4, 512  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = 1  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference 遍历每张图片
        # Apply constraints 去除一些背景、小目标等
        x = x[x[:, 4] > conf_thres]  # confidence 根据obj confidence虑除背景目标
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # If none remain process next image
        if not x.shape[0]:
            continue

        # # Compute conf
        # x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)左上角和右下角坐标
        box = xywh2xyxy(x)

        # Detections matrix nx6 (xyxy, conf, cls)
        # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
        x = box

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        # x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        boxes, scores = x[:, :4].clone(), x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_num]  # 最多只保留前max_num个目标信息
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        # if (time.time() - t) > time_limit:
        #     break  # time limit exceeded

    return output


def non_max_suppression_for_batch(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=100):
    """
    Performs  Non-Maximum Suppression on inference results

    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10000.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference 遍历每张图片
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence 根据obj confidence虑除背景目标
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        # x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x)

        x = box

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        boxes, scores = x[:, :4].clone(), x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_num]  # 最多只保留前max_num个目标信息
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    y[:, 4] = x[:, 4]
    return y


def xywh2xyxy_(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param coords: 预测的box信息
    :param img0_shape: 缩放前的图像尺度
    :param ratio_pad: 缩放过程中的缩放比例以及pad
    :return:
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def letterbox(img: np.ndarray,
              new_shape=(63, 512),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img: 输入的图像numpy格式
    :param new_shape: 输入网络的shape
    :param color: padding用什么颜色填充
    :param auto:
    :param scale_fill: 简单粗暴缩放到指定大小
    :param scale_up:  只缩小，不放大
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def train(net, train_iter, test_iter, num_epochs, lr, weight, writer, tag, tag_model_save, device):
    """用GPU训练模型。"""
    print(f'net : {tag}  training on', device)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.956)  # 0.956^100 = 0.01111

    loss = nn.CrossEntropyLoss(weight=weight)

    now = datetime.datetime.now()
    timer = now.strftime("%Y-%m-%d")

    train_l, train_acc, num_batches = 0., 0., len(train_iter)
    highest_acc, highest_recall, highest_F1, highest_Precision = 0., 0., 0., 0.

    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = Accumulator(3)
        net.train()
        # 一个epoch的训练
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]  # average loss
            train_acc = metric[1] / metric[2]  # average accuracy
            if (i + 1) % ((num_batches // 5)+1) == 0 or i == num_batches - 1:  # log values every five iter
                writer.add_scalar(f'{tag}/train_loss', train_l, global_step=epoch + (i + 1) / num_batches)
                writer.add_scalar(f'{tag}/train_acc', train_acc, global_step=epoch + (i + 1) / num_batches)
        # renew learning rate
        scheduler.step()

        # 评价指标 every two epoch
        if epoch % 2 == 0:
            indexes = evaluations(net, test_iter)

            test_acc = (indexes[0] + indexes[2]) / indexes[4]
            Recall = indexes[0] / (indexes[0] + indexes[3] + 0.0001)
            Precision = indexes[0] / (indexes[0] + indexes[1] + 0.0001)
            F1 = 2 * Precision * Recall / (Precision + Recall)
            highest_acc = max(highest_acc, test_acc)
            highest_recall = max(highest_recall, Recall)
            highest_Precision = max(highest_Precision, Precision)

            writer.add_scalar(f'{tag}/test_acc', test_acc, global_step=epoch)
            writer.add_scalar(f'{tag}/Recall', Recall, global_step=epoch)
            writer.add_scalar(f'{tag}/F1', F1, global_step=epoch)
            writer.add_scalar(f'{tag}/Precision', Precision, global_step=epoch)

            print(f'loss {train_l:.4f}, train acc {train_acc*100:.2f}, '
                  f'test acc {test_acc*100:.2f}%,'
                  f'Recall {Recall*100:.2f}%,'
                  f'F1 {F1:.3f},'
                  f'epoch {epoch}')

            # save_model when model are stable and have a greater result
            if F1 > highest_F1:
                highest_F1 = max(highest_F1, F1)
                torch.save(net.state_dict(), f"./run_log/{tag_model_save}/"  # save as: eg. 2022-10-17-mynet-156.pth
                                             f"{timer}-{tag}-{epoch}-F1{highest_F1*100:.0f}.pth")

    print(f'highest_acc: {highest_acc:.4f}, highest_recall: {highest_recall:.4f}, '
          f'highest_F1: {highest_F1:.4f}, highest_Precision: {highest_Precision:.4f}')


def customLoss(y_hat, y):
    """
    自定义loss
    """
    loss = nn.CrossEntropyLoss(reduction='None')
    loss_value = loss(y_hat, y)


def test_based_on_vote(net, test_iter):
    """基于投票机制的模型测试"""
    for i, (X, Y) in enumerate(test_iter):
        X = X.cuda()
        Y = Y.cuda()
        y0 = net[0](X)
        y_predicted0 = y0.argmax(axis=1)
        y1 = net[1](X)
        y_predicted1 = y1.argmax(axis=1)
        y2 = net[2](X)
        y_predicted2 = y2.argmax(axis=1)
        y_pred = y_predicted0 + y_predicted1 + y_predicted2
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == 2] = 1
        tp = y_pred[(y_pred == 0) & (Y == 0)].numel()
        fp = y_pred[(y_pred == 0) & (Y == 1)].numel()
        tn = y_pred[(y_pred == 1) & (Y == 1)].numel()
        fn = y_pred[(y_pred == 1) & (Y == 0)].numel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        fpr = fn / (tp + fn)
        net0_acc = y_predicted0[y_predicted0 == Y].numel() / Y.numel()
        net1_acc = y_predicted1[y_predicted1 == Y].numel() / Y.numel()
        net2_acc = y_predicted2[y_predicted2 == Y].numel() / Y.numel()
        print(f'acc: {acc*100:.2f}%,  '
              f'fpr: {fpr*100:.2f}%   '
              f'acc: {net0_acc*100:.2f}%,  '
              f'acc: {net1_acc*100:.2f}%,  '
              f'acc: {net2_acc*100:.2f}%,  ')


def test_model(net, test_iter):
    """测试模型的准确率及FPR"""
    indexes = evaluations(net, test_iter)
    test_acc = (indexes[0] + indexes[2]) / indexes[4]
    FPR = indexes[3] / (indexes[0] + indexes[3])
    print(f'acc: {test_acc*100:.2f}%,  '
          f'fpr: {FPR*100:.2f}% ')


def eval_detected(pred, target, matrics, iou_th=0.5):
    """
    测试模型的准确率
    matrics : TP, FP, FN, TN = 0, 0, 0, 0
    """

    if target.shape[0] == 0:
        if pred is not None:
            matrics[1] += pred.shape[0]
        return

    if pred is None:
        matrics[2] += target.shape[0]
        return

    targ = torch.zeros(target.shape[0], 5)
    targ[:, 0:4] = target[:, 2:]
    target = xywh2xyxy(targ)
    target = target[:, 0:4]

    right = 0
    for box in target:
        for pred_box in pred:
            if bbox_iou(box, pred_box) > iou_th:
                matrics[0] += 1
                right += 1
                continue

    matrics[0] += right
    matrics[1] += max(0, pred.shape[0] - right)
    matrics[2] += target.shape[0] - right


def trainDetect(net, device, test_root, save_path, save_flag):
    category_index = {1: "defect"}
    matric = [0, 0, 0, 0]
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.LoadCOCOImagesAndLabels(test_root, transform=transform)
    train_iter = data.DataLoader(train_data, 1)

    net.eval()
    with torch.no_grad():
        # init 创建一张图片初始化模型
        # img = torch.zeros((1, 3, 63, 512), device=device)
        # model(img)

        # 读取测试图片
        for i, (img, target, z) in enumerate(train_iter):
            # 得到预测结果
            img = img.to(device)
            pred = net(img)  # only get inference result

            pred = torch.cat([pred[0][0], pred[1][0], pred[2][0]], dim=1)
            # 对预测结果进行过滤，采用非极大值抑制的方法
            # conf_thres滤除一部分低于这个阈值的目标框，代表是否为目标的置信度
            pred = non_max_suppression(pred, conf_thres=0.01, max_num=10,
                                       iou_thres=0.2, multi_label=True)[0]

            # 评价指标
            target = get_targets_COCO(target, z)
            target = torch.squeeze(target, 0)
            eval_detected(pred, target, matric)

            if pred is None:
                print("No target detected.")
                continue

            bboxes = pred[:, :4].detach().cpu().numpy()
            scores = pred[:, 4].detach().cpu().numpy()
            classes = np.ones(pred.shape[0], dtype=np.int32)

            # 将坐标画在原图上
            img = torch.squeeze(img, 0)
            img = torch.clamp(img*255, 0, 255).cpu().int().numpy()
            img = img.transpose(1, 2, 0).astype(np.uint8)
            img_o = draw_box(img, bboxes, classes, scores, category_index)
            # plt.imshow(img_o)
            # plt.show()

            s_path = save_path + '/' + save_flag + '_' + str(i) + '.png'
            img_o.save(s_path)

    precision = matric[0] / (matric[0] + matric[1] + 1)
    recall = matric[0] / (matric[0] + matric[2] + 1)
    # print(f"precision: {precision*100:.2f}%,  recall: {recall*100:.2f}%")

    return img_o, precision, recall


def loadSingleImage(path):
    """
    输入图像路径，返回可以直接送进网络的图像矩阵
    """
    img = cv2.resize(cv2.imread(path), (64, 64))  # BGR

    # Convert BGR to RGB, and HWC to CHW(3x512x512)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cuda').float()
    img /= 255.0  # scale (0, 255) to (0, 1)
    img = img.unsqueeze(0)  # add batch dimension
    return img

def viz(module, input, output):
    x = output[0]
    # 最多显示4张图
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i + 1)
        plt.imshow(x[i].cpu().detach().numpy())
        plt.colorbar()
    plt.show()


def lookintoNet():
    # t = transforms.Compose([transforms.ToPILImage(),
    #                         transforms.Resize((64, 64)),
    #                         transforms.ToTensor(),
    #                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         #                      std=[0.229, 0.224, 0.225])
    #                         ])

    t = transforms.Compose([transforms.ToTensor(),
                            # transforms.RandomRotation(180, expand=1),
                            # transforms.RandomVerticalFlip(),
                            # transforms.RandomHorizontalFlip(),
                            transforms.Resize((64, 64))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net1 = my_net.new_cbam_net(kernel_size=3, padding=1)
    # net1 = models.resnet18(pretrained=False, num_classes=2)
    net1.load_state_dict(torch.load('./models/experiment3/2022-10-10-my_net-136.pth'))
    # net1.load_state_dict(torch.load('./models/experiment3/2022-09-30-resnet18-66.pth'))

    model = net1.to(device).eval()

    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size[0] > 1 and m.stride[0] > 1:
            m.register_forward_hook(viz)

    path_defects = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\new_rgb2\defects'
    imgps = os.listdir(path_defects)

    # # LOAD IMAGE METHOD1
    # transform_test = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Resize((64, 64))])
    # test_data = ImageFolder(path_defects, transform=transform_test)
    # test_iter = data.DataLoader(test_data, 1)

    # LOAD IMAGE METHOD2
    for imgp in imgps:
        path = path_defects + '/' + imgp
        img = loadSingleImage(path)
        y = model(img.cuda())
        print(y)

    # for i, (img, l) in enumerate(test_iter):
    #     y = model(img.cuda())
    #     print(y)