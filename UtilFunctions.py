import math
import os
import shutil
import time

import cv2
import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch import nn


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
        metric.add(accuracy(net(X), y), y.numel())
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


def compute_loss(p, targets, l_box=5, l_obj=10):  # predictions, targets, weight of box loss/object loss
    device = p[0].device
    lbox = torch.zeros(1, device=device)  # Tensor(0) 预测的box的损失
    lobj = torch.zeros(1, device=device)  # Tensor(0) 预测的目标损失
    tcls, tbox, indices, anchors = build_targets(p, targets)  # targets
    anchors_vec = anchors[0] / torch.Tensor([[32, 8]])  # 转换为相对尺寸
    anchors_vec = anchors_vec.to(device)
    tbox = tbox[0].to(device)
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device), reduction=red)

    # per output
    b, a, gj, gi = indices  # image_idx, anchor_idx, grid_y, grid_x
    tobj = torch.zeros_like(p[..., 0], device=device)  # target obj

    nb = b.shape[0]  # number of positive samples
    if nb:
        # 对应匹配到正样本的预测信息
        ps = p[b, a, gj, gi]  # prediction subset corresponding to targets

        # GIoU
        pxy = ps[:, :2].sigmoid()
        pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors_vec
        pbox = torch.cat((pxy, pwh), 1)  # predicted box
        giou = bbox_iou(pbox.t(), tbox, x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
        lbox += (1.0 - giou).mean()  # giou loss

        # Obj
        tobj[b, a, gj, gi] = 1  # giou ratio
        lobj += BCEobj(p[..., 4], tobj)  # obj loss

        # Append targets to text file
        # with open('targets.txt', 'a') as file:
        #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    else:
        lbox = torch.tensor(0, dtype=torch.float32)
        lobj += BCEobj(p[..., 4], tobj)

    # 乘上每种损失的对应权重
    lbox *= l_box
    lobj *= l_obj

    # loss = lbox + lobj
    return {"box_loss": lbox,
            "obj_loss": lobj}


def build_targets(p, targets):
    # Build targets for compute_loss(), input targets(image_idx,class,x,y,w,h)
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

    # 注意anchor_vec是anchors缩放到对应特征层上的尺度
    anchors = torch.Tensor([[200, 60], [50, 15], [100, 40]])
    # p[i].shape: [batch_size, 3, grid_h, grid_w, num_params]
    gain[2:] = torch.tensor(p.shape)[[3, 2, 3, 2]]  # xyxy gain
    na = 3  # number of anchors
    # [3] -> [3, 1] -> [3, nt]
    at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

    # Match targets to anchors
    a, t = [], targets
    if nt:  # 如果存在target的话
        # 通过计算anchor模板与所有target的wh_iou来匹配正样本
        # j: [3, nt] , iou_t = 0.20 , 计算的是每个anchor（3个）和每个target框（groundtruth框）的粗略的iou
        j = wh_iou(anchors, t[:, 4:6]) > 0.2  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
        # t.repeat(na, 1, 1): [nt, 6] -> [3, nt, 6]
        # 获取正样本对应的anchor模板与target信息
        a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

    # Define
    # long等于to(torch.int64), 数值向下取整
    b, c = t[:, :2].long().T  # image_idx, class
    gxy = t[:, 2:4]  # grid xy
    gxy[:, 0] /= 32
    gxy[:, 1] /= 8
    gwh = t[:, 4:6]  # grid wh
    gwh[:, 0] /= 32
    gwh[:, 1] /= 8
    gij = gxy.long()  # 匹配targets所在的grid cell左上角坐标
    gi, gj = gij.T  # grid xy indices  gi->x,gj->y

    # Append
    # gain[3]: grid_h, gain[2]: grid_w
    # image_idx, anchor_idx, grid indices(y, x)
    indices = (b, a, gj, gi)
    tbox.append(torch.cat((gxy - gij, gwh), 1))  # gt box相对anchor的x,y偏移量以及w,h
    anch.append(anchors[a])  # anchors
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
                    l = np.array([x.lstrip('[').rstrip(']').split(',') for x in f.read().splitlines()],
                                 dtype=np.float32)  # l(l_x, u_y, w, h, area)
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


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=10):
    """
    Performs  Non-Maximum Suppression on inference results

    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 4, 512  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
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
