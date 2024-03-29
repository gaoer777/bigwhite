import torch
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
from torch.nn import functional as F


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


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxout = self.shared_MLP(self.max_pool(x))
        minout = self.shared_MLP(-self.max_pool(-x))
        return self.sigmoid(maxout - minout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        minout, _ = torch.max(-x, dim=1, keepdim=True)
        minout = -minout
        out = torch.cat([maxout, avgout, minout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel, 4)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class Residual(nn.Module):  # @save定义残差快
    def __init__(self, input_channels, num_channels,use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.cbam = CBAM(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.cbam(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


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


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


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
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def evaluate_sensitivity_gpu(net, data_iter, lst, device=None):  # @save
    """使用GPU计算模型在数据集上的敏感度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    for i, (X, y) in enumerate(data_iter):
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        metric.add(sensitivity(y_hat, y), y[y == 0].numel())
        y_temp = y_hat.argmax(axis=1)
        for j in range(0, y.numel()):
            if y[j] != y_temp[j]:
                lst[j][1] += 1
    return 1 - metric[0] / metric[1]


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


# 定义残差层
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# 定义Resnet18
def resnet_18():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net_resnet18 = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(512, 2))
    return net_resnet18


# 定义Resnet34
def resnet_34():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2_34 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
    b3_34 = nn.Sequential(*resnet_block(64, 128, 4))
    b4_34 = nn.Sequential(*resnet_block(128, 256, 6))
    b5_34 = nn.Sequential(*resnet_block(256, 512, 3))
    net_resnet34 = nn.Sequential(b1, b2_34, b3_34, b4_34, b5_34,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(512, 2))
    return net_resnet34


# train
def train_ch6(net, train_iter, test_iter, num_epochs, lst, lr, device):
    """用GPU训练模型(在第六章定义)。"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            y_temp = y_hat.argmax(axis=1)
            l = loss(y_hat, y) + y_temp[y == 0].sum() / (y[y == 0].numel() + 0.01)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        if epoch % 2 == 0:
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            se = evaluate_sensitivity_gpu(net, test_iter, lst)
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
                  f'test acc {test_acc:.3f},'
                  f'senstivity {se:.3f},'
                  f'proccessed {epoch * 100 / num_epochs:.2f}%')

            # torch.save(net.state_dict(), 'ds_csv_resnet18_adam.params')
            # torch.save(metric, 'ds_csv_metric18_adam')
            # torch.save(epoch, 'ds_csv_epoch18_adam')
        animator.add(epoch + 1, (None, None, test_acc))
    torch.save(net, '3-22-mynet.pth')
    #animator.show()


# 加载数据集
train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220322\dataset\train_data'
test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220322\dataset\test_data'

transform = transforms.Compose([transforms.Resize((64, 64)),
                                #transforms.Grayscale(1),
                                transforms.ToTensor()])
train_data = ImageFolder(train_root, transform=transform)
test_data = ImageFolder(test_root, transform=transform)
batch_size = 32
train_iter = data.DataLoader(train_data, batch_size, shuffle=True, sampler=None)
test_iter = data.DataLoader(test_data, len(test_data.imgs))

def train():
    # 训练
    lr, num_epochs = 0.0005, 50
    net = resnet_18()
    lst = [list(row) for row in test_data.imgs]  # store wrong test datasets in training proccess
    train_ch6(net, train_iter, test_iter, num_epochs, lst, lr, try_gpu(0))
    a = 0
    for i in range(0, len(lst)):
        if lst[a][1] < 10:
            del lst[a]
        else:
            a += 1
    for ele in lst:
        element, indx = ele[0], ele[1]
        elements = element.split('\\')
        new_name = r'D:/gsw/Projects/WOLIU/bigwhite/dataset/Dataset220322/test_wrong_set_modified/'+elements[-2]+'_'+str(indx)+'_'+elements[-1]
        shutil.copy(element, new_name)

def test():
    net = torch.load('3-22-mynet.pth')
    net.cuda()
    test_acc = evaluate_accuracy_gpu(net, test_iter)
    print(test_acc)


if __name__ == '__main__':
    test()