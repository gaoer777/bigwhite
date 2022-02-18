import torchvision.models

import UtilFunctions as utf
import torch
import my_net
from ContrastiveLearning import ContrastiveNetwork
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn


class ContrasNet(nn.Module):
    def __init__(self):
        super(ContrasNet, self).__init__()
        self.backbone = torch.load('SiameseModified-epoch-50.pth')
        self.classifier = nn.Linear(504, 2)

    def forward(self, x):
        out = self.backbone.forward_once(x)
        out = self.classifier(out)
        return torch.sigmoid(out)


def train_step1(net, train_it, test_it, epochs, lr, device):
    # 只训练分类器部分
    for k in net.backbone.parameters():
        k.requires_grad = False

    print('step1: training on', device)
    net.to(device)  # 网络移植到GPU上
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)  # 设置优化器
    loss = nn.CrossEntropyLoss()  # 设置loss function
    for epoch in range(epochs):
        metric = utf.Accumulator(3)  # 设置数据点
        net.train()  # 将网络设置为训练模式
        train_l, train_acc = 0., 0.
        for i, (X, y) in enumerate(train_it):  # 训练过程
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], utf.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

        test_acc = utf.evaluate_accuracy_gpu(net, test_it)  # 每个epoch测试一次测试集的准确率
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f},'
              f'proccessed {epoch * 100 / epochs:.2f}%')


def train_step2(net, train_it, test_it, epochs, lr, device):
    # 训练所有数据
    for param in net.parameters():
        param.requires_grad = True

    print('step2: training on', device)
    net.to(device)  # 网络移植到GPU上
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 设置优化器
    loss = nn.CrossEntropyLoss()  # 设置loss function

    animator = utf.Animator(xlabel='epoch', xlim=[1, epochs],  # 图像中添加数据
                              legend=['train loss', 'train acc', 'test acc'])

    num_batches = len(train_it)
    for epoch in range(epochs):
        metric = utf.Accumulator(3)  # 设置数据点
        net.train()  # 将网络设置为训练模式
        train_l, train_acc = 0., 0.
        for i, (X, y) in enumerate(train_it):  # 训练过程
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], utf.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:  # 每个epoch添加5个数据点
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        test_acc = utf.evaluate_accuracy_gpu(net, test_it)  # 每个epoch测试一次测试集的准确率
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f},'
              f'proccessed {epoch * 100 / epochs:.2f}%')
        animator.add(epoch + 1, (None, None, test_acc))
    animator.show()


def train_step3(net, train_it, test_it, epochs, lr, device):
    # 训练所有数据
    for param in net.parameters():
        param.requires_grad = True

    print('step3: training on', device)
    net.to(device)  # 网络移植到GPU上
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 设置优化器
    loss = nn.CrossEntropyLoss()  # 设置loss function

    animator = utf.Animator(xlabel='epoch', xlim=[1, epochs],  # 图像中添加数据
                            legend=['train loss', 'train acc', 'test acc'])

    num_batches = len(train_it)
    for epoch in range(epochs):
        metric = utf.Accumulator(3)  # 设置数据点
        net.train()  # 将网络设置为训练模式
        train_l, train_acc = 0., 0.
        for i, (X, y) in enumerate(train_it):  # 训练过程
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], utf.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:  # 每个epoch添加5个数据点
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        test_acc = utf.evaluate_accuracy_gpu(net, test_it)  # 每个epoch测试一次测试集的准确率
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f},'
              f'proccessed {epoch * 100 / epochs:.2f}%')
        animator.add(epoch + 1, (None, None, test_acc))
    animator.show()


def main():
    # 加载数据集
    train_root = r'dataset/Dataset211118/dataset/dataset211118_4/train_data'
    test_root = r'dataset/Dataset211118/dataset/dataset211118_4/test_data'
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    # transforms.Grayscale(1),
                                    transforms.ToTensor()])
    train_data = ImageFolder(train_root, transform=transform)
    test_data = ImageFolder(test_root, transform=transform)
    batch_size = 64
    train_iter = data.DataLoader(train_data, batch_size, shuffle=True, sampler=None)
    test_iter = data.DataLoader(test_data, batch_size)

    model = ContrasNet()
    net = my_net.new_cbam_net()
    net2 = torchvision.models.vgg11(pretrained=False, num_classes=2)
    lr = 0.00005
    # train_step1(model, train_iter, test_iter, 20, lr, utf.try_gpu(0))
    # train_step2(model, train_iter, test_iter, 25, lr, utf.try_gpu(0))
    train_step3(net2, train_iter, test_iter, 25, lr, utf.try_gpu(0))


if __name__ == "__main__":
    main()
