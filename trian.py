import torchvision.datasets
import SEResnet
import Resnet
import CBAMResnet
import UtilFunctions as utf
import my_net
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn


def train(net, train_iter, test_iter, num_epochs, lst, lr, device):
    """用GPU训练模型(在第六章定义)。"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = utf.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1.2],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = utf.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = utf.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            y_temp = y_hat.argmax(axis=1)
            l = loss(y_hat, y)  # + 20 * y_temp[y == 0].sum() / (y[y == 0].numel() + 0.01)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], utf.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        if epoch % 2 == 0:
            test_acc = utf.evaluate_accuracy_gpu(net, test_iter)
            se = utf.evaluate_sensitivity_gpu(net, test_iter, lst)
            fp = utf.evaluate_false_positive_gpu(net, test_iter)
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
                  f'test acc {test_acc:.3f},'
                  f'senstivity {se:.3f},'
                  f'false positive {fp:.3f},'
                  f'proccessed {epoch * 100 / num_epochs:.2f}%')

            # torch.save(net.state_dict(), 'ds_csv_resnet18_adam.params')
            # torch.save(metric, 'ds_csv_metric18_adam')
            # torch.save(epoch, 'ds_csv_epoch18_adam')
        animator.add(epoch + 1, (None, None, test_acc))
    animator.show()


def train_cifar_10():
    transform = transforms.Compose([  # transforms.Resize((64, 64)),
                                      # transforms.Grayscale(1),
                                      transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR10(root='/opt/disk1/YANGSHUBIN/Resnet/homework4/Cifar-10/',
                                              transform=transform, train=True, download=False)
    test_data = torchvision.datasets.CIFAR10(root='/opt/disk1/YANGSHUBIN/Resnet/homework4/Cifar-10/',
                                             transform=transform, train=False, download=False)
    batch_size = 128
    train_iter = data.DataLoader(train_data, batch_size, shuffle=True, sampler=None)
    test_iter = data.DataLoader(test_data, batch_size)

    lr, num_epochs = 0.0005, 50
    lst = []
    net = my_net.new_cbam_net()
    train(net, train_iter, test_iter, num_epochs, lst, lr, utf.try_gpu(0))


def train_my_net():
    # 加载数据集
    # test_root = 'E:\\BaiduNetdiskWorkspace\\workhard\\涡流数据\\Dataset211113\\dataset_im1113\\test_data_im'
    # train_root = 'E:\\BaiduNetdiskWorkspace\\workhard\\涡流数据\\Dataset211113\\dataset_im1113\\train_data_im'
    # train_root = r'dataset_im1113/train_data_im'
    # test_root = r'dataset_im1113/test_data_im'
    # train_root = r'dataset211118_4_deleteSomeDefects/train_data'
    # test_root = r'dataset211118_4_deleteSomeDefects/test_data'
    # train_root = r'dataset211118_4/train_data'
    # test_root = r'dataset211118_4/test_data'
    # train_root = r'dataset211118_3/dataset211118_3_fdt/train_data'
    # test_root = r'dataset211118_3/dataset211118_3_fdt/test_data'
    train_root = r'dataset211213_4_128/train_data'
    test_root = r'dataset211213_4_128/test_data'

    transform = transforms.Compose([transforms.Resize((64, 128)),
                                    # transforms.Grayscale(1),
                                    transforms.ToTensor()])
    train_data = ImageFolder(train_root, transform=transform)
    test_data = ImageFolder(test_root, transform=transform)
    batch_size = 64
    train_iter = data.DataLoader(train_data, batch_size, shuffle=True, sampler=None)
    test_iter = data.DataLoader(test_data, batch_size)

    # 训练
    lr, num_epochs = 0.0005, 50
    net = my_net.new_cbam_net()
    # net = Resnet.resnet18()
    # net = SEResnet.serenet18()
    # net = CBAMResnet.CBAMResnet_18()
    lst = [list(row) for row in test_data.imgs]  # store wrong test datasets in training proccess
    train(net, train_iter, test_iter, num_epochs, lst, lr, utf.try_gpu(0))
    utf.save_wrong_set(lst, save=False)


train_my_net()