import numpy as np
import torch
import my_net
import UtilFunctions as utf
import datasets
import ObjectDetect0412
from torch.utils import data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter


def experiment1():
    """
    实验1：
    input：未融合的数据，FDA/FDT/FAT
    network：浅层卷积网络/经典网络/自己设计的网络
    """

    # 数据集加载
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_fda\train_data'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_fda\test_data'
    # train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_fdt\train_data'
    # test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_fdt\test_data'
    # train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_fat\train_data'
    # test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_fat\test_data'
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = ImageFolder(train_root, transform=transform)
    test_data = ImageFolder(test_root, transform=transform)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)

    # 网络加载
    net1 = models.resnet18(pretrained=False, num_classes=2)
    net2 = models.densenet121(pretrained=False, num_classes=2)
    net3 = my_net.new_cbam_net(kernel_size=3, padding=1)
    net = [net1, net2, net3]
    tag = ['resnet18', 'densenet121', 'my_net']

    writer = SummaryWriter('run_log/experiment1')
    for i in range(len(net)):
        utf.train(net[i], train_iter, test_iter, 50, 0.00001, writer, tag[i], utf.try_gpu(0))


def experiment2():
    """
    实验2：
    input：融合后的数据，FDA+FDT+FAT<-->FDA+FDT
    network：浅层卷积网络<-->经典网络<-->自己设计的网络
    """

    # 数据集加载
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\train_data'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\test_data'
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = ImageFolder(train_root, transform=transform)
    test_data = ImageFolder(test_root, transform=transform)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)

    # 网络加载
    net1 = models.resnet18(pretrained=False, num_classes=2)
    net2 = models.densenet121(pretrained=False, num_classes=2)
    net3 = my_net.new_cbam_net(kernel_size=3, padding=1)
    net = [net1, net2, net3]
    tag = ['resnet18', 'densenet121', 'my_net']

    writer = SummaryWriter('run_log/experiment2')
    for i in range(len(tag)):
        utf.train(net[i], train_iter, test_iter, 50, 0.00005, writer, tag[i], utf.try_gpu(0))


def experiment3():
    """
    实验3：
    input：融合后的数据，FDA+FDT+FAT
    network：经典网络<-->自己设计的网络
    增强技术：对比增强
    """

    # 数据集加载
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\train_data'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\test_data'
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = ImageFolder(train_root, transform=transform)
    test_data = ImageFolder(test_root, transform=transform)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)

    # 网络加载
    net1 = torch.load('resnet18-retrained.pth')
    net2 = torch.load('densenet121-retrained.pth')
    net3 = torch.load('my_net-retrained.pth')
    net = [net1, net2, net3]
    tag = ['resnet18', 'densenet121', 'my_net']

    writer = SummaryWriter('run_log/experiment3')
    for i in range(len(tag)):
        utf.train(net[i], train_iter, test_iter, 50, 0.00001, writer, tag[i], utf.try_gpu(0))
        torch.save(net[i], f'{tag[i]}-retrained.pth')


def experiment4():
    """
    实验4：
    input：融合后的数据，FDA+FDT+FAT
    network：浅层卷积网络<-->经典网络<-->自己设计的网络
    增强技术：使用投票机制进行判断
    """

    # 数据集加载
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\test_data'
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    test_data = ImageFolder(test_root, transform=transform)
    test_iter = data.DataLoader(test_data, len(test_data))

    # 网络加载
    net1 = torch.load('resnet18-retrained.pth')
    net2 = torch.load('densenet121-retrained.pth')
    net3 = torch.load('my_net-retrained.pth')
    net = [net1, net2, net3]

    # utf.test_model(net1, test_iter)
    utf.test_based_on_vote(net, test_iter)


def experiment5():
    """
    实验4：
    input：融合后的数据，FDA+FDT+FAT
    network：浅层卷积网络<-->经典网络<-->自己设计的网络
    增强技术：使用投票机制进行判断
    """

    # 数据集加载
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\train_data'
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = ImageFolder(train_root, transform=transform)
    train_iter = data.DataLoader(train_data)

    # 网络加载
    net1 = torch.load('resnet18-retrained.pth')
    net2 = torch.load('densenet121-retrained.pth')
    net3 = torch.load('my_net-retrained.pth')
    net = [net1, net2, net3]

    # utf.test_model(net1, train_iter)
    utf.test_based_on_vote(net, train_iter)


def experiment6():
    """
    使用目标检测网络训练
    """
    # 数据集
    train_root = r'dataset/Dataset220324/train/images'

    # 参数
    anchors = [[[60, 60]], [[40, 40]], [[20, 20]]]
    lr, num_epochs = 0.001, 1000
    batch_size = 10
    l_box = 3  # 框损失的权重系数
    l_obj = 100  # 目标损失的权重系数

    # 数据加载
    device = utf.try_gpu(0)
    # device = torch.device('cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.LoadImagesAndLabels(train_root, transform=transform)
    train_iter = data.DataLoader(train_data, batch_size)
    # test_iter = data.DataLoader(test_data, batch_size)

    # 网络
    # net = my_net.object_detect_new_cbam_net(anchors=anchors, kernel_size=3, padding=1, groups=1)
    net = ObjectDetect0412.ODAB()
    net.to(device)

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9994)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.92)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=lr, mode='exp_range', step_size_up=400, gamma=0.999)

    # 记录run log
    writer = SummaryWriter('run_log/experiment6')
    writer_tag = "05-12-1"
    min_loss = 1

    for epoch in range(num_epochs):
        net.train()
        mean_loss = 0
        for i, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X = x.to(device)
            result = net(X)
            target = utf.get_targets(y)
            loss = utf.compute_loss(result, anchors=anchors, targets=target, l_box=l_box, l_obj=l_obj)
            sum_loss = loss["box_loss"] + loss["obj_loss"]
            sum_loss.backward()
            optimizer.step()
            # scheduler.step()

            temp1 = loss['box_loss'].data
            temp2 = loss['obj_loss'].data
            temp3 = sum_loss.data
            mean_loss += temp3
            writer.add_scalar(f"{writer_tag}/loss/box_loss", temp1, global_step=epoch * len(train_iter) + i)
            writer.add_scalar(f"{writer_tag}/loss/obj_loss", temp2, global_step=epoch * len(train_iter) + i)
            writer.add_scalar(f"{writer_tag}/loss/sum_loss", temp3, global_step=epoch * len(train_iter) + i)
            writer.add_scalar(f"{writer_tag}/lr", scheduler.get_last_lr()[0], global_step=epoch * len(train_iter) + i)
            if i % len(train_iter) == 0:
                print(f"epoch:{epoch}",
                      f"  box loss:{temp1.cpu().numpy()}",
                      f"  obj loss:{temp2.cpu().numpy()}",
                      f"  sum loss:{temp3.cpu().numpy()}")
        if mean_loss/((epoch+1)*len(train_iter)) < min_loss:
            torch.save(net.state_dict(), f"./models/{writer_tag}-detect.pth")
            min_loss = mean_loss/((epoch+1)*len(train_iter))
        scheduler.step()
    # torch.save(net, f"./models/{writer_tag}-detect.pth")


if __name__ == '__main__':
    experiment6()
