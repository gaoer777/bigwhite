import my_net
import UtilFunctions as utf
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
    # net1 = models.resnet18(pretrained=False, num_classes=2)
    # net2 = models.densenet121(pretrained=False, num_classes=2)
    net3 = my_net.new_cbam_net()

    writer = SummaryWriter('run_log/experiment1')
    utf.train(net3, train_iter, test_iter, 50, 0.00001, writer, 'my_net', utf.try_gpu(0))


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
    # net1 = models.resnet18(pretrained=False, num_classes=2)
    net2 = models.densenet121(pretrained=False, num_classes=2)
    # net3 = my_net.new_cbam_net()

    writer = SummaryWriter('run_log/experiment2')
    utf.train(net2, train_iter, test_iter, 50, 0.00005, writer, 'densenet121', utf.try_gpu(0))


def experiment3():
    """
    实验2：
    input：融合后的数据，FDA+FDT+FAT
    network：浅层卷积网络<-->经典网络<-->自己设计的网络
    增强技术：未增强<-->对比增强
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
    # net1 = models.resnet18(pretrained=False, num_classes=2)
    net2 = models.densenet121(pretrained=False, num_classes=2)
    # net3 = my_net.new_cbam_net()

    writer = SummaryWriter('run_log/experiment3')
    utf.train(net2, train_iter, test_iter, 50, 0.00001, writer, 'densenet121', utf.try_gpu(0))


if __name__ == '__main__':
    experiment2()