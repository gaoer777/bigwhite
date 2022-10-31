import datetime
import os
import random
import shutil
import cv2
import torch
import my_net
import models_c
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
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\old_fda'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\new_fda'

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = ImageFolder(train_root, transform=transform)
    test_data = ImageFolder(test_root, transform=transform)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)
    weight = torch.FloatTensor([1, 1])  # 分类的损失函数权重参数

    # 网络加载
    net1 = models.resnet18(pretrained=False, num_classes=2)
    net2 = models.densenet121(pretrained=False, num_classes=2)
    net3 = my_net.new_cbam_net(kernel_size=3, padding=1)
    net = [net1, net2, net3]
    tag_net = ['resnet18', 'densenet121', 'my_net']
    tag_model_save = 'experiment1'
    now = datetime.datetime.now()
    timer = now.strftime("%Y-%m-%d")

    writer = SummaryWriter(f'run_log/experiment1/{timer}')
    for i in range(len(net)):
        utf.train(net[i], train_iter, test_iter, 100, 0.001, weight, writer, tag_net[i], tag_model_save, utf.try_gpu(0))


def experiment1_1():
    """
    实验1_1：使用1通道，2通道，3通道的图像进行实验。使用数据增强，使用的是新的数据Dataset220927
    实验1-2：实验内容同实验1-1。数据集使用Dataset220217下的数据，并使用了数据增强
    实验1-3：实验内容同实验1-2。不使用数据增强，使用一个或两个卷积核得到通道权重
    """

    # 数据集加载
    # train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\old_fdat'
    # test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\new_fdat'
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_fda\train_data'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_fda\test_data'
    # train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220913_fdat\train_data'
    # test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220913_fdat\test_data'
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomRotation(180, expand=1),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Resize((64, 64))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((64, 64))])
    train_data = ImageFolder(train_root, transform=transform_train)
    test_data = ImageFolder(test_root, transform=transform_test)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)
    device = utf.try_gpu(0)
    weight = torch.FloatTensor([1, 1]).to(device)  # 分类的损失函数权重参数

    # 网络加载
    net1 = models_c.RAM_Net()
    net2 = models.resnet18(pretrained=False, num_classes=2)
    net3 = models.densenet121(pretrained=False, num_classes=2)
    net = [net1, net2, net3]

    tag_net = ['RAM_Net-1C', 'resnet18-1C', 'densenet121-1C']
    tag_model_save = 'experiment1-2'
    now = datetime.datetime.now()
    timer = now.strftime("%Y-%m-%d")

    writer = SummaryWriter(f'run_log/experiment1-2/{timer}')

    for i in range(3):
        utf.train(net[i], train_iter, test_iter, 100, 0.0005, weight, writer, tag_net[i], tag_model_save, device)


def experiment2():
    """
    实验2：消融实验(22-10-18)
        数据集： train_root = 'D:/gsw/Projects/WOLIU/bigwhite/dataset/Dataset220217/dataset_220217_rgb/train_data'
                test_root = 'D:/gsw/Projects/WOLIU/bigwhite/dataset/Dataset220217/dataset_220217_rgb/test_data'
        数据增强： 翻转
        网络：RAM-Net，RAM-Net with no TA，RAM-Net with no SA，RAM-Net use pooling，
             RAM-Net use 2/3 convolution kernel，ReseNet18，DenseNet121
    实验2-1：
        数据增强：翻转 + 旋转
        数据集：同实验2
        网络：RAM-Net，RAM-Net with no TA，RAM-Net with no SA， RAM-Net with no SA and TA，RAM-Net use pooling，
             RAM-Net use 2/3 convolution kernel，ReseNet18，DenseNet121，RAM-Net without shared-MLP in TA
    实验2-2：
        数据增强：翻转 + 旋转
        数据集：train_root = r'D:/gsw/Projects/WOLIU/bigwhite/dataset/Dataset220927/old_fdat'
               test_root = r'D:/gsw/Projects/WOLIU/bigwhite/dataset/Dataset220927/new_fdat'
        网络：RAM-Net，RAM-Net with no TA，RAM-Net with no SA， RAM-Net with no SA and TA，RAM-Net use pooling，
             RAM-Net use 2/3 convolution kernel，ReseNet18，DenseNet121，RAM-Net without shared-MLP in TA，
             VIT
    实验2-3：
        数据增强：翻转 + 旋转
        数据集：同2-2
        网络：RAM-Net No separate
    """

    # 数据集加载
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\old_rgb2'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\new_rgb2'
    # train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\train_data'
    # test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\test_data'
    # train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220913_fdat\train_data'
    # test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220913_fdat\test_data'
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomRotation(180, expand=1),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Resize((64, 64))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((64, 64))])
    train_data = ImageFolder(train_root, transform=transform_train)
    test_data = ImageFolder(test_root, transform=transform_test)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)
    device = utf.try_gpu(0)
    weight = torch.FloatTensor([1, 1]).to(device)  # 分类的损失函数权重参数

    # 网络加载
    net1 = models_c.RAM_Net()  # 卷积权重 + TA + SA + 1kernel
    net2 = my_net.RAM_Net_P(kernel_size=(3, 3), padding=(1, 1))  # 池化权重 + TA + SA + 0kernel
    net3 = models_c.RAM_Net_Without_TA()  # 卷积权重 + SA + 1kernel
    net4 = models_c.RAM_Net_Without_SA()  # 卷积权重 + TA + 1kernel
    net5 = models_c.RAM_Net_Without_SA_TA()  # 卷积权重 + 1kernel
    net6 = models_c.RAM_Net_2kernels()  # 卷积权重 + TA + SA + 2kernels
    net7 = models_c.RAM_Net_3kernels()  # 卷积权重 + TA + SA + 3kernels
    net8 = models.resnet18(pretrained=False, num_classes=2)
    net9 = models.densenet121(pretrained=False, num_classes=2)
    net10 = models_c.RAM_Net_without_shared_mlp()  # 卷积权重 + (TA-shared-mlp) + SA + 1kernel
    net11 = models.efficientnet_b0(pretrained=False, num_classes=2)
    net = [net1, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11]

    tag_net = ['RAM_Net', 'RAM_Net_P', 'RAM_Net_Without_TA', 'RAM_Net_Without_SA', 'RAM_Net_Without_SA_TA',
               'RAM_Net_2kernels', 'RAM_Net_3kernels', 'resnet18', 'densenet121', 'RAM_Net_without_shared_mlp',
               'efficient-net']
    tag_model_save = 'experiment2-2'
    now = datetime.datetime.now()
    timer = now.strftime("%Y-%m-%d")

    writer = SummaryWriter(f'run_log/experiment2-2/{timer}')

    for i in range(10, 11):
        utf.train(net[i], train_iter, test_iter, 120, 0.0005, weight, writer, tag_net[i], tag_model_save, device)


def eperiment2_3():
    """
    实验2-3：
    5倍交叉验证
    input：融合后的数据，FDA+FDT+FAT
    network：['resnet18', 'densenet121', 'my_net']
    """
    dataset = ['10.png', '100.png', '101.png', '103.png', '104.png', '107.png', '11.png', '112.png', '113.png',
               '118.png', '12.png', '120.png', '121.png', '122.png', '124.png', '125.png', '126.png', '127.png',
               '128.png', '13.png', '133.png', '134.png', '135.png', '136.png', '14.png', '144.png', '145.png',
               '146.png', '147.png', '148.png', '149.png', '15.png', '150.png', '151.png', '152.png', '16.png',
               '17.png', '18.png', '19.png', '20.png', '21.png', '22.png', '23.png', '25.png', '26.png', '27.png',
               '28.png', '29.png', '30.png', '31.png', '33.png', '35.png', '36.png', '37.png', '39.png', '40.png',
               '41.png', '42.png', '43.png', '44.png', '45.png', '47.png', '48.png', '49.png', '50.png', '51.png',
               '52.png', '53.png', '54.png', '55.png', '56.png', '57.png', '58.png', '59.png', '6.png', '60.png',
               '61.png', '64.png', '65.png', '66.png', '67.png', '68.png', '69.png', '7.png', '70.png', '71.png',
               '72.png', '73.png', '74.png', '75.png', '76.png', '77.png', '78.png', '79.png', '8.png', '80.png',
               '81.png', '82.png', '83.png', '84.png', '85.png', '88.png', '89.png', '9.png', '90.png', '91.png',
               '92.png', '94.png', '97.png']

    defectset = [r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220915_fda_for_5',
                 r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220915_fdat_for_5',
                 r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220915_rgb_for_5']
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220915_for_5\test_data'
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220915_for_5\train_data'
    im_classes = ['defects', 'undefects']

    e_tag = ['1C', '2C', '3C']

    t = 2  # 使用defectset哪个数据集进行试验
    random.shuffle(dataset)
    k = int(len(dataset) / 5)
    for i in range(5):  # 5倍交叉验证
        shutil.rmtree(test_root + '/' + im_classes[0])  # 清空文件夹
        shutil.rmtree(test_root + '/' + im_classes[1])
        shutil.rmtree(train_root + '/' + im_classes[0])
        shutil.rmtree(train_root + '/' + im_classes[1])
        os.mkdir(test_root + '/' + im_classes[0])  # 清空文件夹
        os.mkdir(test_root + '/' + im_classes[1])
        os.mkdir(train_root + '/' + im_classes[0])
        os.mkdir(train_root + '/' + im_classes[1])

        start_index = k * i
        end_index = k * i + k
        if i == 4:
            end_index = len(dataset)
        testset = dataset[start_index: end_index]  # 获取测试集图片 ##以下是将测试集和训练集分开
        for c in im_classes:
            de_names = os.listdir(defectset[t] + '/' + c)
            for de_name in de_names:
                or_name = de_name.split('_')[1] + '.png'
                old_file_name = defectset[t] + '/' + c + '/' + de_name
                if testset.count(or_name) > 0:
                    new_file_name = test_root + '/' + c + '/' + de_name
                else:
                    new_file_name = train_root + '/' + c + '/' + de_name
                shutil.copyfile(old_file_name, new_file_name)

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
        tag = [f'resnet18-{e_tag[t]}-{i}', f'densenet121-{e_tag[t]}-{i}', f'my_net-{e_tag[t]}-{i}']
        tag_f = ['resnet18', 'densenet121', 'my_net']

        for n in range(len(tag)):
            writer = SummaryWriter(f'run_log/experiment2-3/{tag_f[n]}/{e_tag[t]}')
            # utf.train(net[n], train_iter, test_iter, 50, 0.001, writer, tag[n], utf.try_gpu(0))


def experiment3():
    """
    实验3：
    input：融合后的数据，FDA+FDT+FAT
    network：经典网络<-->自己设计的网络
    """

    # 数据集加载
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\old_rgb2'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\new_rgb2'
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomRotation(180, expand=1, ),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Resize((64, 64))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((64, 64))])
    train_data = ImageFolder(train_root, transform=transform_train)
    test_data = ImageFolder(test_root, transform=transform_test)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)
    device = utf.try_gpu(0)
    weight = torch.FloatTensor([10, 1]).to(device)  # 分类的损失函数权重参数

    # 网络加载
    net1 = models.resnet18(pretrained=False, num_classes=2)
    net2 = models.densenet121(pretrained=False, num_classes=2)
    net3 = my_net.new_cbam_net(kernel_size=3, padding=1)
    net = [net1, net2, net3]
    tag_net = ['resnet18', 'densenet121', 'my_net']
    tag_model_save = 'experiment3'
    now = datetime.datetime.now()
    timer = now.strftime("%Y-%m-%d")

    writer = SummaryWriter(f'run_log/experiment3/{timer}')
    # for i in range(len(net)):
    i = 2
    utf.train(net[i], train_iter, test_iter, 150, 0.001, weight, writer, tag_net[i], tag_model_save, device)


def experiment3_1():
    """
    实验3-1：实验三的对比试验，减少了网络的卷积层数
    """

    # 数据集加载
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\old_rgb2'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\new_rgb2'
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomRotation(180, expand=1, ),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Resize((64, 64))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((64, 64))])
    train_data = ImageFolder(train_root, transform=transform_train)
    test_data = ImageFolder(test_root, transform=transform_test)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)
    device = utf.try_gpu(0)
    weight = torch.FloatTensor([10, 1]).to(device)  # 分类的损失函数权重参数

    # 网络加载
    net1 = my_net.new_cbam_net_ss2()
    net2 = my_net.new_cbam_net_ss2_no_group()
    net3 = my_net.new_cbam_net_s1(kernel_size=3, padding=1)
    net = [net1, net2, net3]
    tag_net = ['my_net_ss2', 'my_net_ss2_no_group', 'my_net_s1']
    tag_model_save = 'experiment3-1'
    now = datetime.datetime.now()
    timer = now.strftime("%Y-%m-%d")

    writer = SummaryWriter(f'run_log/experiment3-1/{timer}')
    for i in range(2):
        utf.train(net[i], train_iter, test_iter, 100, 0.0005, weight, writer, tag_net[i], tag_model_save, device)


def experiment3_2():
    """
    实验3_1：
    对比实验3的不同之处在于使用了自己写的Dataset
    """

    # 数据集加载
    train_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\old_rgb2'
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\new_rgb2'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(180, expand=1, fill=[0.5, 0.5, 0]),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize((64, 64))])
    train_data = ImageFolder(train_root, transform=transform)
    test_data = ImageFolder(test_root, transform=transform)
    train_iter = data.DataLoader(train_data, 32, shuffle=True)
    test_iter = data.DataLoader(test_data, 32)

    # 网络加载
    net1 = models.resnet18(pretrained=False, num_classes=2)
    net2 = models.densenet121(pretrained=False, num_classes=2)
    net3 = my_net.new_cbam_net(kernel_size=3, padding=1)
    net = [net1, net2, net3]
    tag_net = ['resnet18', 'densenet121', 'my_net']
    tag_model_save = 'experiment3'
    now = datetime.datetime.now()
    timer = now.strftime("%Y-%m-%d")

    writer = SummaryWriter(f'run_log/experiment3/{timer}')
    # for i in range(len(net)):
        # utf.train(net[i], train_iter, test_iter, 100, 0.001, writer, tag_net[i], tag_model_save, utf.try_gpu(0))


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
    net.load_state_dict(torch.load('./models/06-17-1-detect.pth'))

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9994)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.92)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=lr, mode='exp_range'
    #                                               , step_size_up=400, gamma=0.999)

    # 记录run log
    writer = SummaryWriter('run_log/experiment6')
    writer_tag = "06-20-1"
    min_loss = 100

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
        if mean_loss / ((epoch + 1) * len(train_iter)) < min_loss:
            torch.save(net.state_dict(), f"./models/{writer_tag}-detect.pth")
            min_loss = mean_loss / ((epoch + 1) * len(train_iter))
        scheduler.step()
    # torch.save(net, f"./models/{writer_tag}-detect.pth")


def experiment7():
    """
    使用COCO数据集与预训练目标检测网络
    """
    # 数据集
    train_root = r'D:/gsw/Projects/WOLIU/yolov5/dataset/train2017/images'
    train_test_root = r'D:\gsw\Projects\WOLIU\yolov5\dataset\coco128\images'
    train_test_save_root = './run_log/e7trainlog'

    # 参数
    anchors = [[[60, 60]], [[40, 40]], [[20, 20]]]
    lr, num_epochs = 0.001, 10
    batch_size = 32
    l_box = 3  # 框损失的权重系数
    l_obj = 10  # 目标损失的权重系数

    # 数据加载
    device = utf.try_gpu(0)
    # device = torch.device('cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.LoadCOCOImagesAndLabels(train_root, transform=transform)
    train_iter = data.DataLoader(train_data, batch_size)
    # test_iter = data.DataLoader(test_data, batch_size)

    # 网络
    # net = my_net.object_detect_new_cbam_net(anchors=anchors, kernel_size=3, padding=1, groups=1)
    net = ObjectDetect0412.ODAB()
    net.to(device)

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9994)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.998)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=lr, mode='exp_range'
    #                                               , step_size_up=400, gamma=0.999)

    # 记录run log
    writer = SummaryWriter('run_log/experiment7')
    writer_tag = "06-17-1"
    min_loss = 100

    for epoch in range(num_epochs):
        net.train()
        mean_loss = 0
        for i, (x, y, z) in enumerate(train_iter):
            optimizer.zero_grad()
            X = x.to(device)
            result = net(X)
            target = utf.get_targets_COCO(y, z)
            loss = utf.compute_loss(result, anchors=anchors, targets=target, l_box=l_box, l_obj=l_obj)
            sum_loss = loss["box_loss"] + loss["obj_loss"]
            sum_loss.backward()
            optimizer.step()
            scheduler.step()

            temp1 = loss['box_loss'].data
            temp2 = loss['obj_loss'].data
            temp3 = sum_loss.data
            mean_loss += temp3
            writer.add_scalar(f"{writer_tag}/loss/box_loss", temp1, global_step=epoch * len(train_iter) + i)
            writer.add_scalar(f"{writer_tag}/loss/obj_loss", temp2, global_step=epoch * len(train_iter) + i)
            writer.add_scalar(f"{writer_tag}/loss/sum_loss", temp3, global_step=epoch * len(train_iter) + i)
            writer.add_scalar(f"{writer_tag}/lr", scheduler.get_last_lr()[0], global_step=epoch * len(train_iter) + i)

            if i % 10 == 0:
                print(f"epoch:{epoch}",
                      f"  processed: {(i / len(train_iter) * 100):.2f}",
                      "  box loss:{%4f}" % temp1,
                      "  obj loss:{%4f}" % temp2,
                      "  sum loss:{%4f}" % temp3)

        # 每个epoch测试一下效果如何
        img, pre, re = utf.trainDetect(net, device, train_test_root, train_test_save_root, 'epoch' + str(epoch))
        # writer.add_image(str(epoch), np.array(img), global_step=epoch)
        writer.add_scalars(f"{writer_tag}/loss/box_loss", {'precesion': pre, 'recall': re}, epoch)

        # 保存模型
        if mean_loss / ((epoch + 1) * len(train_iter)) < min_loss:
            torch.save(net.state_dict(), f"./models/{writer_tag}-detect.pth")
            min_loss = mean_loss / ((epoch + 1) * len(train_iter))
        # scheduler.step()
    # torch.save(net, f"./models/{writer_tag}-detect.pth")


def experiment8():
    """
    输入原始图像，将原始图像进行卷积操作
    """
    input_im = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220927\new_rgb2\defects\RGB_1_0.png'
    im = cv2.imread(input_im)
    transformer = transforms.Compose([transforms.ToTensor()])
    im = transformer(im)


if __name__ == '__main__':
    # experiment1()
    experiment2()
    # experiment1_1()
    # main()
