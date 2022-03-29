import numpy as np
import os
import random
import cv2
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import UtilFunctions as utf
import torch
from torch.utils import data
import torch.nn as nn
import my_net


class ContrastiveNetwork(nn.Module):
    """
    使用对比学习网络，输入样本对，输出每个样本的特征提向量
    """
    def __init__(self, model):
        super(ContrastiveNetwork, self).__init__()
        self.cnn0 = model
        self.enginlayer = nn.Linear(504, 504, bias=False)
        self.sigmoid = torch.sigmoid

    def forward_once(self, x):
        output = self.cnn0(x)
        output = self.enginlayer(self.sigmoid(output))
        return self.sigmoid(output)

    def forward(self, input0, input1):
        output0 = self.forward_once(input0)
        output1 = self.forward_once(input1)
        return output0, output1


class ContrastiveNetwork2(nn.Module):
    """
    使用对比学习网络2，输入样本对，输出他们是否是同类样本
    """
    def __init__(self, model):
        super(ContrastiveNetwork2, self).__init__()
        self.cnn0 = model
        self.enginlayer = nn.Linear(504, 504)
        self.sigmoid = torch.sigmoid
        self.Linear = nn.Linear(504*2, 2048)
        self.Linear1 = nn.Linear(2048, 1024)
        self.Linear2 = nn.Linear(1024, 2)

    def forward_once(self, x):
        output = self.cnn0(x)
        output = self.enginlayer(self.sigmoid(output))
        return self.sigmoid(output)

    def forward(self, input0, input1):
        output0 = self.forward_once(input0)
        output1 = self.forward_once(input1)
        out = torch.cat((output0, output1), dim=1)
        out = self.Linear(out)
        out = self.Linear1(self.sigmoid(out))
        out = self.Linear2(self.sigmoid(out))
        return self.sigmoid(out)


class Dataset(object):
    """
    Class Dataset:
    Input: numpy values
    Output: torch variables.
    """
    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return (self.x0[index],
                self.x1[index],
                self.label[index])

    def __len__(self):
        return self.size


class ContrasNetClassify(nn.Module):
    """给特征提取网络后面添加一个线性层用来分类，线性层输入为特征维数，输出为分类数"""
    def __init__(self, model_path):
        super(ContrasNetClassify, self).__init__()
        self.backbone = torch.load(model_path)
        self.classifier = nn.Linear(504, 2)

    def forward(self, x):
        out = self.backbone.forward_once(x)
        out = self.classifier(out)
        return torch.sigmoid(out)


def create_pairs(data, digit_indices):
    """
    将数据成对输出，输出的标签根据他们是否是同一类型
    只给正样本匹配一个正样本和一个负样本
    输出总数据集=2*正样本数
    """
    x0_data = []
    x1_data = []
    label = []
    n = min([len(digit_indices[d]) for d in range(2)]) - 1  # 样本对数取决于数量最小的类
    for d in range(2):  # 让每个类别和其他类进行配对
        dn = (d + 1) % 2  # 负样本的类别
        for i in range(n):  # 给类的每个图象都匹配一个负样本和一个正样本
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]  # 匹配正样本（相同标签）
            x0_data.append(data[z1] / 255.)  # 图像归一化
            x1_data.append(data[z2] / 255.)
            label.append(1)  # 相同样本对标签置为1
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]  # 匹配负样本（不同标签）
            x0_data.append(data[z1] / 255.)
            x1_data.append(data[z2] / 255.)
            label.append(0)  # 不相同样本对的标签置为0

    x0_data = np.array(x0_data, dtype=np.float32)  # 转换数据类型
    x0_data = x0_data.reshape([-1, 3, 64, 64])  # (h,w,c)-->(c,h,w)
    x1_data = np.array(x1_data, dtype=np.float32)
    x1_data = x1_data.reshape([-1, 3, 64, 64])
    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label


def create_pairs2(data, digit_indices, m, n, t):
    """
    将数据成对输出，输出的标签根据他们是否是同一类型
    给正样本随机匹配m个正样本，给负样本随机匹配n个正样本，给负样本匹配k个负样本
    输出样本总数 = m*正样本数 + n*负样本数 + t*负样本数
    """
    # digit_indices[0]：缺陷图片路径列表、digit_indices[1]：非缺陷图片路径列表
    x0_data = []
    x1_data = []
    label = []
    n0 = len(digit_indices[0])
    n1 = len(digit_indices[1])
    random.shuffle(digit_indices[0])
    random.shuffle(digit_indices[1])
    for i in range(n0):
        randnum = int((n0 - m - 1) * random.random())
        for k in range(m):  # 给每个正样本匹配m个正样本，标签为1
            z = [digit_indices[0][i], digit_indices[0][randnum+k]]
            random.shuffle(z)
            x0_data.append(data[z[0]] / 255.)  # 图像归一化
            x1_data.append(data[z[1]] / 255.)
            label.append(1)
    for i in range(n1):
        randnum = int((n0 - n - 1) * random.random())
        for k in range(n):  # 给每个负样本匹配n个正样本，标签为0
            z = [digit_indices[1][i], digit_indices[0][randnum+k]]
            random.shuffle(z)
            x1_data.append(data[z[0]] / 255.)  # 图像归一化
            x0_data.append(data[z[1]] / 255.)
            label.append(0)
        randnum1 = int((n1 - t - 1) * random.random())
        for j in range(t):  # 给每个负样本匹配t个负样本，标签为1
            z = [digit_indices[1][i], digit_indices[1][randnum1+j]]
            random.shuffle(z)
            x0_data.append(data[z[0]] / 255.)  # 图像归一化
            x1_data.append(data[z[1]] / 255.)
            label.append(1)

    x0_data = np.array(x0_data, dtype=np.float32)  # 转换数据类型
    x0_data = x0_data.reshape([-1, 3, 64, 64])  # (h,w,c)-->(c,h,w)
    x1_data = np.array(x1_data, dtype=np.float32)
    x1_data = x1_data.reshape([-1, 3, 64, 64])
    label = np.array(label, dtype=np.int64)
    return x0_data, x1_data, label


def evaluations(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的混淆矩阵，返回tp, fp, tn, fn, y.numel()。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = utf.Accumulator(5)
    for i, (X0, X1, y) in enumerate(data_iter):
        X0, X1 = X0.to(device), X1.to(device)
        y = y.to(device)
        y_hat = net(X0, X1)
        y_temp = y_hat.argmax(axis=1)
        tp = y_temp[(y_temp == 0) & (y == 0)]
        fp = y_temp[(y_temp == 0) & (y == 1)]
        tn = y_temp[(y_temp == 1) & (y == 1)]
        fn = y_temp[(y_temp == 1) & (y == 0)]
        metric.add(tp.numel(), fp.numel(), tn.numel(), fn.numel(), y.numel())
    return metric


def create_iterator(data, label):
    # print ("max label", max(label))
    # 创建一个包含两个np数组的列表，每个列表包含同一标签图像的索引。
    digit_indices = [np.where(label == i)[0] for i in range(max(label) + 1)]
    x0, x1, label = create_pairs2(data, digit_indices, 5, 2, 1)
    ret = Dataset(x0, x1, label)
    return ret


def contrastive_loss_function(x0, x1, y, margin=1.0):
    """euclidean distance 使用欧式距离来作为损失函数， 用来衡量两个向量的相似性"""
    diff = x0 - x1
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)
    mdist = margin - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    loss = torch.sum(loss) / 2.0 / x0.size()[0]
    return loss


def extract_features(path):
    """输入数据集的路径，返回numpy形式的图像数据和其对应的标签"""
    directory_lists = os.listdir(path)
    X = []
    Y = []
    count = 0
    if '.DS_Store' in directory_lists:
        directory_lists.remove('.DS_Store')
    for d in directory_lists:
        nest = os.listdir(path + "/" + d)
        if '.DS_Store' in nest:
            nest.remove('.DS_Store')
        for f in nest:
            img = cv2.imread(path + "/" + d + "/" + f)
            img = cv2.resize(img, (64, 64))
            X.append(img)
            Y.append(count)
        count += 1
    X = np.array(X)
    y = np.array(Y)
    return X, y


def train_step1(net, train_it, test_it, epochs, lr, device):
    """只训练分类器部分"""
    for k in net.backbone.parameters():
        k.requires_grad = False

    print('step1: 只训练分类器部分， training on', device)
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


def train_step2(net, train_it, test_it, writer, tag, epochs, lr, device):
    """拉通训练整个网络"""
    # 训练所有数据
    for param in net.parameters():
        param.requires_grad = True

    print('step2: 训练所有数据， training on', device)
    net.to(device)  # 网络移植到GPU上
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 设置优化器
    loss = nn.CrossEntropyLoss()  # 设置loss function

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
                writer.add_scalar(f'{tag}/train-loss', train_l, epoch + (i + 1)/num_batches)
                writer.add_scalar(f'{tag}/train-acc', train_acc, epoch + (i + 1) / num_batches)

        indexes = utf.evaluations(net, test_it)
        test_acc = (indexes[0] + indexes[2]) / indexes[4]  # 每个epoch测试一次测试集的准确率
        FPR = indexes[0] / (indexes[0] + indexes[3])
        writer.add_scalar(f'{tag}/test_acc', test_acc, global_step=epoch)
        writer.add_scalar(f'{tag}/FPR', FPR, global_step=epoch)
        print(f'loss {train_l:.3f}, train acc {train_acc*100:.4f}%, '
              f'test acc {test_acc*100:.4f}%,'
              f'FPR {FPR*100:.4f}%,'
              f'proccessed {epoch * 100 / epochs:.2f}%')
    #  保存模型
    torch.save(net, f'{tag}-retrained.pth')


def retrain_main():
    """训练通用分类网络的主程序"""
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

    model_path1 = 'resnet18-epoch-30.pth'
    model_path2 = 'densenet121-epoch-30.pth'
    model_path3 = 'my_net-epoch-30.pth'
    model_path = [model_path1, model_path2, model_path3]
    tag = ['resnet18', 'densenet121', 'my_net']
    writer = SummaryWriter('./run_log/retrain')
    lr = 0.00005

    for i in range(3):
        model = ContrasNetClassify(model_path[i])
        train_step2(model, train_iter, test_iter, writer, tag[i], 25, lr, utf.try_gpu(0))


def train_contrast_model(model, train_loader, writer, tag, batch_size=64,
                         device='cpu', learning_rate=0.0001, epochs=30):
    """训练对比模型并保存，模型的输出为504维的特征提取网络"""
    # 加载模型
    model = ContrastiveNetwork(model)
    model.to(device)

    # 设置优化器
    learning_rate = learning_rate  # learning rate for optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # momentum = 0.9  # momentum
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # 设置损失函数
    # criterion = contrastive_loss_function
    criterion = nn.CrossEntropyLoss
    train_loss = []
    epochs = epochs
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x0, x1, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.float().to(torch.device(0))
            x0, x1, labels = Variable(x0).to(torch.device(0)), \
                             Variable(x1).to(torch.device(0)), Variable(labels).to(
                torch.device(0))
            output1, output2 = model.forward(x0, x1)
            loss = criterion(output1, output2, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        # scheduler.step(epoch)
        print('Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss * 1.0 / batch_size))
        writer.add_scalar(f'{tag}/train_loss', total_loss, epoch)

    torch.save(model, f'./{tag}-epoch-%s.pth' % epochs)


def train_contra_net_main():
    """训练对比网络的特征提取能力，输出为504为特征向量"""
    # 加载数据集
    X_train, y_train = extract_features("dataset/Dataset211118/dataset/dataset211118_4/train_data")
    batch_size = 64
    train_data = create_iterator(X_train, y_train)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    net1 = models.resnet18(pretrained=False, num_classes=504)
    net2 = models.densenet121(pretrained=False, num_classes=504)
    net3 = my_net.train_siamese_net(kernel_size=3, padding=1)
    net = [net1, net2, net3]
    tag = ['resnet18', 'densenet121', 'my_net']
    writer = SummaryWriter('./run_log/train_contrast_model')

    for i in range(3):
        train_contrast_model(net[i], train_loader, writer, tag[i],
                             device=utf.try_gpu(0), learning_rate=0.0001)


def train_contrast_model2(model, train_loader, test_loader, writer, tag, batch_size=64,
                          device='cpu', learning_rate=0.0001, epochs=30):
    """训练对比模型并保存，模型的输出为2维的分类网络"""
    # 加载模型
    model.to(device)

    # 设置优化器
    learning_rate = learning_rate  # learning rate for optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # momentum = 0.9  # momentum
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # 设置损失函数
    # criterion = contrastive_loss_function
    criterion = nn.CrossEntropyLoss()
    epochs = epochs
    train_l, train_acc, num_batches = 0, 0, len(train_loader)

    for epoch in range(epochs):
        metric = utf.Accumulator(3)
        model.train()
        for i, (x0, x1, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
            output = model(x0, x1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * labels.shape[0], utf.accuracy(output, labels), labels.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % ((num_batches // 5)+1) == 0 or i == num_batches - 1:
                writer.add_scalar(f'{tag}/train_loss', train_l, global_step=epoch + (i + 1) / num_batches)
                writer.add_scalar(f'{tag}/train_acc', train_acc, global_step=epoch + (i + 1) / num_batches)
        indexes = evaluations(model, test_loader)
        test_acc = (indexes[0] + indexes[2]) / indexes[4]
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}, '
              f'processed {epoch * 100 / epochs:.2f}%')

    torch.save(model, f'./{tag}-vote-contra-epoch%s.pth' % epochs)


def train_contra_net_main2():
    """训练对比网络的特征提取能力，输出为对比相似度，1*2的向量，类似于2分类"""
    # 加载数据集
    X_train, y_train = extract_features(r"dataset\Dataset220217\dataset_220217_rgb\train_data")
    batch_size = 16
    train_data = create_iterator(X_train, y_train)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    X_test, y_test = extract_features(r"dataset\Dataset220217\dataset_220217_rgb\test_data")
    test_data = create_iterator(X_test, y_test)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    net1 = ContrastiveNetwork2(models.resnet34(pretrained=False, num_classes=504))
    net2 = ContrastiveNetwork2(models.densenet121(pretrained=False, num_classes=504))
    net3 = ContrastiveNetwork2(my_net.train_siamese_net(kernel_size=3, padding=1))
    net = [net1, net2, net3]
    tag = ['resnet18', 'densenet121', 'my_net']
    writer = SummaryWriter('./run_log/train_contra_net2')

    for i in range(1, 3):
        train_contrast_model2(net[i], train_loader, test_loader, writer, tag[i], epochs=50,
                              device=utf.try_gpu(0), learning_rate=0.00005)


def test_contra_main2_based_on_vote():
    """测试之前训练的网络（train_contra_net_main2）"""

    X_test, y_test = extract_features(r"dataset\Dataset220217\dataset_220217_rgb\test_data")
    # 随机挑选5张缺陷图像和5张非缺陷图像，作为投票对比，哪个得分高就判为哪一类。
    contra_num = 10  # 挑选的图像的数目
    digit_indices = [np.where(y_test == i)[0] for i in range(max(y_test) + 1)]
    random.shuffle(digit_indices[0])
    random.shuffle(digit_indices[1])
    eval = [0, 0, 0, 0]  # tp, fn, fp, tn
    # net = torch.load('my_net-vote-contra-epoch50.pth')
    net = torch.load('densenet121-vote-contra-epoch50.pth')
    net = net.cuda()
    for i in range(contra_num, len(digit_indices[0])):  # 测试缺陷图像
        x0_data = []
        x1_data = []
        for k in range(contra_num):  # 选5张非缺陷图像配对
            z1, z2 = digit_indices[0][i], digit_indices[1][k]
            x0_data.append(X_test[z1] / 255.)
            x1_data.append(X_test[z2] / 255.)
        for j in range(contra_num):  # 选5张缺陷图像匹配
            z1, z2 = digit_indices[0][i], digit_indices[0][j]
            x0_data.append(X_test[z2] / 255.)
            x1_data.append(X_test[z1] / 255.)

        x0_data = np.array(x0_data, dtype=np.float32)  # 转换数据类型
        x0_data = torch.from_numpy(x0_data.reshape([-1, 3, 64, 64])).cuda()  # (h,w,c)-->(c,h,w)
        x1_data = np.array(x1_data, dtype=np.float32)
        x1_data = torch.from_numpy(x1_data.reshape([-1, 3, 64, 64])).cuda()
        out = net(x0_data, x1_data)
        out = out.argmax(axis=1)
        out = out.reshape(2, -1)
        out = out.sum(axis=1)
        out = out.argmax(axis=0)
        if out == 0:
            eval[0] += 1
        else:
            eval[1] += 1

    for i in range(contra_num, len(digit_indices[0])):  # 测试非缺陷图像
        x0_data = []
        x1_data = []
        for k in range(contra_num):  # 选5张缺陷图像配对
            z1, z2 = digit_indices[1][i], digit_indices[0][k]
            x0_data.append(X_test[z2] / 255.)
            x1_data.append(X_test[z1] / 255.)
        for j in range(contra_num):  # 选5张非缺陷图像配对
            z1, z2 = digit_indices[1][i], digit_indices[1][j]
            x0_data.append(X_test[z1] / 255.)
            x1_data.append(X_test[z2] / 255.)

        x0_data = np.array(x0_data, dtype=np.float32)  # 转换数据类型
        x0_data = torch.from_numpy(x0_data.reshape([-1, 3, 64, 64])).cuda()  # (h,w,c)-->(c,h,w)
        x1_data = np.array(x1_data, dtype=np.float32)
        x1_data = torch.from_numpy(x1_data.reshape([-1, 3, 64, 64])).cuda()
        out = net(x0_data, x1_data)
        out = out.argmax(axis=1)
        out = out.reshape(2, -1)
        out = out.sum(axis=1)
        out = out.argmax(axis=0)
        if out == 1:
            eval[3] += 1
        else:
            eval[2] += 1
    acc = (eval[0] + eval[3] + 0.)*100 / sum(eval)
    fpr = (eval[1] + 0.)*100 / (eval[0] + eval[1])

    print(f'acc: {acc:.2f}%, '
          f'fpr: {fpr:.2f}%')


if __name__ == '__main__':
    # train_contra_net_main2()
    test_contra_main2_based_on_vote()

