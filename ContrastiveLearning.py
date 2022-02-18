import numpy as np
import os
import cv2
from torch.autograd import Variable
import torch
from torch.utils import data
import torch.nn as nn
import my_net


class ContrastiveNetwork(nn.Module):
    """
    使用对比学习训练的方式提升网络效果
    """
    def __init__(self):
        super(ContrastiveNetwork, self).__init__()
        self.cnn0 = my_net.train_siamese_net()
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


def create_pairs(data, digit_indices):
    """
    将数据成对输出，输出的标签根据他们是否是同一类型
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
            label.append(0)  # 相同样本对标签置为0
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]  # 匹配负样本（不同标签）
            x0_data.append(data[z1] / 255.)
            x1_data.append(data[z2] / 255.)
            label.append(1)  # 不相同样本对的标签置为1

    x0_data = np.array(x0_data, dtype=np.float32)  # 转换数据类型
    x0_data = x0_data.reshape([-1, 3, 64, 64])  # (h,w,c)-->(c,h,w)
    x1_data = np.array(x1_data, dtype=np.float32)
    x1_data = x1_data.reshape([-1, 3, 64, 64])
    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label


def create_iterator(data, label, batchsize, shuffle=False):
    # print ("max label", max(label))
    digit_indices = [np.where(label == i)[0] for i in range(max(label) + 1)]  # 创建一个包含两个np数组的列表，每个列表包含同一标签图像的索引。
    x0, x1, label = create_pairs(data, digit_indices)
    ret = Dataset(x0, x1, label)
    return ret


def contrastive_loss_function(x0, x1, y, margin=1.0):
    # euclidean distance 使用欧式距离来作为损失函数
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


def main():
    # 加载数据集
    X_train, y_train = extract_features("dataset/Dataset211118/dataset/dataset211118_4/train_data")
    X_test, y_test = extract_features("dataset/Dataset211118/dataset/dataset211118_4/test_data")
    print(X_train.shape, y_train.shape)
    batchsize = 64
    train_iter = create_iterator(X_train, y_train, batchsize)
    train_loader = torch.utils.data.DataLoader(train_iter, batch_size=batchsize, shuffle=True)

    # 调用模型
    model = ContrastiveNetwork()
    model.to(torch.device(0))

    # 设置优化器
    learning_rate = 0.0001  # learning rate for optimization
    momentum = 0.9  # momentum
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # 设置损失函数
    criterion = contrastive_loss_function
    train_loss = []
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x0, x1, labels) in enumerate(train_loader):
            labels = labels.float().to(torch.device(0))
            x0, x1, labels = Variable(x0).to(torch.device(0)), Variable(x1).to(torch.device(0)), Variable(labels).to(
                torch.device(0))
            output1, output2 = model.forward(x0, x1)
            loss = criterion(output1, output2, labels)
            train_loss.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(epoch)
        print('Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss * 1.0 / batchsize))
    # if epoch % 10 == 0:
    torch.save(model, './SiameseModified-epoch-%s.pth' % epochs)


if __name__ == '__main__':
    main()
