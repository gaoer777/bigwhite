from torch import nn
from torch.nn import functional as F


'''#load excel dataset class
class datasets(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        flist = os.listdir(path)
        first_class_path = path + os.sep + flist[0]
        first_class_data = os.listdir(first_class_path)
        for i, file in enumerate(first_class_data):
            first_class_data[i] = os.path.join(first_class_path, file)
        first_class_num = len(first_class_data)
        second_class_path = path + os.sep + flist[1]
        second_class_data = os.listdir(second_class_path)
        for i, file in enumerate(second_class_data):
            second_class_data[i] = os.path.join(second_class_path, file)
        second_class_num = len(second_class_data)
        data_num = first_class_num + second_class_num
        self.label = np.zeros(data_num)
        self.label[:first_class_num] = 1
        self.data_list = first_class_data + second_class_data
        self.transform = transform

    def __getitem__(self, indx):
        item = self.data_list[indx]
        img = np.array(pd.read_table(item, header=None))
        img = img[:, :-1]
        img = torch.from_numpy(img)
        if img.shape[1]==63:
            zero_row = torch.zeros(1, 63)
            zero_col = torch.zeros(64, 1)
            img = torch.cat((zero_row, img), 0)
            img = torch.cat((zero_col, img), 1)
        elif img.shape[1]==62:
            zero_row = torch.zeros(1, 62)
            zero_col = torch.zeros(64, 2)
            img = torch.cat((zero_row, img), 0)
            img = torch.cat((zero_col, img), 1)
        img = torch.unsqueeze(img, dim=0)
        img = torch.FloatTensor(img)
        if self.transform is not None:
            img = self.transform(img)
        y = self.label[indx]
        return img, y

    def __len__(self):
        return len(self.label)

transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.Grayscale(1),
                                transforms.ToTensor()])
train_data = datasets(train_root)
test_data = datasets(test_root)
batch_size = 32
train_iter = data.DataLoader(train_data, batch_size, shuffle=True, sampler=None)
test_iter = data.DataLoader(test_data, batch_size)
'''


class Residual(nn.Module):  # 定义残差快
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
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
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


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


def resnet18():
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


def resnet34():
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
