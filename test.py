import os
import random
from torchvision import transforms
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from thop import profile
import models_c
import ObjectDetect0412
from torchvision import models
from tensorboard.backend.event_processing import event_accumulator
import my_net


# 测试模型的参数量
def printParamNum():
    # model = ObjectDetect0412.ODAB()
    net1 = models.resnet18(pretrained=False, num_classes=2)
    net2 = models.densenet121(pretrained=False, num_classes=2)
    net3 = my_net.new_cbam_net(kernel_size=3, padding=1)
    net4 = my_net.new_cbam_net_s1(kernel_size=3, padding=1)
    net5 = my_net.new_cbam_net_ss1(kernel_size=3, padding=1)
    net6 = my_net.new_cbam_net_s2(kernel_size=3, padding=1)
    net = [net1, net2, net3, net4, net5, net6]
    input = torch.randn(1, 3, 64, 64)

    for i in range(6):
        flops, params = profile(net[i], (input, ))
        print(f'flops: {flops},  params {params}')
        # total = sum([param.nelement() for param in net[i].parameters()])
        # print(total)
        # print(net[i])


def applyColor():
    convert_path = r'C:\Users\gsw\Desktop\imgs\dataset'
    save_path = r'C:\Users\gsw\Desktop\imgs\color_dataset'

    cdirs = os.listdir(convert_path)
    for p in cdirs:
        cpath = convert_path + '/' + p
        fpath = os.listdir(cpath)
        for f in fpath:
            imph = cpath + '/' + f
            imgs = os.listdir(imph)
            for img in imgs:
                im_path = imph + '/' + img
                im = cv2.imread(im_path, 0)
                im = cv2.applyColorMap(im, 3)
                save_im = save_path + '/' + p + '/' + f + '/' + img
                cv2.imwrite(save_im, im)


def makeDataset():
    """
    把FDA和FDT通道的数据融合起来制作FDA+FDT+0的数据集
    """
    fda_path = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\FDA'
    fdt_path = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\FDT'
    savepath = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220913_fdat\trian_data'

    imf = os.listdir(fda_path)
    for f in imf:
        impa = fda_path + '/' + f
        impt = fdt_path + '/' + f
        imgs = os.listdir(impa)
        for img in imgs:
            fda_im = cv2.resize(cv2.imread(impa + '/' + img, 0), (63, 63))
            fdt_im = cv2.resize(cv2.imread(impt + '/' + img, 0), (63, 63))
            zero_map = np.zeros((63, 63), np.uint8)
            im = cv2.merge([fda_im, fdt_im, zero_map])
            save_im = savepath + '/' + f + '/' + img
            cv2.imwrite(save_im, im)


def datasetClassify():
    d_path = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220913_fdat\trian_data'
    classes = os.listdir(d_path)
    for c in classes:
        c_path = d_path + '/' + c
        imgs = os.listdir(c_path)
        random.shuffle(imgs)
        for i in range(int(len(imgs)*0.2)):
            old_path = c_path + '/' + imgs[i]
            new_path = d_path.replace('trian_data', 'test_data') + '/' + c + '/' + imgs[i]
            os.rename(old_path, new_path)


def tensorboardDraw():
    """
    从tensorboard中拿出数据，然后使用np的拟合曲线，最后使用plt画出图像并保存
    针对的是不同的网络对于同一个数据去画图
    """
    ea = event_accumulator.EventAccumulator('./run_log/experiment2/events.out.tfevents.1662904538.BigBlack.31480.0').Reload()
    print(ea.scalars.Keys())
    my_loss_re = ea.scalars.Items('resnet18/train_loss')
    my_loss_de = ea.scalars.Items('densenet121/train_loss')
    my_loss_my = ea.scalars.Items('my_net/train_loss')

    re_list_x = [i.step*2 for i in my_loss_re]
    re_list_y = [i.value for i in my_loss_re]
    de_list_x = [i.step*2 for i in my_loss_de]
    de_list_y = [i.value for i in my_loss_de]
    my_list_x = [i.step*2 for i in my_loss_my]
    my_list_y = [i.value for i in my_loss_my]

    x = [re_list_x, de_list_x, my_list_x]
    y = [re_list_y, de_list_y, my_list_y]
    labels = ['Resnet18', 'Densenet121', 'Ours']
    color = ['b', 'g', 'r']
    line = []

    plt.xlabel('epoch')
    plt.ylabel('loss')
    for i in range(3):
        poly_re = np.polyfit(x[i], y[i], deg=20)
        y_value = np.polyval(poly_re, x[i])
        l = plt.plot(x[i], y_value, label=labels[i], color=color[i])
        plt.savefig(f'./run_log/{labels[i]}.png')
        plt.close()

    # plt.legend()
    # plt.savefig(f'./loss.png')


def tensorboardDrawCopy():
    """
    从tensorboard中拿出数据，然后使用np的拟合曲线，最后使用plt画出图像并保存
    针对的是同一个网络，不同的数据去画的图像
    """
    # ea1 = event_accumulator.EventAccumulator('./run_log/experiment1/events.out.tfevents.1662988573.BigBlack.35172.0').Reload()
    # ea2 = event_accumulator.EventAccumulator('./run_log/experiment1-1/events.out.tfevents.1663059141.BigBlack.42096.0').Reload()
    # ea3 = event_accumulator.EventAccumulator('./run_log/experiment2/events.out.tfevents.1663075854.BigBlack.39956.0').Reload()
    ea1 = event_accumulator.EventAccumulator('./run_log/experiment2-2/2022-10-20/events.out.tfevents.1666242294.BigBlack.26800.0').Reload()
    ea2 = event_accumulator.EventAccumulator('./run_log/experiment2-2/2022-10-20/events.out.tfevents.1666267719.BigBlack.11356.0').Reload()
    # ea3 = event_accumulator.EventAccumulator('./run_log/experiment2-2/2022-10-20/events.out.tfevents.1666068898.BigBlack.33408.0').Reload()
    ea1_loss = ea1.scalars.Items('RAM_Net/train_loss')
    ea2_loss = ea2.scalars.Items('resnet18/train_loss')
    ea3_loss = ea2.scalars.Items('densenet121/train_loss')

    re_list_x = [i.step for i in ea1_loss]
    re_list_y = [i.value for i in ea1_loss]
    de_list_x = [i.step for i in ea2_loss]
    de_list_y = [i.value for i in ea2_loss]
    my_list_x = [i.step for i in ea3_loss]
    my_list_y = [i.value for i in ea3_loss]

    x = [re_list_x, de_list_x, my_list_x]
    y = [re_list_y, de_list_y, my_list_y]
    labels = ['1C', 'resnet18', 'densenet121']
    color = ['b', 'g', 'r']
    font = {'family': 'Times New Roman', 'size' : 16}
    line_type = ['-.', '--', '-']

    plt.xlabel('Epoch', font)
    plt.ylabel('Loss', font)
    for i in range(3):
        poly_re = np.polyfit(x[i], y[i], deg=40)
        y_value = np.polyval(poly_re, x[i])
        plt.plot(x[i], y_value, line_type[i], label=labels[i], color=color[i])
        # plt.savefig(f'./run_log/{labels[i]}.png')
        # plt.close()

    plt.legend(prop=font)
    plt.savefig('./run_log/tensorboard_draw/loss_for_three_data2.png')


def classPredict():
    """
    把预测出来的有缺陷的图片名字存到 lists 中
    """
    test_im_path = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Testset_for_classify\Croped_im'
    imgs = os.listdir(test_im_path)

    # net1 = models.resnet18(pretrained=False, num_classes=2)
    # net1 = models.densenet121(pretrained=False, num_classes=2)
    net1 = my_net.new_cbam_net(kernel_size=3, padding=1)

    # net1.load_state_dict(torch.load('./models/22-9-13-resnet18-100epoch-3datas.pth'))
    # net1.load_state_dict(torch.load('./models/22-9-13-densenet121-100epoch-3datas.pth'))
    net1.load_state_dict(torch.load('./models/22-9-13-my_net-100epoch-3datas.pth'))

    net1.eval()
    net1.to('cuda')

    lists = []
    for imgp in imgs:
        # load image
        path = test_im_path + '/' + imgp
        img = cv2.resize(cv2.imread(path), (64, 64))  # BGR

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cuda').float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension

        pred = net1(img)
        p = pred.argmax(axis=1)
        if p == 0:
            lists.append(imgp)
    return lists


def drawList(list):
    """
    根据输入列表中判断为缺陷的图像名称，把他标记在对应的原始图像上
    """
    draw_list = ['104.jpg', '144.jpg', '145.jpg']
    draw_path = ['./dataset/Dataset211118/FDAOaddE',
                 './dataset/Dataset211118/FDTOaddE',
                 './dataset/Dataset211118/FATOaddE']

    for dp in draw_path:
        for dl in draw_list:
            im_path = dp + '/' + dl
            im = cv2.imread(im_path)
            k = 0
            for pd in list:
                if pd.split('_')[0] == dl.split('.')[0]:
                    start = pd.split('_')[1].split('.')[0]
                    # im = cv2.applyColorMap(im, 2)
                    cv2.rectangle(im, (int(start)*32, 0), (int(start)*32+63, 63), (0, 0, 255), 1)
                    # cv2.putText(im, str(k), (int(start)*32, 17), 0, 0.75, (0, 255, 0))
                    k += 1
            save_name = dp.split('/')[3][0: 3]
            save_path = './dataset/Testset_for_classify/predict/' + save_name + '-' + dl.split('.')[0] + '.png'
            cv2.imwrite(save_path, im)


def statisticTensorBoard():
    net_set = ['densenet121', 'resnet18', 'my_net']
    data_set = ['1C', '2C', '3C']
    or_path = './run_log/experiment2-3'


    for net in net_set:
        for channel in data_set:
            ts_path = or_path + '/' + net + '/' + channel


def threshTest():
    """
    使用阈值的方式测试测试集
    """
    thresh = 50
    test_root = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Dataset220217\dataset_220217_rgb\test_data'
    path = ['defects', 'undefects']

    defects_path = test_root + '/' + path[0]
    undefects_path = test_root + '/' + path[1]
    imgs1 = os.listdir(defects_path)
    imgs2 = os.listdir(undefects_path)

    tp = 0
    fn = 0
    for imgs in imgs1:
        im = cv2.imread(defects_path + '/' + imgs)
        b, g, r = cv2.split(im)

        minb, maxb, _, _ = cv2.minMaxLoc(b)
        ming, maxg, _, _ = cv2.minMaxLoc(g)
        minr, maxr, _, _ = cv2.minMaxLoc(r)

        # # fda+fdt+fat
        # if max([maxb, maxg]) > 128 + thresh or maxr > thresh or min([minb, ming]) < 128 - thresh:
        #     tp += 1
        # else:
        #     fn += 1

        # # fda+fdt
        # if max([maxb, maxg]) > 128 + thresh or min([minb, ming]) < 128 - thresh:
        #     tp += 1
        # else:
        #     fn += 1

        # fda
        if maxb > 128 + thresh or minb < 128 - thresh:
            tp += 1
        else:
            fn += 1

    fp = 0
    tn = 0
    for imgs in imgs2:
        im = cv2.imread(undefects_path + '/' + imgs)
        b, g, r = cv2.split(im)

        minb, maxb, _, _ = cv2.minMaxLoc(b)
        ming, maxg, _, _ = cv2.minMaxLoc(g)
        minr, maxr, _, _ = cv2.minMaxLoc(r)

        # # fda+fdt+fat
        # if max([maxb, maxg]) > 128 + thresh or maxr > thresh or min([minb, ming]) < 128 - thresh:
        #     fp += 1
        # else:
        #     tn += 1

        # # fda+fdt
        # if max([maxb, maxg]) > 128 + thresh or min([minb, ming]) < 128 - thresh:
        #     fp += 1
        # else:
        #     tn += 1

        # fda
        if maxb > 128 + thresh or minb < 128 - thresh:
            fp += 1
        else:
            tn += 1

    acc = (tp + tn) / (len(imgs1) + len(imgs2))
    Recall = tp / (tp + fn)
    Precise = tp / (tp + fp)
    F1 = 2 * Recall * Precise / (Recall + Precise)
    print(f'acc : {acc}, Recall : {Recall}, Precise : {Precise}, F1 : {F1}')


def testTime():
    test_im_path = r'D:\gsw\Projects\WOLIU\bigwhite\dataset\Testset_for_classify\Croped_im'
    imgs = os.listdir(test_im_path)
    batch = 200

    net = models_c.RAM_Net()

    net.load_state_dict(torch.load('./run_log/experiment2-2/2022-10-20-RAM_Net-104-F191.pth'))

    net.eval()
    net.to('cuda')

    t1 = time.time()
    initial_input = torch.randn(1, 3, 64, 64).cuda()
    out = net(initial_input)
    print(out)
    t2 = time.time()
    print(f'初始化图像运行时间：{(t2 - t1) * 1000} ms')

    test_input = initial_input
    t3 = time.time()
    for i in range(batch-1):
        imgp = imgs[i]
        # load image
        path = test_im_path + '/' + imgp
        img = cv2.resize(cv2.imread(path), (64, 64))  # BGR

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cuda').float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension
        test_input = torch.cat([test_input, img], dim=0)
    t4 = time.time()
    print(f'{batch}张图像读取的时间：{(t4 - t3) * 1000} ms')
    t5 = time.time()
    pred = net(test_input.cuda())
    t6 = time.time()
    print(f'{batch}张图像预测的时间：{(t6 - t5) * 1000} ms')
    print(f'总预测的时间：{(t6 - t3) * 1000} ms')
    p = pred.argmax(axis=1)
    print(p)


if __name__ == '__main__':

    # select_list = [1, 2, 4, 5, 6, 7, 8, 16, 17, 18, 20, 21, 23, 24, 26, 46, 52, 56, 78]
    #
    # lists = classPredict()
    # drawList(lists)

    # threshTest()

    testTime()


