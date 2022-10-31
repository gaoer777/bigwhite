import os
from abc import ABC
from pathlib import Path
import cv2
import random
import numpy as np
import torch

import UtilFunctions as utf
from torch.utils.data import Dataset

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


class LoadClassifyImages(Dataset):
    def __init__(self, path, transform):
        self.classes = os.listdir(path)
        self.transform = transform
        self.imgs = []
        for c in range(len(self.classes)):
            c_path = path + '/' + self.classes[c]
            self.imgs += [str(c) + '__' + c_path + '/' + x for x in os.listdir(c_path)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # load image
        path = self.imgs[index].split('__')[1]
        label = int(self.imgs[index].split('__')[0])
        img = cv2.imread(path)  # BGR

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        # img /= 255.0
        img = img[:, :, ::-1]  # .transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = self.transform(img)

        return img, label


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, transform):
        self.transform = transform
        # read images file path to img_files
        path = str(Path(path))
        self.img_files = os.listdir(path)
        self.img_files.sort()  # 防止不同系统排序不同，导致shape文件出现差异
        self.img_files = [os.path.join(path, x) for x in self.img_files]
        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)

        # Define labels
        # 遍历设置图像对应的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.img_files]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path

        labels = path.replace("images", "labels").replace(os.path.splitext(path)[-1], ".txt")
        # load labels

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = self.transform(img)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return img, labels


class LoadTestImages(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        # read images file path to img_files
        path = str(Path(path))
        self.img_files = os.listdir(path)
        self.img_files.sort()  # 防止不同系统排序不同，导致shape文件出现差异
        self.img_files = [os.path.join(path, x) for x in self.img_files]
        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        img = self.transform(img)

        return img


class LoadCOCOImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, transform):
        self.transform = transform
        # read images file path to img_files
        path = str(Path(path))
        self.img_files = os.listdir(path)
        self.img_files.sort()  # 防止不同系统排序不同，导致shape文件出现差异
        self.img_files = [os.path.join(path, x) for x in self.img_files]
        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)

        # Define labels
        # 遍历设置图像对应的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt") for x in self.img_files]

    def __len__(self):
        return len(self.img_files) // 5

    def __getitem__(self, index):
        # load image
        img, labels, n = self.catImages(index)

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        img = self.transform(img)

        return img, labels, n

    def catImages(self, index):
        """
        将COCO数据集中每5张图像进行一次拼接，然后resize到(64 512)，并将框的尺寸转换为对应的尺度
        """
        imgs = []
        targets = []
        start = 0
        index *= 5

        for i in range(index, index+5):
            img = cv2.imread(self.img_files[i])
            imgs.append(img)

            label_path = self.img_files[i].replace('images', 'labels').replace('jpg', 'txt')
            target = utf.getOneTarget(label_path)  # 读取label文件，将坐标读取出来
            i %= 5
            if len(target) > 0:  # 如果有目标
                target[:, [1, 3]] *= img.shape[1]
                target[:, [2, 4]] *= img.shape[0]

                if imgs[i].shape[0] > imgs[i].shape[1]:  # 使图像的h <= w，框的维度也要变化
                    imgs[i] = np.rot90(imgs[i])
                    target = target[:, [0, 2, 1, 4, 3]]
                    target[:, 2] = imgs[i].shape[0] - target[:, 2]

                scale = 64 / imgs[i].shape[0]
                target[:, 1:] *= scale
                target = target[target[:, 3] > 10]
                target = target[target[:, 4] > 10]
                target[:, 1] += start
                target[:, 1:] = utf.xywh2xyxy_(target[:, 1:])
            else:
                if imgs[i].shape[0] > imgs[i].shape[1]:  # 使图像的h <= w，框的维度也要变化
                    imgs[i] = np.rot90(imgs[i])

            scale = 64 / imgs[i].shape[0]
            imgs[i] = cv2.resize(imgs[i], (int(scale * imgs[i].shape[1]), 64))
            start += imgs[i].shape[1]

            if len(targets) > 0 and len(target) > 0:
                targets = np.concatenate((targets, target), axis=0)
            elif len(targets) == 0:
                targets = target

        big_im = cv2.hconcat(imgs)
        scale = 512 / big_im.shape[1]
        big_im = cv2.resize(big_im, (512, 64))
        targets[:, [1, 3]] *= scale
        targets = targets.astype(int)

        n = len(targets)
        temp = np.zeros((50-n, 5))
        targets = np.concatenate((targets, temp), axis=0)

        return big_im, targets, n
        # for box in targets:
        #     cv2.rectangle(big_im, (box[1], box[2]), (box[3], box[4]), (0, 0, 225), 2)

        # start = random.randint(start, )
        # cv2.imshow('001', big_im)
        # cv2.waitKey()
        # save_path = self.img_files[index].replace('images', 'test')
        # cv2.imwrite(save_path, big_im)


class fiveCrossdataset(Dataset):  # for training/testing
    def __init__(self, path, transform):
        self.transform = transform
        # read images file path to img_files
        path = str(Path(path))
        self.img_files = os.listdir(path)
        self.img_files.sort()  # 防止不同系统排序不同，导致shape文件出现差异
        self.img_files = [os.path.join(path, x) for x in self.img_files]
        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)

        # Define labels
        # 遍历设置图像对应的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt") for x in self.img_files]

    def __len__(self):
        return len(self.img_files) // 5

    def __getitem__(self, index):
        # load image
        img, labels, n = self.catImages(index)

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        img = self.transform(img)

        return img, labels, n

    def catImages(self, index):
        """
        将COCO数据集中每5张图像进行一次拼接，然后resize到(64 512)，并将框的尺寸转换为对应的尺度
        """
        imgs = []
        targets = []
        start = 0
        index *= 5

        for i in range(index, index+5):
            img = cv2.imread(self.img_files[i])
            imgs.append(img)

            label_path = self.img_files[i].replace('images', 'labels').replace('jpg', 'txt')
            target = utf.getOneTarget(label_path)  # 读取label文件，将坐标读取出来
            i %= 5
            if len(target) > 0:  # 如果有目标
                target[:, [1, 3]] *= img.shape[1]
                target[:, [2, 4]] *= img.shape[0]

                if imgs[i].shape[0] > imgs[i].shape[1]:  # 使图像的h <= w，框的维度也要变化
                    imgs[i] = np.rot90(imgs[i])
                    target = target[:, [0, 2, 1, 4, 3]]
                    target[:, 2] = imgs[i].shape[0] - target[:, 2]

                scale = 64 / imgs[i].shape[0]
                target[:, 1:] *= scale
                target = target[target[:, 3] > 10]
                target = target[target[:, 4] > 10]
                target[:, 1] += start
                target[:, 1:] = utf.xywh2xyxy_(target[:, 1:])
            else:
                if imgs[i].shape[0] > imgs[i].shape[1]:  # 使图像的h <= w，框的维度也要变化
                    imgs[i] = np.rot90(imgs[i])

            scale = 64 / imgs[i].shape[0]
            imgs[i] = cv2.resize(imgs[i], (int(scale * imgs[i].shape[1]), 64))
            start += imgs[i].shape[1]

            if len(targets) > 0 and len(target) > 0:
                targets = np.concatenate((targets, target), axis=0)
            elif len(targets) == 0:
                targets = target

        big_im = cv2.hconcat(imgs)
        scale = 512 / big_im.shape[1]
        big_im = cv2.resize(big_im, (512, 64))
        targets[:, [1, 3]] *= scale
        targets = targets.astype(int)

        n = len(targets)
        temp = np.zeros((50-n, 5))
        targets = np.concatenate((targets, temp), axis=0)

        return big_im, targets, n
        # for box in targets:
        #     cv2.rectangle(big_im, (box[1], box[2]), (box[3], box[4]), (0, 0, 225), 2)

        # start = random.randint(start, )
        # cv2.imshow('001', big_im)
        # cv2.waitKey()
        # save_path = self.img_files[index].replace('images', 'test')
        # cv2.imwrite(save_path, big_im)


# if __name__ == '__main__':

    # catImages('D:/gsw/Projects/WOLIU/yolov5/dataset/train2017/images', 0)
