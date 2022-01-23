import os
from pathlib import Path
import cv2
from torch.utils.data import Dataset


help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


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
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        return img, labels


