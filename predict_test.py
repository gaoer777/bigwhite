import os
import json
import torch
import cv2
import numpy as np
import UtilFunctions as utf
from matplotlib import pyplot as plt
from draw_box_utils import draw_box


def main():
    img_path = "362.png"
    category_index = {1: "defect"}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load("./models/object_detect.pth")
    model.to(device)
    model.eval()
    with torch.no_grad():
        # init 创建一张图片初始化模型
        img = torch.zeros((1, 3, 63, 512), device=device)
        model(img)

        # 读取测试图片
        img_o = cv2.imread(img_path)  # BGR
        assert img_o is not None, "Image Not Found " + img_path

        # Convert 将图片转换为RGB的tensor，并将图像维度扩充
        img = img_o[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension

        # 得到预测结果
        pred = model(img)[0]  # only get inference result

        # 对预测结果进行过滤，采用非极大值抑制的方法
        # conf_thres滤除一部分低于这个阈值的目标框，代表是否为目标的置信度
        pred = utf.non_max_suppression(pred, conf_thres=0.45, max_num=5,
                                       iou_thres=0.6, multi_label=True)[0]

        if pred is None:
            print("No target detected.")
            exit(0)

        # process detections
        pred[:, :4] = utf.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        print(pred.shape)

        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = np.ones(pred.shape[0], dtype=np.int32)

        img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
        plt.imshow(img_o)
        plt.show()

        img_o.save(img_path.split('.')[0] + "_d" + ".png")
        # img_o.save("164_d.png")


if __name__ == "__main__":
    main()
