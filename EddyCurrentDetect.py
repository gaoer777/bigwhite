import torch
from torch.utils import data
import datasets
import UtilFunctions as utf
from torchvision import transforms
from ObjectDetect0412 import ODAB


def objectDetectMethod(task):
    """
    output：[image index, X1, Y1, X2, Y2, confidence]*n
    """

    category_index = {1: "defect"}

    model_param_path = "models/object_detect/06-20-1-detect.pth"
    test_root = task['Data_dir']

    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.LoadTestImages(test_root, transform=transform)
    test_iter = data.DataLoader(test_data, 10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ODAB()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    model.eval()

    out = torch.zeros(1, 6)
    img_index = 0

    with torch.no_grad():
        for i, (X) in enumerate(test_iter):
            # 得到预测结果
            X = X.to(device)
            pred = model(X)  # only get inference result

            pred = torch.cat([pred[0][0], pred[1][0], pred[2][0]], dim=1)
            # 对预测结果进行过滤，采用非极大值抑制的方法
            # conf_thres滤除一部分低于这个阈值的目标框，代表是否为目标的置信度
            pred = utf.non_max_suppression_for_batch(pred, conf_thres=0.5, max_num=10,
                                                     iou_thres=0.2, multi_label=True)

            for preds in pred:
                if preds is not None:
                    preds = preds.cpu()
                    insert = torch.zeros(preds.shape[0])
                    insert = insert.unsqueeze(1)
                    insert[:] = img_index
                    preds = torch.cat((insert, preds), 1)
                    out = torch.cat((out, preds), 0)
                img_index += 1
    out = out[1:, :]

    return out


def eddyCurrentDetect(task):
    """
    @description  : 涡流检测 自动缺陷识别 处理函数
    ---------
    @param  : task: 请求参数
    -------
    @Returns  : back_info: 请求返回信息
    -------
    """
    back_info = {'Type': task['Type'], 'Name': task['Name']}
    objectDetectMethod(task)


if __name__ == "__main__":
    # main()
    task = {'Type': '123', 'Name': 'test', 'Data_dir': './dataset/Dataset220324/test/images'}
    eddyCurrentDetect(task)
