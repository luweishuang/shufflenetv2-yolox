import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

from nets.shufflenet_v2_plus import ShuffleNetV2_Plus
from nets.detection_head import YoloxHead


class ShuffleNetV2Plus(nn.Module):
    def __init__(self, model_size):
        super(ShuffleNetV2Plus, self).__init__()
        self.model = ShuffleNetV2_Plus(model_size=model_size)

    def forward(self, x):
        x = self.model.first_conv(x)
        out3 = self.model.features[0:8](x)
        out4 = self.model.features[8:16](out3)
        out5 = self.model.features[16:20](out4)
        return out3, out4, out5


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_classes, model_size="Large"):
        super(YoloBody, self).__init__()
        self.backbone = ShuffleNetV2Plus(model_size)
        if model_size == 'Small':
            in_filters = [104, 208, 416]
        elif model_size == "Medium":
            in_filters = [128, 256, 512]
        elif model_size == "Large":
            in_filters = [168, 336, 672]
        else:
            print("输入的宽度参数有误")
        self.head = YoloxHead(num_classes, in_filters)

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)
        # head
        return self.head(x2, x1, x0)


if __name__ == '__main__':
    model = YoloBody(1)
    model_dict = model.state_dict()
    pretrained_dict = torch.load('./shufflenetv2plus.pth')
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))