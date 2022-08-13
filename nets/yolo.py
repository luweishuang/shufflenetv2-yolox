import torch
import torch.nn as nn

from nets.shufflenet_v2_plus import shufflenet_v2_plus
from nets.detection_head import YoloxHead


class ShuffleNetV2Plus(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(ShuffleNetV2Plus, self).__init__()
        self.model = shufflenet_v2_plus(pretrained=pretrained, **kwargs)

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
    def __init__(self, num_classes, pretrained=False, model_size="Large"):
        super(YoloBody, self).__init__()
        self.backbone = ShuffleNetV2Plus(pretrained=pretrained, model_size=model_size)
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
    model = YoloBody(num_classes=1)
    inputs = torch.rand((2, 3, 640, 640))
