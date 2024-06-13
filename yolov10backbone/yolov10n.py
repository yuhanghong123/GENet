from yolov10backbone.v10modules import Conv, C2f, SCDown, SPPF, PSA
import torch.nn as nn
import torch
from torchsummary import summary


class features_1024(nn.Module):
    def __init__(self, alpha=1):
        self.alpha = alpha
        super(features_1024, self).__init__()

        self.backbone = nn.Sequential(
            Conv(3, int(64 * self.alpha), 3, 2),
            Conv(int(64 * self.alpha), int(128 * self.alpha), 3, 2),
            C2f(int(128 * self.alpha), int(128 * self.alpha), True),
            Conv(int(128 * self.alpha), int(256 * self.alpha), 3, 2),
            C2f(int(256 * self.alpha), int(256 * self.alpha), True),
            SCDown(int(256 * self.alpha), int(512 * self.alpha), 3, 2),
            C2f(int(512 * self.alpha), int(512 * self.alpha), True),
            SCDown(int(512 * self.alpha), int(1024 * self.alpha), 3, 2),
            C2f(int(1024 * self.alpha), int(1024 * self.alpha), True),
            SPPF(int(1024 * self.alpha), int(1024 * self.alpha), 5),
            PSA(int(1024 * self.alpha), int(1024 * self.alpha))
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class features_2048(nn.Module):
    def __init__(self, alpha=1):
        self.alpha = alpha
        super(features_2048, self).__init__()

        self.backbone = nn.Sequential(
            Conv(3, int(64 * self.alpha), 3, 2),
            Conv(int(64 * self.alpha), int(128 * self.alpha), 3, 2),
            C2f(int(128 * self.alpha), int(128 * self.alpha), True),
            Conv(int(128 * self.alpha), int(256 * self.alpha), 3, 2),
            C2f(int(256 * self.alpha), int(256 * self.alpha), True),
            Conv(int(256 * self.alpha), int(512 * self.alpha), 3, 2),
            # SCDown(int(256 * self.alpha), int(512 * self.alpha), 3, 2),
            C2f(int(512 * self.alpha), int(512 * self.alpha), True),
            Conv(int(512 * self.alpha), int(1024 * self.alpha), 3, 2),
            # SCDown(int(512 * self.alpha), int(1024 * self.alpha), 3, 2),
            C2f(int(1024 * self.alpha), int(2048 * self.alpha), True),
            #SCDown(int(1024 * self.alpha), int(2048 * self.alpha), 3, 2),
            #C2f(int(1024 * self.alpha), int(2048 * self.alpha), True),
            SPPF(int(2048 * self.alpha), int(2048 * self.alpha), 5)
            # PSA(int(2048 * self.alpha), int(2048 * self.alpha))
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class ResGazeEs(nn.Module):

    def __init__(self, alpha=1):
        super(ResGazeEs, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = alpha
        self.fc = nn.Linear(int(1024*self.alpha), 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ResDeconv(nn.Module):
    def __init__(self, block, alpha=1):
        self.alpha = alpha
        self.inplanes=1024 * self.alpha
        super(ResDeconv, self).__init__()
        model = []
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 256, 2)] # 28
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 128, 2)] # 56
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 64, 2)] # 112
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 32, 2)] # 224
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 16, 2)] # 224
        model += [nn.Conv2d(16, 3, stride=1, kernel_size=1)]

        self.deconv = nn.Sequential(*model)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, features):
        img = self.deconv(features)
        return img
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = features_1024(alpha=0.5).to(device)
    input = torch.randn(1, 3, 224, 224).to(device)
    output = model(input)
    print(output.shape)
    input_size = (3, 224, 224)
    summary(model, input_size)
