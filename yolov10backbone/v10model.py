import torch
import torch.nn as nn
from yolov10backbone.yolov10n import features_1024,features_2048, ResGazeEs, ResDeconv, BasicBlock 
# import torch.utils.model_zoo as model_zoo


class Model(nn.Module):
    def __init__(self, alpha =1):
        super(Model, self).__init__()
        self.alpha = alpha
        # self.feature = features_1024(self.alpha)
        # self.gazeEs = ResGazeEs(self.alpha)
        # self.deconv = ResDeconv(BasicBlock,self.alpha)
        self.feature = features_2048(self.alpha)
        # self.feature.load_state_dict(torch.load(pretrained_url), strict=False )

        self.gazeEs = ResGazeEs(self.alpha *2)
        # self.gazeEs.load_state_dict(torch.load(pretrained_url), strict=False )

        self.deconv = ResDeconv(BasicBlock,self.alpha *2)

    def forward(self, x_in, require_img=True):
        features = self.feature(x_in["face"])
        gaze = self.gazeEs(features)
        if require_img:
            img = self.deconv(features)
            img = torch.sigmoid(img)
        else:
            img = None
        return gaze, img


class Gelossop():
    def __init__(self, attentionmap, w1=1, w2=1):
        self.gloss = torch.nn.L1Loss().cuda()
        # self.gloss = torch.nn.MSELoss().cuda()
        self.recloss = torch.nn.MSELoss().cuda()
        self.attentionmap = attentionmap.cuda()
        self.w1 = w1
        self.w2 = w2

    def __call__(self, gaze, img, gaze_pre, img_pre):
        loss1 = self.gloss(gaze, gaze_pre)
        # loss2 = 1-self.recloss(img, img_pre)
        loss2 = 1 - (img - img_pre)**2
        zeros = torch.zeros_like(loss2)
        loss2 = torch.where(loss2 > 0.75, zeros, loss2)
        loss2 = torch.mean(self.attentionmap * loss2)

        return self.w1 * loss1 + self.w2 * loss2


class Delossop():
    def __init__(self):
        self.recloss = torch.nn.MSELoss().cuda()

    def __call__(self, img, img_pre):
        return self.recloss(img, img_pre)

if __name__ == '__main__':
    model = Model()
    input = torch.randn(1, 3, 224, 224).cuda()
    output, _ = model(input)
    print(output.shape)
    