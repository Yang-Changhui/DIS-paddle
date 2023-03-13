import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class muti_loss_fusion:
    def __init__(self,ignore_index=255):
        self.bce_loss = nn.BCELoss()
        self.ignore_index = ignore_index
    def __call__(self, preds, target):
        loss = 0.0
        for i in range(0, len(preds)):
            if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
                # tmp_target = _upsample_like(target,preds[i])
                tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
                loss = loss + self.bce_loss(preds[i], tmp_target)
            else:
                loss = loss + self.bce_loss(preds[i], target)
        return loss

@manager.LOSSES.add_component
class muti_loss_fusion_kl:
    def __init__(self, mode='MSE'):
        self.mode = mode
        self.bce_loss = nn.BCELoss()
        self.fea_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def foward(self, preds, target, dfs, fs):
        loss = 0.0

        for i in range(0, len(preds)):
            # print("i: ", i, preds[i].shape)
            if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
                # tmp_target = _upsample_like(target,preds[i])
                tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
                loss = loss + self.bce_loss(preds[i], tmp_target)
            else:
                loss = loss + self.bce_loss(preds[i], target)

        for i in range(0, len(dfs)):
            if self.mode == 'MSE':
                loss = loss + self.fea_loss(dfs[i], fs[i])  ### add the mse loss of features as additional constraints
                # print("fea_loss: ", fea_loss(dfs[i],fs[i]).item())
            elif self.mode == 'KL':
                loss = loss + self.kl_loss(F.log_softmax(dfs[i], axis=1), F.softmax(fs[i], axis=1))
                # print("kl_loss: ", kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1)).item())
            elif self.mode == 'MAE':
                loss = loss + self.l1_loss(dfs[i], fs[i])
                # print("ls_loss: ", l1_loss(dfs[i],fs[i]))
            elif self.mode == 'SmoothL1':
                loss = loss + self.smooth_l1_loss(dfs[i], fs[i])
                # print("SmoothL1: ", smooth_l1_loss(dfs[i],fs[i]).item())

        return loss
