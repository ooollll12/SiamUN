import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.anchors import Anchors
from model.unet import UNet
from model.rpn import RPN, DepthCorr
from model.loss import *


class DownS(nn.Module):
    def __init__(self, inplane, outplane):
        super().__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x

class MaskCorr(nn.Module):
    def __init__(self, oSz=63):
        super().__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
        return self.mask(z, x)


class Refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 4, 3, padding=1),nn.ReLU())

        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(64, 8, 3, padding=1), nn.ReLU())

        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(128, 16, 3, padding=1), nn.ReLU())

        self.v3 = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(256, 32, 3, padding=1), nn.ReLU())

        self.h3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

        self.h2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

        self.h1 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())

        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 8, 3, padding=1)
        self.post2 = nn.Conv2d(8, 4, 3, padding=1)
        self.post3 = nn.Conv2d(4, 1, 3, padding=1)
        

    def forward(self, f, corr_feature, pos = [0, 0] ):

        p0 = torch.nn.functional.pad(f[0], [32, 32, 32, 32])[:, :, 8*pos[0]:8*pos[0]+127, 8*pos[1]:8*pos[1]+127]
        p1 = torch.nn.functional.pad(f[1], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+63, 4*pos[1]:4*pos[1]+63]
        p2 = torch.nn.functional.pad(f[2], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
        p3 = torch.nn.functional.pad(f[3], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]

        p4 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)



        out = self.deconv(p4)
        out = self.post0(F.interpolate(self.h3(out) + self.v3(p3), size=(31, 31)))
        out = self.post1(F.interpolate(self.h2(out) + self.v2(p2), size=(63, 63)))
        out = self.post2(F.interpolate(self.h1(out) + self.v1(p1), size=(127, 127)))
        out = self.post3(self.h0(self.h0(out) + self.v0(p0)))

        out = out.view(-1, 127*127)
        return out



class SiamUN(nn.Module):
    def __init__(self, anchors=None, o_sz=127, g_sz=127):
        super().__init__()
        self.backbone = UNet()
        self.adjust = DownS(1024, 256)
        
        self.anchor_numm = 5
        self.rpn_model = RPN(anchor_num=self.anchor_numm, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])
        self.all_anchors = None

    def template(self, template):
        _, self.zf = self.forward_all(template)

    def set_all_anchors(self, image_center, size):
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]


    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                      rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)#Lscore分类损失

        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)#Lbox位置损失

        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(rpn_pred_mask, label_mask, label_mask_weight)#Lmask掩码损失

        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

   


    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls
    
    def forward_all(self, x):
        output = self.backbone(x)
        p3 = self.adjust(output[-1])
        return output, p3





    def forward(self, z , x):
        '''
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']
        '''

        template_feature,z_f = self.forward_all(z)
        search_feature,x_f  = self.forward_all(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(z_f,x_f)
        corr_feature = self.mask_model.mask.forward_corr(z_f,x_f)  # (b, 256, w, h)
        rpn_pred_mask = self.refine_model(template_feature, corr_feature)
        softmax=self.training
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        
        return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask

        '''
        outputs = dict()

        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, rpn_pred_mask, template_feature, search_feature]

        if self.training:
            rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
                self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                                   rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
            outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs
        '''


    def track(self, search):
        search = self.backbone(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        self.feature, self.search = self.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(self.zf, self.search)
        self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        pred_mask = self.mask_model.mask.head(self.corr_feature)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

    def track_refine(self, pos):
        pred_mask = self.refine_model(self.feature, self.corr_feature, pos=pos, test=True)
        return pred_mask



if __name__ == "__main__":
    anchors = {
            "stride": 8,
            "ratios": [0.33, 0.5, 1, 2, 3],
            "scales": [8],
            "round_dight": 0
    }
    net = SiamUN(anchors)
    input = {}
    print(net)
    net = net.cuda()

    var1 = torch.FloatTensor(1, 3, 127, 127).cuda()
    input['template'] = var1

    #a = net(var)
    print('*************')
    var2 = torch.FloatTensor(1, 3, 255, 255).cuda()
    input['search'] = var2
    #a = net.backbone(var1)


    b = net(var1, var2)
    mask = b[2].sigmoid().squeeze().view(127,127).cpu().data.numpy()
    print(mask)