
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils import loc2bbox


class ProposalCreator():
    def __init__(self, mode, nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=600,
                 n_test_pre_nms=3000,
                 n_test_post_nms=300,
                 min_size=16):
        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):

        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()
        #-----------------------------------#
        #   将RPN网络预测结果转化成建议框，（roi得到的是左上角和右下角坐标，不是中心坐标，而且是以图片左上角为原点的）
        #-----------------------------------#
        roi = loc2bbox(anchor, loc)   # anchor是布满feature map每一个点的，loc是回归支路预测出来的50*50*9 = 22500个预测框的，anchor和loc都是22500个bbox

        #-----------------------------------#
        #   防止建议框超出图像边缘
        #-----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])  # 如果proposal bbox的左上角坐标和右下角x坐标超界，就钳位
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        
        #-----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        #-----------------------------------#
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]   # torch.where()函数的作用是按照一定的规则合并两个tensor类型。剔除宽或者高小于16的proposal bbox
        roi = roi[keep, :]   # 挑选满足条件的proposal bbox
        score = score[keep]  # 挑选满足条件的proposal bbox的相应的 score（score对应的是二分类的第二的值）

        #-----------------------------------#
        #   根据得分进行排序，取出建议框
        #-----------------------------------#
        order = torch.argsort(score, descending=True)  # 返回索引，而且是从大到小排（降序排）
        if n_pre_nms > 0:
            order = order[:n_pre_nms]   # 按分数从大到小取出 n_pre_nums 个proposal bbox，可以看作proposal的预处理
        roi = roi[order, :]   # 取出预处理后的 proposal bbox
        score = score[order]  # 取出预处理后的proposal bbox 的分数， 最后留下了12000个按分数从大到小排序的proposal bbox

        #-----------------------------------#
        #   对建议框进行非极大抑制
        #-----------------------------------#
        keep = nms(roi, score, self.nms_thresh)  # 返回满足nms条件的索引
        keep = keep[:n_post_nms]   # 留下满足nms条件的前600个proposal bbox
        roi = roi[keep]  # 挑出这 600 个proposal bbox
        return roi

# RPN是一个全卷积网络（fully convolutional network），这样对输入图片的尺寸就没有要求了
class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            mode = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        self.feat_stride = feat_stride   # 加入图像尺寸800*800经过特征采样变成50*50，下降了16倍，所以feature map中一个点对应原图一个16*16的感受野
        self.proposal_layer = ProposalCreator(mode)  #
        #-----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)  # 生成9个基础先验框
        n_anchor = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体，RPN第一路分支，用来分类
        #-----------------------------------------#
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)    # 1*1 卷积，9代表9个anchor_base,2代表每个anchor二分类，用交叉熵损失
        #-----------------------------------------#
        #   回归预测对先验框进行调整，RPN第二路分支，用来回归
        #-----------------------------------------#
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)     # 1*1 卷积，9个anchor，每个anchor有4个位置参数，4代表每个anchor对应的中心位置和长宽

        #--------------------------------------#
        #   对FPN的网络部分进行权值初始化
        #--------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):   # x.shape = [1024, 50, 2, 50]
        n, _, h, w = x.shape   # 1024, 50, 2, 50
        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        x = F.relu(self.conv1(x))
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        rpn_locs = self.loc(x)   # 回归支路
        print('rpn_locs.shape', rpn_locs.shape)  # [2, 36, 50, 50]
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # [2, 22500, 4]
        print('rpn_locs.shape', rpn_locs.shape)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        rpn_scores = self.score(x)  # 分类支路
        print('rpn_scores.shape', rpn_scores.shape)  # [2, 9*2, 50, 50]
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)  # [2, 22500, 2]
        print('rpn_scores.shape', rpn_scores.shape)
        
        #--------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        #--------------------------------------------------------------------------------------#
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)  # [2, 22500, 2]， dim=-1维也就是将最里面那层[]里面的数做softmax
        print('rpn_softmax_scores.shape', rpn_softmax_scores.shape)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()  # [2, 22500]，包含物体的概率
        print('rpn_fg_scores.shape', rpn_fg_scores.shape)
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # [2, 22500]
        print('rpn_fg_scores.shape', rpn_fg_scores.shape)

        #------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        #------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)   # 得到了一个遍布feature map的anchor
        print('anchor.shape', anchor.shape)   # (22500, 4)
        
        rois = list()
        roi_indices = list()
        for i in range(n):    # 得到每一张图片的 roi（regions of interest）
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            print('roi.shape', roi.shape)   # 得到经过NMS后的bbox， [600, 4]
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)
        print('len(rois)', len(rois))

        rois = torch.cat(rois, dim=0)
        print('rois.shape', rois.shape)   # [1200, 4]
        roi_indices = torch.cat(roi_indices, dim=0)   # [1200]  第一张图片的roi_indices 为0， 第二张为 1
        print('roi_indices.shape', roi_indices.shape)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
