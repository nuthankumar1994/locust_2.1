import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmcv.runner import runner #current_lr #epoch,iter

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms, multiclass_nms_count)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
import numpy as np
import copy

@HEADS.register_module
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 with_count = True,          #Fixme: +
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=141,  #81,
                 num_counts=57,  #81,                 
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_count=dict( 
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0)):    #Fixme:  'MSELoss', loss_weight=1.0   ||   'SmoothL1Loss', beta=1.0, loss_weight=1.0, pick=-1
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg or with_count      #Fixme: +
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.with_count = with_count             #Fixme: +
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_counts = num_counts
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False

        self.loss_count = build_loss(loss_count)         #Fixme: +
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        if self.with_count:  
            self.fc_count = nn.Linear(in_channels, num_counts)  #Fixme: +  Resized binary count 1==>7
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
        if self.with_count:
            nn.init.normal_(self.fc_count.weight, 0, 0.001)
            nn.init.constant_(self.fc_count.bias, 0)

    def Stage1Div(self, Counts):
        ''' Stage1: Divide the range into 8 parts. '''
        Counts_np = Counts.cpu().numpy()
        GtCountsDiv_list = Counts_np.copy()
        # print(Counts_np.shape, Counts_np[0], Counts_np[0].shape)
        for idx, count in enumerate(Counts_np):
            #if count ==0:
            #    GtCountsDiv_list[idx] = 0
            if 0<=count<5:
                GtCountsDiv_list[idx] = 1-1
            elif 5<=count<11:
                GtCountsDiv_list[idx] = 2-1
            elif 11<=count<16:
                GtCountsDiv_list[idx] = 3-1
            elif 16<=count<21:
                GtCountsDiv_list[idx] = 4-1
            elif 21<=count<26:
                GtCountsDiv_list[idx] = 5-1
            elif 26<=count<31:
                GtCountsDiv_list[idx] = 6-1
            elif 31<=count<36:
                GtCountsDiv_list[idx] = 7-1
            elif 36<=count<41:
                GtCountsDiv_list[idx] = 8-1
            elif 41<=count<46:
                GtCountsDiv_list[idx] = 9-1
            elif 46<=count<100:
                GtCountsDiv_list[idx] = 10-1


        #GtCountBins_list = []
        #GtCountOneHot_list = [0,0,0,0]        
        #for Val in GtCountsDiv_list:
        #    GtCountOneHot_list[Val-1]=1
        #    GtCountBins_list.append(copy.deepcopy(GtCountOneHot_list))

        return GtCountsDiv_list #GtCountBins_list

    def Stage2Div(self, Counts):
        ''' Stage1: Divide the range into 6 parts. '''
        Counts_np = Counts.cpu().numpy()
        GtCountsDiv_list = Counts_np.copy()
        for idx, count in enumerate(Counts_np):
            #if count ==0:
            #    GtCountsDiv_list[idx] = 0
            if 0<=count<= 3:
                GtCountsDiv_list[idx] = 1-1
            elif 4<=count<= 7:
                GtCountsDiv_list[idx] = 2-1
            elif 8<=count<= 11:
                GtCountsDiv_list[idx] = 3-1
            elif 12<=count<= 15:
                GtCountsDiv_list[idx] = 4-1
            elif 16<=count<= 19:
                GtCountsDiv_list[idx] = 5-1
            elif 20 <=count <=23:
                GtCountsDiv_list[idx] = 6-1
            elif 24 <=count <=27:
                GtCountsDiv_list[idx] = 7-1
            elif 28 <=count <=31:
                GtCountsDiv_list[idx] = 8-1           
            elif 32 <=count <= 35:
                GtCountsDiv_list[idx] = 9-1
            elif 36 <=count <= 39:
                GtCountsDiv_list[idx] = 10-1
            elif 40 <=count <=43:
                GtCountsDiv_list[idx] = 11-1                
            elif 44 <=count <=47:
                GtCountsDiv_list[idx] = 12-1
            elif 48 <=count<= 51:
                GtCountsDiv_list[idx] = 13-1
            elif 52 <=count <=55:
                GtCountsDiv_list[idx] = 14-1
            elif 56 <=count <=59:
                GtCountsDiv_list[idx] = 15-1
            elif 60 <=count <=63:
                GtCountsDiv_list[idx] = 16-1

        #GtCountBins_list = []
        #GtCountOneHot_list = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]        
        #for Val in GtCountsDiv_list:
        #    GtCountOneHot_list[Val-1]=1
        #    GtCountBins_list.append(copy.deepcopy(GtCountOneHot_list))

        return GtCountsDiv_list  #GtCountBins_list


    def Stage3Div(self, Counts):
        ''' Stage1: Divide the range into 8 parts. '''
        Counts_np = Counts.cpu().numpy()
        GtCountsDiv_list = Counts_np.copy()
        for idx, count in enumerate(Counts_np):
            if count ==0:
                GtCountsDiv_list[idx] = 0
            else:
                GtCountsDiv_list[idx] -= 1
        #GtCountBins_list = []
        #GtCountOneHot_list = [0 for i in range(64)] 
        #for Val in GtCountsDiv_list:
        #    GtCountOneHot_list[Val-1]=1
        #    GtCountBins_list.append(copy.deepcopy(GtCountOneHot_list))

        return GtCountsDiv_list  #GtCountBins_list


    def MatchCountWeights(self, CountWeights):
        ''' Stage1: Divide the range into 8 parts. '''
        Counts_np = CountWeights.cpu().numpy()
        GtCountsDiv_list = Counts_np.copy()
        GtCountBins_list = []
        for Val in GtCountsDiv_list:
            #GtCountBins_list.append([Val,Val,Val,Val,Val,Val])
            #GtCountBins_list.append([32*Val,16*Val,8*Val,4*Val,2*Val,1*Val])
            GtCountBins_list.append([11*Val,9*Val,7*Val,5*Val,3*Val,1*Val])
            #GtCountBins_list.append([14.0,12.0,10.0,8.0,6.0,4.0,2.0,1.0])
        return GtCountBins_list

    def MatchCountWeights_Stage1(self, CountWeights):
        ''' Stage1: Divide the range into 8 parts. '''
        Counts_np = CountWeights.cpu().numpy()
        GtCountsDiv_list = Counts_np.copy()
        GtCountBins_list = []
        for Val in GtCountsDiv_list:
            GtCountBins_list.append([32*Val,16*Val,0*Val,0*Val,0*Val,0*Val])
        return GtCountBins_list
    def MatchCountWeights_Stage2(self, CountWeights):
        ''' Stage1: Divide the range into 8 parts. '''
        Counts_np = CountWeights.cpu().numpy()
        GtCountsDiv_list = Counts_np.copy()
        GtCountBins_list = []
        for Val in GtCountsDiv_list:
            GtCountBins_list.append([32*Val,16*Val,8*Val,4*Val,0*Val,0*Val])
        return GtCountBins_list
    def MatchCountWeights_Stage3(self, CountWeights):
        ''' Stage1: Divide the range into 8 parts. '''
        Counts_np = CountWeights.cpu().numpy()
        GtCountsDiv_list = Counts_np.copy()
        GtCountBins_list = []
        for Val in GtCountsDiv_list:
            GtCountBins_list.append([32*Val,16*Val,8*Val,4*Val,2*Val,1*Val])
        return GtCountBins_list


    def int2bin(self, Val):
        bin_list = [0,0,0,0,0,1]
        Valb = bin(Val)
        Valb = Valb[2:].zfill(6)  #填充0，最长8位，
        bin_list2 = [int(e) for e in Valb]
        if len(bin_list2) == 6:
            return bin_list2
        return bin_list
    

    def bin2int(self, bin_list):
        if len(bin_list) != 6:
            return 1
        Valb=''
        for e in bin_list:
            if e < 0.5:
                Valb += '0'
            else:
                Valb += '1'
        return int(Valb,2)

    @auto_fp16()
    def forward(self, x):
        # print('./bbox_head/forward======>')
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        count_score = self.fc_count(x) if self.with_count else None
        # print('bbox_head/forwad==>',count_score.shape,cls_score.shape,bbox_pred.shape)
        return cls_score, bbox_pred, count_score

    def get_target(self, sampling_results, gt_bboxes, gt_labels, gt_counts,            # Fixme: +
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        pos_gt_counts = [res.pos_gt_counts for res in sampling_results]

        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            pos_gt_counts,                        # Fixme: +
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'count_score'))
    def loss(self, Stage_i,    # Fixme: + Resized Coarse-to-fine division and loss. Add the stage Flag.
             cls_score,
             bbox_pred,
             count_score,
             counts,
             count_weights,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        #print("Runner",runner.epoch)
        #print("Runner",self.epoch)
        
        #print("bbox_head/cls_score",cls_score.shape,label_weights.shape, labels.shape)   #bbox_head/cls_score torch.Size([2048, 141]) torch.Size([2048]) torch.Size([2048])
        # bbox_head/count_score tensor(0.0046, device='cuda:0', grad_fn=<MulBackward0>) torch.Size([1024, 1]) torch.Size([1024, 1])
        #print('bbox_head/count_score',counts.shape, count_score.shape)  #

        Learning_cnt_weights=1.0
        Learning_cls_weights=1.0        
        Learning_box_weights=1.0        
        if Stage_i == 0:
            counts = torch.from_numpy(np.array(self.Stage3Div(counts))).cuda()
            #count_weights = torch.from_numpy(np.array(self.MatchCountWeights_Stage1(count_weights),np.float32)).cuda()
            Learning_cnt_weights=0.1
            Learning_cls_weights=1.0   
            Learning_box_weights=1.0 
        
        elif Stage_i == 1:
            counts = torch.from_numpy(np.array(self.Stage3Div(counts))).cuda()
            #count_weights = torch.from_numpy(np.array(self.MatchCountWeights_Stage2(count_weights),np.float32)).cuda()            
            Learning_cnt_weights=0.1
            Learning_cls_weights=0.5        
            Learning_box_weights=0.5
            
        else:
            counts = torch.from_numpy(np.array(self.Stage3Div(counts))).cuda()
            #count_weights = torch.from_numpy(np.array(self.MatchCountWeights_Stage3(count_weights),np.float32)).cuda()
            Learning_cnt_weights=0.1
            Learning_cls_weights=0.25        
            Learning_box_weights=0.25

        #counts = torch.from_numpy(np.array(self.Stage3Div(counts))).cuda()
        #BatchSize = 12
        #Learning_cnt_weights=Learning_box_weights/BatchSize
        #Learning_cls_weights=Learning_box_weights/BatchSize
        #Learning_box_weights=Learning_box_weights/BatchSize  

        #print("bbox_head/count_score",count_score.shape,count_weights.shape, counts.shape)
        if count_score is not None:
            avg_factor = max(torch.sum(count_weights > 0).float().item(), 1.)
            losses['loss_count'] = Learning_cnt_weights*self.loss_count(
                count_score,
                counts,
                count_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc_count'] = accuracy(count_score, counts)

        #print("bbox_head/cls_score",cls_score.shape,label_weights.shape, labels.shape)
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = Learning_cls_weights*self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = Learning_box_weights*self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)

       
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'count_score'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       count_score,              # Fixmea: + for test
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if isinstance(count_score, list):
            count_score = sum(count_score) / float(len(count_score))
        cnt_scores = F.softmax(count_score, dim=1) if count_score is not None else None
        
        ArgMax_tensor = torch.argmax(cnt_scores,dim=1)+1
        count_score = ArgMax_tensor.unsqueeze(1).type(torch.float)
        #print('\nbbox_heads/bbox_head/get_det_bboxes',count_score.shape,scores.shape,bbox_pred.shape,cls_score.shape)
        #print('\nbbox_heads/bbox_head/get_det_bboxes',cnt_scores.shape,ArgMax_tensor.shape,count_score.shape)
        #print('\nbbox_heads/bbox_head/get_det_bboxes',ArgMax_tensor[:50],count_score[:50])
        #Fixme: For test + Resized count binary: transform the 7 bit binary to the float32 type.        
        #count_score_np = count_score.cpu().numpy() 
        #print(count_score_np)   bin2int  N3bin2int  
        #count_score_list = np.array([[self.bin2int(e)] for e in count_score_np],dtype=np.float32)
        #count_score = torch.from_numpy(count_score_list).cuda()


        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores, count_score
        else:
            #print('bbox_heads/bbox_head/get_det_bboxes ==>', count_score.shape, scores.shape, bbox_pred.shape)
            #Fixme: mmdet/core/post_processing/bbox_nms.py/multiclass_nms_count()
            det_bboxes, det_labels, det_counts = multiclass_nms_count(bboxes, scores, count_score,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels, det_counts

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
