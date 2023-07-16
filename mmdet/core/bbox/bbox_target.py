import torch
import numpy as np
from ..utils import multi_apply
from .transforms import bbox2delta


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                pos_gt_counts_list,          #Fixme: +
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    counts, count_weights, labels, label_weights, bbox_targets, bbox_weights = multi_apply(      #Fixme:  +
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        pos_gt_counts_list,      #Fixme: +
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        counts = torch.cat(counts, 0)           #Fixme:  +

        label_weights = torch.cat(label_weights, 0)
        count_weights = torch.cat(count_weights, 0)            #Fixme:  +

        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return counts, count_weights, labels, label_weights, bbox_targets, bbox_weights            #Fixme:  +


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       pos_gt_counts,   #Fixme: +
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    counts = pos_bboxes.new_zeros(num_samples, dtype=torch.long)    #Fixme: +  Resized binary count new_zeros(num_samples, dtype=torch.long)==>new_zeros(num_samples, 7)

    label_weights = pos_bboxes.new_zeros(num_samples)
    count_weights = pos_bboxes.new_zeros(num_samples)  #Fixme: +  Resized binary count new_zeros(num_samples)==>new_zeros(num_samples, 7)

    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        counts[:num_pos] = pos_gt_counts                       # Fixme: +

        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        count_weights[:num_pos] = pos_weight       #Fixme: +  Resized binary count count_weights[:num_pos] = pos_weight ==> count_weights[:num_pos,:] = pos_weight

        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
        # Fixme: +  Resized binary count count_weights[-num_neg:] = 1.0 ==> count_weights[-num_neg:, :] = 1.0
        # add weight:  count_weights[-num_neg:, :] = 1.0 ==> count_weights[-num_neg:,:] = [2**0,2**1,2**2,2**3,2**4,2**5,2**6
        #count_weights[-num_neg:,:] = 1.0#np.array([2**0,2**1,2**2,2**3,2**4,2**5,2**6])
    #print('core/bbox/bbox_target.py',cfg,cfg.assigner,cfg.assigner.min_pos_iou)
    #if cfg.assigner.min_pos_iou == 0.5:
    #    for idx in range(num_pos):
    #        count_weights[idx]=torch.from_numpy(np.array([2**6,2**5,1,1,1,1,1])).cuda()
    #elif cfg.assigner.min_pos_iou == 0.6:
    #    for idx in range(num_pos):
    #        count_weights[idx]=torch.from_numpy(np.array([2**6,2**5,2**4,2**3,1,1,1])).cuda()
    #else:# cfg.assigner.min_pos_iou == 0.7:
    #    for idx in range(num_pos):
    #        count_weights[idx]=torch.from_numpy(np.array([2**6,2**5,2**4,2**3,2**2,2**1,1])).cuda()

    #for idx in range(num_pos):
    #    count_weights[idx]=torch.from_numpy(np.array([14.0,12.0,10.0,8.0,6.0,4.0,2.0,1.0])).cuda()



    return counts, count_weights, labels, label_weights, bbox_targets, bbox_weights


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros(
        (bbox_targets.size(0), 4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros(
        (bbox_weights.size(0), 4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
