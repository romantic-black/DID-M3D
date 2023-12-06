import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from aug.iou3d_nms.iou3d_nms_utils import boxes_aligned_iou3d_gpu

class Hierarchical_Task_Learning:
    def __init__(self, epoch0_loss, stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term: self.index2term.index(term) for term in self.index2term}  # term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses = []
        self.loss_graph = {'seg_loss': [],
                           'size2d_loss': [],
                           'offset2d_loss': [],
                           'offset3d_loss': ['size2d_loss', 'offset2d_loss'],
                           'size3d_loss': ['size2d_loss', 'offset2d_loss'],
                           'heading_loss': ['size2d_loss', 'offset2d_loss'],
                           'depth_loss': ['size2d_loss', 'size3d_loss', 'offset2d_loss'],
                           'iou_loss': ['size2d_loss', 'size3d_loss', 'offset2d_loss', 'depth_loss', 'heading_loss']
                           }

    def compute_weight(self, current_loss, epoch):
        T = 140
        # compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)  # [1, 7]
        for term in self.loss_graph:
            if len(self.loss_graph[term]) == 0:  # 若没有前置任务
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)  # 权重为 1
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device)  # 权重为 0
                # update losses list
        if len(self.past_losses) == self.stat_epoch_nums:  # 跳过前五个epoch，因为有 pop(0) 所以只有 5 个epoch
            past_loss = torch.cat(self.past_losses)  # [5, L]
            mean_diff = (past_loss[:-2] - past_loss[2:]).mean(0)  # 前 3 个 - 后 3 个, [L]
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff  # 保留初始残差，以后一直留着
            c_weights = 1 - (mean_diff / self.init_diff).relu().unsqueeze(0)  # [1, L]

            time_value = min(((epoch - 5) / (T - 5)), 1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic]) != 0:  # 若有前置任务
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]
                    loss_weights[current_topic] = time_value ** (1 - control_weight)
                    if loss_weights[current_topic] != loss_weights[current_topic]:
                        for pre_topic in self.loss_graph[current_topic]:
                            print('NAN===============', time_value, control_weight,
                                  c_weights[0][self.term2index[pre_topic]], pre_topic, self.term2index[pre_topic])
            # pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)

        return loss_weights

    def update_e0(self, eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class DIDLoss(nn.Module):
    def __init__(self, epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch

    def forward(self, preds, targets):

        if targets['mask_2d'].sum() == 0:
            bbox2d_loss = 0
            bbox3d_loss = 0
            self.stat['offset2d_loss'] = 0
            self.stat['size2d_loss'] = 0
            self.stat['depth_loss'] = 0
            self.stat['offset3d_loss'] = 0
            self.stat['size3d_loss'] = 0
            self.stat['heading_loss'] = 0
            self.stat['iou_loss'] = 0
        else:
            bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
            bbox3d_loss = self.compute_bbox3d_loss(preds, targets)

        seg_loss = self.compute_segmentation_loss(preds, targets)

        mean_loss = seg_loss + bbox2d_loss + bbox3d_loss
        return float(mean_loss), self.stat

    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss

    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss

        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')

        loss = offset2d_loss + size2d_loss
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss

    def compute_bbox3d_loss(self, input, target, mask_type='mask_2d'):
        vis_depth = input['vis_depth'][input['train_tag']]
        att_depth = input['att_depth'][input['train_tag']]
        vis_depth_target = extract_target_from_tensor(target['vis_depth'], target[mask_type])
        att_offset_target = extract_target_from_tensor(target['att_depth'], target[mask_type])
        depth_mask_target = extract_target_from_tensor(target['depth_mask'], target[mask_type])
        ins_depth_target = extract_target_from_tensor(target['depth'], target[mask_type])
        vis_depth_uncer = input['vis_depth_uncer'][input['train_tag']]
        att_depth_uncer = input['att_depth_uncer'][input['train_tag']]

        # 各有各的 target, 同时结合起来也有 target
        vis_depth_loss = laplacian_aleatoric_uncertainty_loss(vis_depth[depth_mask_target],
                                                              vis_depth_target[depth_mask_target],
                                                              vis_depth_uncer[depth_mask_target])
        att_depth_loss = laplacian_aleatoric_uncertainty_loss(att_depth[depth_mask_target],
                                                              att_offset_target[depth_mask_target],
                                                              att_depth_uncer[depth_mask_target])

        ins_depth = input['ins_depth'][input['train_tag']]
        ins_depth_uncer = input['ins_depth_uncer'][input['train_tag']]
        ins_depth_loss = laplacian_aleatoric_uncertainty_loss(ins_depth.view(-1, 7 * 7),
                                                              ins_depth_target.repeat(1, 7 * 7),
                                                              ins_depth_uncer.view(-1, 7 * 7))

        depth_loss = vis_depth_loss + att_depth_loss + ins_depth_loss

        merge_prob = (-(0.5 * ins_depth_uncer).exp()).exp()
        merge_depth = (torch.sum((ins_depth * merge_prob).view(-1, 7 * 7), dim=-1) /
                       torch.sum(merge_prob.view(-1, 7 * 7), dim=-1)).unsqueeze(1)

        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]
        offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')

        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']]
        size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
        mean_size_target = extract_target_from_tensor(target['mean_size'], target[mask_type])
        size3d_loss = F.l1_loss(size3d_input, size3d_target, reduction='mean')

        heading_input = input['heading'][input['train_tag']]
        # compute heading loss
        heading_loss = compute_heading_loss(heading_input,
                                            target[mask_type],  ## NOTE
                                            target['heading_bin'],
                                            target['heading_res'])
        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss

        # iou loss
        pred_input = input['pred'][input['train_tag']]
        alpha = get_alpha(heading_input)
        size3d = size3d_input + mean_size_target
        bbox3d_lidar_target = extract_target_from_tensor(target['bbox3d_lidar'], target[mask_type])
        uv_target = extract_target_from_tensor(target['uv'], target[mask_type])
        p2_target = extract_target_from_tensor(target['p2'], target[mask_type])
        inv_r0_target = extract_target_from_tensor(target['inv_r0'], target[mask_type])
        c2v_target = extract_target_from_tensor(target['c2v'], target[mask_type])

        bbox3d_lidar = get_bbox3d(uv_target, merge_depth, size3d, alpha, p2_target)
        bbox3d_lidar = rect2lidar(bbox3d_lidar, inv_r0_target, c2v_target)
        iou = boxes_aligned_iou3d_gpu(bbox3d_lidar, bbox3d_lidar_target)
        threshold = 0.7
        in_range = (iou > threshold).float()
        iou_loss = F.binary_cross_entropy_with_logits(pred_input, in_range, reduction='mean')
        loss += iou_loss



        if depth_loss != depth_loss:
            print('badNAN----------------depth_loss', depth_loss)
            print(vis_depth_loss, att_depth_loss, ins_depth_loss, depth_mask_target.sum())
        if offset3d_loss != offset3d_loss:
            print('badNAN----------------offset3d_loss', offset3d_loss)
        if size3d_loss != size3d_loss:
            print('badNAN----------------size3d_loss', size3d_loss)
        if heading_loss != heading_loss:
            print('badNAN----------------heading_loss', heading_loss)
        if iou_loss != iou_loss:
            print('badNAN----------------iou_loss', iou_loss)

        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss
        self.stat['iou_loss'] = iou_loss

        return loss


### ======================  auxiliary functions  =======================

def xyz_from_rect_to_lidar(xyz, inv_r0, c2v):
    """
    Args:
        xyz: [M, 3]
        inv_r0: [M, 3, 3]
        c2v: [M, 3, 4]
    Returns:
    """
    tmp = (inv_r0 @ xyz.unsqueeze(-1)).permute(0, 2, 1)
    tmp = torch.cat([tmp, torch.ones_like(tmp[..., :1])], dim=-1)
    tmp = (tmp @ c2v.permute(0, 2, 1)).reshape(-1, 3)
    return tmp

def get_bbox3d(uv, depth, size3d, alpha, p2):
    """
    Args:
        uv: [M, 2]
        depth: [M, 1]
        size3d: [M, 3], hwl
        alpha: [M, 1]
        p2: [M, 3, 4]
    Returns:
        bbox3d_lidar: [M, 7]
    """
    fu, fv, cu, cv, tx, ty = p2[:, 0, 0], p2[:, 1, 1], p2[:, 0, 2], p2[:, 1, 2], p2[:, 0, 3], p2[:, 1, 3]
    fu, fv, cu, cv, tx, ty = fu.unsqueeze(1), fv.unsqueeze(1), cu.unsqueeze(1), cv.unsqueeze(1), tx.unsqueeze(1), ty.unsqueeze(1)
    u, v = uv[:, 0:1], uv[:, 1:]
    x = (u - cu) * depth / fu + tx
    y = (v - cv) * depth / fv + ty
    xyz = torch.stack([x, y, depth], dim=1).squeeze(2)
    lhw = size3d[:, [2, 0, 1]]
    ry = alpha2ry(alpha, x, cu, fu)
    bbox3d = torch.cat([xyz, lhw, ry], dim=1)
    return bbox3d

def rect2lidar(bbox3d, inv_r0, c2v):
    xyz, lhw, ry = bbox3d[:, 0:3], bbox3d[:, 3:6], bbox3d[:, 6:]
    xyz = xyz_from_rect_to_lidar(xyz, inv_r0, c2v)
    lwh = lhw[:, [2, 0, 1]]
    rz = -ry - torch.pi / 2
    return torch.cat([xyz, lwh, rz], dim=1)


def alpha2ry(alpha, x, cu, fu):
    # ry = alpha + np.arctan2(u - self.cu, self.fu)
    ry = alpha + torch.atan2(x - cu, fu)
    ry[ry > torch.pi] -= 2 * torch.pi
    ry[ry < -torch.pi] += 2 * torch.pi
    return ry


def get_alpha(heading):
    """
    Args:
        heading, [M, 24], 12 for bin, 12 for res
    Returns:
        alpha: [M, 1]
    """
    heading_bin, heading_res = heading[:, 0:12], heading[:, 12:24]
    cls = torch.argmax(heading_bin, dim=1)
    res = torch.gather(heading_res, dim=1, index=cls.unsqueeze(1)).squeeze(1)
    alpha = cls * 2 * torch.pi / 12 + res
    return alpha.unsqueeze(1)


def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]


# compute heading loss two stage style

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)  # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')

    return cls_loss + reg_loss


'''    

def compute_heading_loss(input, ind, mask, target_cls, target_reg):
    """
    Args:
        input: features, shaped in B * C * H * W
        ind: positions in feature maps, shaped in B * 50
        mask: tags for valid samples, shaped in B * 50
        target_cls: cls anns, shaped in B * 50 * 1
        target_reg: reg anns, shaped in B * 50 * 1
    Returns:
    """
    input = _transpose_and_gather_feat(input, ind)   # B * C * H * W ---> B * K * C
    input = input.view(-1, 24)  # B * K * C  ---> (B*K) * C
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    input_cls, target_cls = input_cls[mask], target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    input_reg, target_reg = input_reg[mask], target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss
'''

if __name__ == '__main__':
    input_cls = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))
