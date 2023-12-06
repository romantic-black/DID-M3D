import unittest
import torch
import numpy as np
from lib.losses.loss_function import get_alpha, xyz_from_rect_to_lidar
from tools.box_util import rect2lidar, rect2lidar_no_calib, xyz_from_rect_to_lidar_np
from tools.dataset_util import Dataset
from aug.iou3d_nms.iou3d_nms_utils import boxes_aligned_iou3d_gpu

class TestLossFunction(unittest.TestCase):

    def test_get_alpha(self):
        B = 3
        K = 50
        heading = torch.randn(B * K, 24)
        alpha = get_alpha(heading)
        print(alpha.shape)
        print(alpha)

    def test_get_bbox3d_lidar(self):
        dataset = Dataset("train", r"/mnt/e/DataSet/kitti")
        for idx in range(100):
            calib = dataset.get_calib(idx)
            bbox3d_gt, _, _ = dataset.get_bbox(idx)
            bbox3d_dt, _, _ = dataset.get_bbox(idx)
            bbox3d_dt[:, :3] += np.random.randn(bbox3d_dt.shape[0], 3)

            p2 = calib.P2
            inv_r0 = np.linalg.inv(calib.R0)
            c2v = calib.C2V

            bbox3d_dt = bbox3d_dt.astype(np.float32)
            bbox3d_gt = bbox3d_gt.astype(np.float32)

            bbox3d_gt_1 = rect2lidar(bbox3d_gt, calib)
            bbox3d_dt_1 = rect2lidar(bbox3d_dt, calib)
            bbox3d_gt_2 = rect2lidar_no_calib(bbox3d_gt, inv_r0, c2v)
            bbox3d_dt_2 = rect2lidar_no_calib(bbox3d_dt, inv_r0, c2v)

            iou3d_1 = boxes_aligned_iou3d_gpu(torch.from_numpy(bbox3d_gt_1).cuda(), torch.from_numpy(bbox3d_dt_1).cuda())
            iou3d_2 = boxes_aligned_iou3d_gpu(torch.from_numpy(bbox3d_gt_2).cuda(), torch.from_numpy(bbox3d_dt_2).cuda())

            print(iou3d_1.cpu().numpy(), iou3d_2.cpu().numpy())
            self.assertTrue(np.allclose(iou3d_1.cpu().numpy(), iou3d_2.cpu().numpy(), atol=1e-3))

    def test_xyz_from_rect_to_lidar(self):
        dataset = Dataset("train", r"/mnt/e/DataSet/kitti")
        for idx in range(100):
            calib = dataset.get_calib(idx)
            p2 = calib.P2
            inv_r0 = np.linalg.inv(calib.R0)
            c2v = calib.C2V
            p2 = np.tile(p2.reshape(1, 3, 4), (100, 1, 1))
            inv_r0 = np.tile(inv_r0.reshape(1, 3, 3), (100, 1, 1))
            c2v = np.tile(c2v.reshape(1, 3, 4), (100, 1, 1))
            xyz = np.random.randn(100, 3).astype(np.float32)

            xyz_1 = xyz_from_rect_to_lidar_np(xyz, inv_r0, c2v)
            xyz = torch.from_numpy(xyz)
            inv_r0 = torch.from_numpy(inv_r0)
            c2v = torch.from_numpy(c2v)
            xyz_2 = xyz_from_rect_to_lidar(xyz, inv_r0, c2v)

            xyz_2 = xyz_2.numpy()
            print(xyz_1, xyz_2)
            self.assertTrue(np.allclose(xyz_1, xyz_2, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
