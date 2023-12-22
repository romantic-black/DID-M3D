import unittest
import random
from tools.sample_util import *
from tools.dataset_util import Dataset
from tools.visualize_util import show_o3d

class MyTestCase(unittest.TestCase):
    def test_ry(self):
        np.random.seed(6)
        random.seed(0)
        random_flip = 0
        database = SampleDatabase("/mnt/e/DataSet/kitti/kitti_inst_database/", random_flip=random_flip)
        dataset = Dataset("train", r"/mnt/e/DataSet/kitti")

        idx = 2131

        calib_ = dataset.get_calib(idx)
        image, depth = dataset.get_image_with_depth(idx, use_penet=False)
        ground, non_ground = dataset.get_lidar_with_ground(idx, fov=True)
        plane_ = dataset.get_plane(idx)
        grid = dataset.get_grid(idx)

        sample = database.samples_from_database(1)[0]

        x_ = np.arange(-15, 15, 5)
        z_ = np.arange(20, 60, 5)
        x_, z_ = np.meshgrid(x_, z_)
        y_ = np.zeros_like(x_)
        xyz_ = np.stack([x_, y_ , z_], axis=-1).reshape(-1, 3)
        samples = [sample for _ in range(xyz_.shape[0])]
        samples, bbox3d_ = database.xyz_to_bbox3d(samples, xyz_, calib_, random_flip=random_flip)

        # bbox3d_, flag = database.sample_put_on_plane(bbox3d_, ground)
        #samples = [samples[i] for i in range(len(samples)) if flag[i]]
        #bbox3d_ = bbox3d_[flag]

        samples = [Sample(samples[i], bbox3d_[i], calib_, database) for i in range(len(samples))]
        print(samples[0].label.pos)
        cord_0 = [sample.get_points()[0] for sample in samples]
        cord_1 = [sample.get_points(True)[0] for sample in samples]
        rgb_0 = [np.zeros_like(cord) + np.array([0, 0, 255]) for cord in cord_0]
        rgb_1 = [np.zeros_like(cord) + np.array([0, 255, 0]) for cord in cord_1]

        cord = np.concatenate([*cord_0, *cord_1], axis=0)
        rgb = np.concatenate([*rgb_0, *rgb_1], axis=0)

        show_o3d(cord, rgb, bbox3d_)

if __name__ == '__main__':
    unittest.main()
