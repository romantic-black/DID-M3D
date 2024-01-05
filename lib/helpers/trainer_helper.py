import os
import tqdm

import torch
import torch.nn as nn
import numpy as np
import pdb
import datetime
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.save_helper import load_checkpoint
from lib.losses.loss_function import DIDLoss, Hierarchical_Task_Learning
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from torch.utils.tensorboard import SummaryWriter
from aug.weight.MGDA import MGDA

from tools import eval


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.label_dir = cfg['dataset']['label_dir']
        self.eval_cls = cfg['dataset']['eval_cls']
        self.writer = SummaryWriter(log_dir=os.path.join(self.cfg_train['log_dir'],
                                                         datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

        self.mgda = MGDA(3, self.model.backbone, self.model.feat_up, self.device)

        if self.cfg_train.get('resume_model', None):
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger,
                                         map_location=self.device)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.model = torch.nn.DataParallel(model).to(self.device)

    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss, max_epoch=self.cfg_train["HTL_stop"])
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' % (epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss, self.epoch)

            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' % (key[:-4], loss_weights[key])
            self.logger.info(log_str)

            ei_loss = self.train_one_epoch(loss_weights)
            # self.record_val_loss()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if ((self.epoch % self.cfg_train['save_frequency']) == 0
                    and self.epoch >= self.cfg_train['eval_start']):
                os.makedirs(self.cfg_train['log_dir'] + '/checkpoints', exist_ok=True)
                ckpt_name = os.path.join(self.cfg_train['log_dir'] + '/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)

            if ((self.epoch % self.cfg_train['eval_frequency']) == 0 and \
                    self.epoch >= self.cfg_train['eval_start']):
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                self.eval_one_epoch()

        return None

    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, targets, info) in enumerate(self.train_loader):
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)

                # train one batch
                criterion = DIDLoss(self.epoch)
                outputs = self.model(inputs, coord_ranges, calibs, targets)
                _, loss_terms = criterion(outputs, targets)

                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch
        return disp_dict

    def train_one_epoch(self, loss_weights=None):
        self.model.train()

        disp_dict = {}
        stat_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True)
        for batch_idx, (inputs, calibs, coord_ranges, targets, info) in enumerate(self.train_loader):
            if type(inputs) != dict:
                inputs = inputs.to(self.device)
            else:
                for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            criterion = DIDLoss(self.epoch)
            outputs = self.model(inputs, coord_ranges, calibs, targets)

            total_loss, loss_terms = criterion(outputs, targets)

            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    total_loss += loss_weights[key].detach() * loss_terms[key]
                total_loss.backward()
            else:
                loss_list = []
                loss_list.append(loss_terms['depth_loss'])
                loss_list.append(loss_terms['seg_loss'])
                loss_list.append(loss_terms['offset2d_loss']
                                 + loss_terms['size2d_loss']
                                 + loss_terms['offset3d_loss']
                                 + loss_terms['size3d_loss']
                                 + loss_terms['heading_loss'])

                sol = self.mgda.backward(loss_list, mgda_gn='loss+')

            self.optimizer.step()

            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0

                if isinstance(loss_terms[key], int):
                    stat_dict[key] += (loss_terms[key])
                else:
                    stat_dict[key] += (loss_terms[key]).detach()
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                # disp_dict[key] += loss_terms[key]
                if isinstance(loss_terms[key], int):
                    disp_dict[key] += (loss_terms[key])
                else:
                    disp_dict[key] += (loss_terms[key]).detach()
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' % (key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)
            progress_bar.update()
        progress_bar.close()
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
            self.writer.add_scalar(f'train/{key}', stat_dict[key], self.epoch)

        return stat_dict

    def record_val_loss(self):

        self.model.eval()
        stat_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Val Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, targets, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys(): targets[key] = targets[key].to(self.device)
                # the outputs of centernet
                criterion = DIDLoss(self.epoch)
                outputs = self.model(inputs, coord_ranges, calibs, targets, K=50)
                total_loss, loss_terms = criterion(outputs, targets)
                for key in loss_terms.keys():
                    if key not in stat_dict.keys():
                        stat_dict[key] = 0

                    if isinstance(loss_terms[key], int):
                        stat_dict[key] += (loss_terms[key])
                    else:
                        stat_dict[key] += (loss_terms[key]).detach()
                trained_batch = batch_idx + 1
                progress_bar.update()
            progress_bar.close()
            for key in stat_dict.keys():
                stat_dict[key] /= trained_batch
                self.writer.add_scalar(f'val/{key}', stat_dict[key], self.epoch)
        return stat_dict

    def eval_one_epoch(self):
        self.model.eval()

        results = {}
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)

                # the outputs of centernet
                outputs = self.model(inputs, coord_ranges, calibs, K=50, mode='val')

                dets = extract_dets_from_outputs(outputs, K=50)
                dets = dets.detach().cpu().numpy()

                # get corresponding calibs & transform tensor to numpy
                calibs = [self.test_loader.dataset.get_calib(index) for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.test_loader.dataset.cls_mean_size
                dets = decode_detections(dets=dets,
                                         info=info,
                                         calibs=calibs,
                                         cls_mean_size=cls_mean_size,
                                         threshold=self.cfg_test['threshold'])
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
        # self.save_results(results)
        out_dir = os.path.join(self.cfg_train['out_dir'], 'EPOCH_' + str(self.epoch))
        self.save_results(results, out_dir)
        eval.eval_from_scrach(
            self.label_dir,
            os.path.join(out_dir, 'data'),
            self.eval_cls,
            ap_mode=40)

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()
