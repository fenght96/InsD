# Ultralytics YOLO 🚀, GPL-3.0 license
from copy import copy
import os
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)
from ultralytics.nn.tasks import DistillationBaseModel
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer, check_amp
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, colorstr, IterableSimpleNamespace, yaml_load, callbacks
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel

cls_num = 10 # dota 8 dior 10 uav 6
# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):
    def __init__(self, cfg, overrides=None):
        super().__init__(cfg, overrides)
        del self.model
        self.tmodel = self.args.model
        self.smodel = self.args.model
        self.id = 'incr'
    
    def _setup_train(self, rank, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.tmodel = self.tmodel.to(self.device)
        self.smodel = self.smodel.to(self.device)
        self.set_model_attributes()
        # Check AMP
        self.amp = torch.tensor(True).to(self.device)
        if RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.smodel), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.tmodel = DDP(self.tmodel, device_ids=[rank])
            self.smodel = DDP(self.smodel, device_ids=[rank])
        # Check imgsz
        gs = max(int(self.smodel.stride.max() if hasattr(self.smodel, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
        if self.batch_size == -1:
            if RANK == -1:  # single-GPU only, estimate best batch size
                self.batch_size = check_train_batch_size(self.smodel, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        self.optimizer = self.build_optimizer(model=self.smodel,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # dataloaders
        batch_size = self.batch_size // world_size if world_size > 1 else self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=rank, mode='train')
        if rank in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.smodel)
            # self.ema = ModelEMA(self.smodel)
            if self.args.plots and not self.args.v5loader:
                self.plot_training_labels()
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

    def get_dataloader(self, dataset_path, batch_size, mode='train', rank=0):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.smodel).stride.max() if self.smodel else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 augment=mode == 'train',
                                 cache=self.args.cache,
                                 pad=0 if mode == 'train' else 0.5,
                                 rect=self.args.rect or mode == 'val',
                                 rank=rank,
                                 workers=self.args.workers,
                                 close_mosaic=self.args.close_mosaic != 0,
                                 prefix=colorstr(f'{mode}: '),
                                 shuffle=mode == 'train',
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, rank=rank, mode=mode,
                             rect=mode == 'val', names=self.data['names'])[0]

    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        # nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.smodel.nc = self.data['nc']  # attach number of classes to model
        self.smodel.names = self.data['names']  # attach class names to model
        self.smodel.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)

        self._setup_train(rank, world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.tmodel.eval()
            self.smodel.train()
            if rank != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

            if rank in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    tpreds = self.tmodel(batch['img'])
                    tfpreds = self.tmodel(torch.flip(batch['img'], [3]))
                    # import pdb; pdb.set_trace()
                    preds = self.smodel(batch['img'])
                    self.loss, self.loss_items = self.criterion(preds, batch, tpreds, tfpreds)
                    if rank != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if rank in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if rank in (-1, 0):

                # Validation
                self.ema.update_attr(self.smodel, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if rank in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
    
    def save_model(self):
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.smodel)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        del ckpt


    def get_model(self, cfg=None, weights=None, verbose=True):
        new_model = DistillationBaseModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        old_model = DistillationBaseModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            new_model.load(weights)
            old_model.load(weights)
        return new_model, old_model

    def setup_model(self):
        """
        load/create/download model for any task.
        """
        if isinstance(self.smodel, torch.nn.Module) and isinstance(self.smodel, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        tmodel, weights = self.smodel, None
        ckpt = None
        if str(tmodel).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(tmodel)
            cfg = ckpt['model'].yaml
        else:
            cfg = tmodel

        self.tmodel, self.smodel = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'dis_loss'
        return v8.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.smodel.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.smodel)

    def resume_training(self, ckpt):
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without --resume, i.e. 'yolo task=... mode=train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.smodel} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            LOGGER.info('Closing dataloader mosaic')
            if hasattr(self.train_loader.dataset, 'mosaic'):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, 'close_mosaic'):
                self.train_loader.dataset.close_mosaic(hyp=self.args)


    def criterion(self, preds, batch, tpreds=None, tfpreds=None):
        if not hasattr(self, 'compute_loss'):
            # TODO dis loss
            self.compute_loss = HLoss(de_parallel(self.smodel))
        return self.compute_loss(preds, batch, tpreds, tfpreds)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg')

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png

    def plot_training_labels(self):
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir)


# Criterion class for computing training losses
class Loss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.cls_num = cls_num
        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, tpreds, tfpreds):
        if tpreds is None:
            return self.nodisloss(preds, batch)
        else:
            return self.dis_loss(preds, batch, tpreds, tfpreds)

    def nodisloss(self, preds, batch):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        # preds = preds[0]
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        # loss[3] = 0

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    def dis_loss(self, preds, batch, tpreds, tfpreds):
        mask_ids = [i for i in range(len(batch['cls']))]
        
        for i in range(len(batch['cls'])):
            if batch['cls'][i] < self.cls_num:
                mask_ids.remove(i)
        if len(mask_ids) > 0:
            batch['cls'] = torch.cat([batch['cls'][i].unsqueeze(0) for i in mask_ids])
            batch['bboxes'] = torch.cat([batch['bboxes'][i].unsqueeze(0) for i in mask_ids])
            batch['batch_idx'] = torch.cat([batch['batch_idx'][i].unsqueeze(0) for i in mask_ids])
        else:
            batch['cls'] = torch.tensor([]).to(self.device)
            batch['bboxes'] = torch.tensor([]).to(self.device)
            batch['batch_idx'] = torch.tensor([]).to(self.device)
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        
        smfeats = preds[1]
        tmfeats = tpreds[1]
        feats = preds[0]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        # dis loss
        for tm, sm in zip(tmfeats, smfeats):
            if tm is None:
                continue
            h = tm.shape[-1]
            m_gt = gt_bboxes * h 
            bs = m_gt.shape[0]
            for b in range(bs):
                mask = torch.ones(tm.shape[-2:]).to(self.device)
                for x,y,w,h in m_gt[b]:
                    x = int(x-w/2)
                    y = int(y-h/2)
                    w = int(w)
                    h = int(h)
                    mask[x:x+w, y:y+h] = 0

                loss[3] += torch.nn.functional.mse_loss(sm[b]*mask.unsqueeze(0), tm[b].detach()*mask.unsqueeze(0))#((tm[b] -sm[b])**2*mask.unsqueeze(0)).mean()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.dis

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

class HLoss(Loss):
    def __init__(self, model):
        super().__init__(model)
        self.args = model.args

    def postprocess(self, preds, batch, imgsz):
        # preds = ops.non_max_suppression(preds,
        #                                 0.5,
        #                                 self.args.iou,
        #                                 agnostic=self.args.agnostic_nms,
        #                                 max_det=self.args.max_det,
        #                                 classes=self.args.classes)
        # preds [x1,y1,x2,y2,score,cls]
        device = batch['cls'].device
        for i, pred in enumerate(preds):
            # pred[:,[0,2]] /= imgsz[0]
            # pred[:,[1,3]] /= imgsz[1]
            # pred[:,0], pred[:,1], pred[:,2], pred[:,3] = (pred[:,0] + pred[:,2]) / 2, (pred[:,1] + pred[:,3]) / 2, (pred[:,2] - pred[:,0]), (pred[:,3] - pred[:,1])
            # pred[:,0].clamp_(0, 1.)
            # pred[:,1].clamp_(0, 1.)
            # pred[:,2].clamp_(0, 1.)
            # pred[:,3].clamp_(0, 1.)
            pred = pred.detach()
            if pred.shape[0] == 0:
                continue
            batch['cls'] = torch.cat([batch['cls'], pred[:,-1].unsqueeze(1).to(device)],dim=0)
            batch['bboxes'] = torch.cat([batch['bboxes'], pred[:,:4].to(device)],dim=0)

            nums = pred.shape[0]
            batch['batch_idx'] = torch.cat([batch['batch_idx'], torch.tensor([i for _ in range(nums)]).to(device)],dim=0)
        # normlize
        return batch

    def mixprocess(self, preds, tpreds, imgsz):
        preds = ops.non_max_suppression(preds,
                                        0.5,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)
        tpreds = ops.non_max_suppression(tpreds,
                                        0.5,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)
        preds = self.bbox_norm(preds, imgsz)
        tpreds = self.bbox_norm(tpreds, imgsz, f=True)
        res = []
        
        for pred, tpred in zip(preds, tpreds):
            keepnums = []
            for i in range(tpred.shape[0]):
                p = tpred[i]
                j = 0 
                while (j < pred.shape[0]):
                # for j in range(pred.shape[0]):
                    t = pred[j]
                    if t[-1] == p[-1]:
                        iou = self.cptiou(p,t)
                        if iou >= 0.5:
                            break
                    j += 1
                if j == pred.shape[0]:
                    keepnums.append(i)
            if len(keepnums) == 0:
                res.append(pred)
            else:
                tmp = torch.cat([tpred[i].unsqueeze(0) for i in keepnums], dim=0).to(pred.device)
                res.append(torch.cat([pred, tmp], dim=0).to(pred.device))
        return res
    
    def cptiou(self, p,t):
        area = p[2]*p[3] + t[2]*t[3]
        px1,py1,px2,py2 = p[0]-p[2]/2,p[1]-p[3]/2,p[0]+p[2]/2,p[1]+p[3]/2
        tx1,ty1,tx2,ty2 = t[0]-t[2]/2,t[1]-t[3]/2,t[0]+t[2]/2,t[1]+t[3]/2
        x1,y1 = max(px1,tx1), max(py1, ty1)
        x2,y2 = min(px2,tx2), min(py2, ty2)
        x1.clamp_(0, 1.)
        y1.clamp_(0, 1.)
        x2.clamp_(0, 1.)
        y2.clamp_(0, 1.)
        if x1 >= x2 or y1 >= y2:
            iou = 0
        else:
            iou = (y2-y1)*(x2-x1)/(area-(y2-y1)*(x2-x1))
        return iou

    def bbox_norm(self, preds, imgsz, f = False):
        for i, pred in enumerate(preds):
            pred[:,[0,2]] /= imgsz[0]
            pred[:,[1,3]] /= imgsz[1]
            pred[:,0], pred[:,1], pred[:,2], pred[:,3] = (pred[:,0] + pred[:,2]) / 2, (pred[:,1] + pred[:,3]) / 2, (pred[:,2] - pred[:,0]), (pred[:,3] - pred[:,1])
            
            pred[:,0].clamp_(0, 1.)
            pred[:,1].clamp_(0, 1.)
            pred[:,2].clamp_(0, 1.)
            pred[:,3].clamp_(0, 1.)
            if f:
                pred[:,0] = 1-pred[:,0]
        return preds

    def dis_loss(self, preds, batch, tpreds, tfpreds):
        mask_ids = [i for i in range(len(batch['cls']))]
        
        for i in range(len(batch['cls'])):
            if batch['cls'][i] < self.cls_num:
                mask_ids.remove(i)
        if len(mask_ids) > 0:
            batch['cls'] = torch.cat([batch['cls'][i].unsqueeze(0) for i in mask_ids])
            batch['bboxes'] = torch.cat([batch['bboxes'][i].unsqueeze(0) for i in mask_ids])
            batch['batch_idx'] = torch.cat([batch['batch_idx'][i].unsqueeze(0) for i in mask_ids])
        else:
            batch['cls'] = torch.tensor([]).to(self.device)
            batch['bboxes'] = torch.tensor([]).to(self.device)
            batch['batch_idx'] = torch.tensor([]).to(self.device)
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        
        smfeats = preds[1]
        tmfeats = tpreds[1]
        
        feats = preds[0]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        tfpreds = tfpreds[0]
        # TODO nms iou
        mtpreds = self.mixprocess(tpreds[0], tfpreds, imgsz)
        allbatch = self.postprocess(mtpreds, batch, imgsz)
        
        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        _, newgt_bboxes = targets.split((1, 4), 2)  # cls, xyxy


        targets = torch.cat((allbatch['batch_idx'].view(-1, 1), allbatch['cls'].view(-1, 1), allbatch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        # dis loss
        for tm, sm in zip(tmfeats, smfeats):
            if tm is None:
                continue
            h = tm.shape[-1]
            m_gt = gt_bboxes * h 
            bs = m_gt.shape[0]
            for b in range(bs):
                mask = torch.ones(tm.shape[-2:]).to(self.device)
                areas = tm.shape[-1] * tm.shape[-2]
                m_areas = 0
                for ib, (x,y,w,h) in enumerate(m_gt[b]):
                    x = int(x-w/2)
                    y = int(y-h/2)
                    w = int(w)
                    h = int(h)
                    if gt_labels[b][ib] >= self.cls_num:
                        x = max(0,x-1)
                        y = max(0,y-1)
                        mask[x:x+w, y:y+h] = 0
                        m_areas += w *h
                    else:
                        mask[x:x+w, y:y+h] += 1.0 # 0，0.25， 0.5， 0.75， 1.0
                area_factor = areas - m_areas
                if area_factor == 0:
                    area_factor = 1
                loss[3] += torch.nn.functional.mse_loss(sm[b]*mask.unsqueeze(0), tm[b].detach()*mask.unsqueeze(0))*areas/area_factor#((tm[b] -sm[b])**2*mask.unsqueeze(0)).mean()
                # loss[3] += torch.sum((sm[b] - tm[b].detach())*mask.unsqueeze(0)**2)/area_factor#((tm[b] -sm[b])**2*mask.unsqueeze(0)).mean()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.dis

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    

def train(cfg, use_python=False):
    cfg_dict = yaml_load(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, str) and v.lower() == 'none':
            cfg_dict[k] = None
    cfg_keys = cfg_dict.keys()
    cfg = IterableSimpleNamespace(**cfg_dict)
    model = cfg.model or 'yolov8s.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(cfg=cfg, overrides=args)
        trainer.train()
    print('gamma = 2.0')

if __name__ == '__main__':
    train(cfg='cfg/dior_default_inc.yaml')
