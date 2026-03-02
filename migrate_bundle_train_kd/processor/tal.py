#!/usr/bin/env python
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torchlight import str2bool
from .processor import Processor


def _binary_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.astype(bool).tolist()):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _segment_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return 0.0 if union <= 0 else float(inter) / float(union)


class TAL_Processor(Processor):
    def load_model(self):
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args).to(self.output_device)
        self.work_dir = getattr(self.arg, "work_dir", "./work_dir/tal")
        self.best_seg_f1 = -1.0
        self.best_epoch = -1

    def load_optimizer(self):
        base_lr = float(self.arg.base_lr)
        wd = float(self.arg.weight_decay)
        backbone_lr_mult = float(getattr(self.arg, "backbone_lr_mult", 0.1))

        if hasattr(self.model, "get_param_groups"):
            param_groups = self.model.get_param_groups(base_lr=base_lr, backbone_lr_mult=backbone_lr_mult)
            for g in param_groups:
                g.setdefault("weight_decay", wd)
        else:
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": wd}]

        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(param_groups, lr=base_lr, momentum=0.9, nesterov=self.arg.nesterov, weight_decay=wd)
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(param_groups, lr=base_lr, weight_decay=wd)
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)
        else:
            raise ValueError("Unknown optimizer")

    def adjust_lr(self):
        epoch = int(self.meta_info.get("epoch", 0))
        steps = self.arg.step if isinstance(self.arg.step, (list, tuple)) else [self.arg.step]
        steps = [int(s) for s in steps if s is not None]
        num_decays = sum(1 for s in steps if epoch >= s)
        lr = float(self.arg.base_lr) * (0.1 ** num_decays)

        warmup_epochs = int(getattr(self.arg, "warm_up_epoch", 0) or 0)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr = float(self.arg.base_lr) * (epoch + 1) / float(warmup_epochs)

        for g in self.optimizer.param_groups:
            base_g_lr = g.get("lr", lr)
            if epoch == 0:
                g.setdefault("_initial_lr", base_g_lr)
            scale = g["_initial_lr"] / float(self.arg.base_lr) if float(self.arg.base_lr) > 0 else 1.0
            g["lr"] = lr * scale
        self.lr = lr

    @staticmethod
    def _resize_targets(target: torch.Tensor, t_out: int) -> torch.Tensor:
        if target.dim() == 2:
            target = target.unsqueeze(1)
            out = F.interpolate(target, size=t_out, mode="nearest").squeeze(1)
            return out
        return F.interpolate(target, size=t_out, mode="nearest")

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        cls_logits = outputs["cls_logits"]
        act_logits = outputs["actionness_logits"]
        t_out = cls_logits.size(-1)

        cls_target = self._resize_targets(target["class_targets"].float().to(self.output_device, non_blocking=True), t_out)
        act_target = self._resize_targets(target["actionness"].float().to(self.output_device, non_blocking=True), t_out)

        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, cls_target)
        loss_act = F.binary_cross_entropy_with_logits(act_logits, act_target)
        loss = float(self.arg.cls_loss_weight) * loss_cls + float(self.arg.actionness_loss_weight) * loss_act
        return loss, loss_cls, loss_act, cls_target, act_target

    def train(self):
        epoch = int(self.meta_info.get("epoch", 0))
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_obj, "set_epoch"):
            model_obj.set_epoch(epoch)

        self.adjust_lr()
        self.model.train()
        losses = []

        for data, target, _sid in self.data_loader["train"]:
            data = data.float().to(self.output_device, non_blocking=True)
            outputs = self.model(data)
            loss, loss_cls, loss_act, _cls_target, _act_target = self._compute_loss(outputs, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            losses.append(float(loss.item()))
            self.iter_info["loss"] = float(loss.item())
            self.iter_info["loss_cls"] = float(loss_cls.item())
            self.iter_info["loss_act"] = float(loss_act.item())
            self.iter_info["lr"] = float(self.lr)
            self.show_iter_info()
            self.meta_info["iter"] += 1

        self.epoch_info["mean_loss"] = float(np.mean(losses)) if losses else 0.0
        self.show_epoch_info()

    def test(self):
        self.model.eval()
        losses = []
        frame_tp = frame_fp = frame_fn = 0
        seg_tp = seg_fp = seg_fn = 0

        with torch.no_grad():
            for data, target, _sid in self.data_loader["test"]:
                data = data.float().to(self.output_device, non_blocking=True)
                outputs = self.model(data)
                loss, _loss_cls, _loss_act, cls_target, _act_target = self._compute_loss(outputs, target)
                losses.append(float(loss.item()))

                scores = outputs["scores"].detach().cpu().numpy()
                cls_gt = cls_target.detach().cpu().numpy()
                pred_bin = (scores >= float(self.arg.frame_threshold)).astype(np.int32)
                gt_bin = (cls_gt >= 0.5).astype(np.int32)

                frame_tp += int(np.logical_and(pred_bin == 1, gt_bin == 1).sum())
                frame_fp += int(np.logical_and(pred_bin == 1, gt_bin == 0).sum())
                frame_fn += int(np.logical_and(pred_bin == 0, gt_bin == 1).sum())

                for b in range(pred_bin.shape[0]):
                    for c in range(pred_bin.shape[1]):
                        pred_segments = _binary_runs(pred_bin[b, c])
                        gt_segments = _binary_runs(gt_bin[b, c])
                        matched = [False] * len(gt_segments)

                        for pred_seg in pred_segments:
                            best_idx = -1
                            best_iou = 0.0
                            for gt_idx, gt_seg in enumerate(gt_segments):
                                if matched[gt_idx]:
                                    continue
                                iou = _segment_iou(pred_seg, gt_seg)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_idx = gt_idx
                            if best_idx >= 0 and best_iou >= float(self.arg.iou_threshold):
                                matched[best_idx] = True
                                seg_tp += 1
                            else:
                                seg_fp += 1

                        seg_fn += sum(1 for x in matched if not x)

        frame_prec = frame_tp / max(1, frame_tp + frame_fp)
        frame_rec = frame_tp / max(1, frame_tp + frame_fn)
        frame_f1 = 2 * frame_prec * frame_rec / max(1e-8, frame_prec + frame_rec)

        seg_prec = seg_tp / max(1, seg_tp + seg_fp)
        seg_rec = seg_tp / max(1, seg_tp + seg_fn)
        seg_f1 = 2 * seg_prec * seg_rec / max(1e-8, seg_prec + seg_rec)

        self.epoch_info["mean_loss"] = float(np.mean(losses)) if losses else 0.0
        self.epoch_info["frame_f1"] = frame_f1
        self.epoch_info[f"seg_f1@{self.arg.iou_threshold:.2f}"] = seg_f1
        self.show_epoch_info()

        if seg_f1 > self.best_seg_f1:
            self.best_seg_f1 = seg_f1
            self.best_epoch = int(self.meta_info.get("epoch", 0))
            self.io.save_model(self.model, "best_model.pt")
            self.io.print_log(f"[BEST] epoch {self.best_epoch} | seg_f1={self.best_seg_f1:.4f}")

    @staticmethod
    def get_parser(add_help=False):
        parent = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(add_help=add_help, parents=[parent], description="Temporal Action Localization")
        parser.add_argument("--base_lr", type=float, default=0.001)
        parser.add_argument("--step", type=int, default=[], nargs="+")
        parser.add_argument("--warm_up_epoch", type=int, default=0)
        parser.add_argument("--optimizer", default="AdamW")
        parser.add_argument("--nesterov", type=str2bool, default=True)
        parser.add_argument("--weight_decay", type=float, default=0.0005)
        parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
        parser.add_argument("--cls_loss_weight", type=float, default=1.0)
        parser.add_argument("--actionness_loss_weight", type=float, default=1.0)
        parser.add_argument("--frame_threshold", type=float, default=0.5)
        parser.add_argument("--iou_threshold", type=float, default=0.5)
        return parser
