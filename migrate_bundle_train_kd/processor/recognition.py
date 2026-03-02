#!/usr/bin/env python
# pylint: disable=W0201
import os
import csv
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchlight import str2bool
from torchlight import import_class
from .processor import Processor
from sklearn.metrics import confusion_matrix



# ---------------- Utilities ----------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _append_csv(csv_path: str, headers, rowdict):
    write_header = (not os.path.exists(csv_path))
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            w.writeheader()
        w.writerow(rowdict)


def soft_ce_kd(student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               T: float = 4.0) -> torch.Tensor:
    """
    KL(student||teacher) with temperature T (Hinton).
    - student 用 log_softmax
    - teacher 用 softmax（冻结，不反传）
    - 返回 batchmean KL * T^2
    """
    with torch.no_grad():
        t = F.softmax(teacher_logits / T, dim=1)
    s = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)


# --------------- Processor -----------------
class REC_Processor(Processor):
    """
    Skeleton-based Action Recognition (single-epoch train; epoch loop in Processor.start()).
    """

    # --------- I/O helpers for metrics/logits ---------
    def _metrics_csv_path(self):
        work = getattr(self, 'work_dir', None) or getattr(self.arg, 'work_dir', './work_dir/run')
        _ensure_dir(work)
        return os.path.join(work, 'metrics.csv')

    def _log_epoch_metric(self, split, **kvs):
        """
        记录每个 epoch 的指标到 work_dir/metrics.csv
        split: 'train' or 'val'
        kvs  : 任意 key=value，例如 loss, top1, top5
        """
        csv_path = self._metrics_csv_path()
        epoch = int(self.meta_info.get('epoch', 0))
        row = {'time': datetime.now().isoformat(timespec='seconds'),
               'epoch': epoch, 'split': split}
        row.update({k: float(v) for k, v in kvs.items()})
        headers = ['time', 'epoch', 'split'] + sorted([k for k in row.keys() if k not in ('time', 'epoch', 'split')])
        _append_csv(csv_path, headers, row)

    def _save_eval_arrays(self, epoch, logits_np, labels_np, names=None):
        """
        保存验证阶段的预测/标签（用于画混淆矩阵）
        """
        work = getattr(self, 'work_dir', None) or getattr(self.arg, 'work_dir', './work_dir/run')
        outdir = os.path.join(work, 'eval_arrays')
        _ensure_dir(outdir)
        np.save(os.path.join(outdir, f'epoch{epoch:03d}_logits.npy'), logits_np)
        np.save(os.path.join(outdir, f'epoch{epoch:03d}_labels.npy'), labels_np)
        if names is not None:
            with open(os.path.join(outdir, f'epoch{epoch:03d}_names.json'), 'w') as f:
                json.dump(list(map(str, names)), f)
        # ===== 新增：confusion matrix & per-class acc =====
        preds = logits_np.argmax(axis=1)
        num_class = int(logits_np.shape[1])

        cm = confusion_matrix(labels_np, preds, labels=list(range(num_class)))
        np.save(os.path.join(outdir, f'epoch{epoch:03d}_confusion.npy'), cm)

        per_class_acc = {}
        for i in range(num_class):
            denom = cm[i].sum()
            acc = float(cm[i, i] / denom) if denom > 0 else 0.0
            per_class_acc[str(i)] = {
                "acc": acc,
                "support": int(denom)
            }

        with open(os.path.join(outdir, f'epoch{epoch:03d}_perclass.json'), 'w') as f:
            json.dump(per_class_acc, f, indent=2)


    # ------------------- Model / Optimizer -------------------
    def load_model(self):
        # device
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device

        # build model
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args).to(self.output_device)

        # CE with optional label smoothing
        ls = float(getattr(self.arg, 'label_smoothing', 0.0))
        self.ce_loss_train = nn.CrossEntropyLoss(label_smoothing=ls)
        self.ce_loss_eval  = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.ce_loss = self.ce_loss_train  # 兼容旧调用

        # KD flags（默认关闭，只有命令行/YAML显式开启才启用）
        self.use_kd = bool(getattr(self.arg, 'use_kd', False))
        self.kd_alpha = float(getattr(self.arg, 'kd_alpha', 0.0))
        self.kd_temperature = float(getattr(self.arg, 'kd_temperature', 4.0))
        self.teacher = None
        if self.use_kd and getattr(self.arg, 'kd_teacher_model', None) and getattr(self.arg, 'kd_teacher_weights', None):
            TeacherCls = import_class(self.arg.kd_teacher_model)
            self.teacher = TeacherCls(**getattr(self.arg, 'kd_teacher_args', {})).to(self.output_device)
            sd = torch.load(self.arg.kd_teacher_weights, map_location='cpu')
            sd = sd.get('state_dict', sd)
            self.teacher.load_state_dict(sd, strict=False)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()
            self.io.print_log('[KD] teacher loaded & frozen.')
        else:
            self.use_kd = False
            self.kd_alpha = 0.0  # 确保不参与 loss

        # best trackers
        self.best_top1 = -1.0
        self.best_loss = float('inf')
        self.best_epoch = -1

        # ensure work_dir
        self.work_dir = getattr(self.arg, 'work_dir', None) or './work_dir/run'
        _ensure_dir(self.work_dir)

    def load_optimizer(self):
        import torch.optim as optim

        # 统计参数量
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        tot = sum(p.numel() for p in self.model.parameters()) / 1e6
        trn = sum(p.numel() for p in trainable) / 1e6
        self.io.print_log(f'[opt] params total {tot:.2f}M | trainable {trn:.2f}M')

        # ====== param groups: head lr + backbone smaller lr ======
        base_lr = float(self.arg.base_lr)
        wd = float(self.arg.weight_decay)

        # 你可以在 yaml/arg 里加这个参数，默认 0.1 => backbone_lr = 1e-5 when base_lr=1e-4
        backbone_lr_mult = float(getattr(self.arg, "backbone_lr_mult", 0.1))
        backbone_lr = base_lr * backbone_lr_mult

        # 如果模型实现了 get_param_groups，就直接用（我给你的 routeA_finetune 文件包含这个接口）
        if hasattr(self.model, "get_param_groups"):
            param_groups = self.model.get_param_groups(base_lr=base_lr, backbone_lr_mult=backbone_lr_mult)
            # 顺带打印一下每组参数量
            for i, g in enumerate(param_groups):
                nparams = sum(p.numel() for p in g["params"]) / 1e6
                self.io.print_log(f'[opt] group{i}: lr={g["lr"]:.2e}, params={nparams:.2f}M')
        else:
            # fallback：按名字分组（假设 backbone 参数名前缀是 "backbone."）
            head_params, bb_params = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("backbone."):
                    bb_params.append(p)
                else:
                    head_params.append(p)

            param_groups = []
            if len(head_params) > 0:
                param_groups.append({"params": head_params, "lr": base_lr, "weight_decay": wd})
            if len(bb_params) > 0:
                param_groups.append({"params": bb_params, "lr": backbone_lr, "weight_decay": wd})

            self.io.print_log(f'[opt] head lr={base_lr:.2e} | backbone lr={backbone_lr:.2e} (mult={backbone_lr_mult})')
            self.io.print_log(f'[opt] head params={sum(p.numel() for p in head_params)/1e6:.2f}M | backbone params={sum(p.numel() for p in bb_params)/1e6:.2f}M')

        # 不再把 backbone 进 optimizer 当“异常”，因为 finetune 时这是正常的
        if any(k.get("lr", None) == backbone_lr for k in param_groups) and backbone_lr_mult < 1.0:
            self.io.print_log("[opt] backbone finetune is ON (smaller lr group).")

        # ====== build optimizer ======
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                param_groups,
                lr=base_lr,  # 这里是默认值，实际每个 group 用自己的 lr
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=wd
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                param_groups,
                lr=base_lr,
                weight_decay=wd
            )
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=wd
            )
        else:
            raise ValueError('Unknown optimizer')

    def adjust_lr(self):
        """
        多步下降学习率：
        - base_lr 按 step 阶梯衰减（每命中一个 step，*0.1）
        - 支持 warm_up_epoch：线性 warmup 到 base_lr
        """
        epoch = int(self.meta_info.get('epoch', 0))
        steps = self.arg.step if isinstance(self.arg.step, (list, tuple)) else [self.arg.step]
        steps = [int(s) for s in steps if s is not None]
        num_decays = sum(1 for s in steps if epoch >= s)
        lr = float(self.arg.base_lr) * (0.1 ** num_decays)

        warmup_epochs = int(getattr(self.arg, 'warm_up_epoch', 0) or 0)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr = float(self.arg.base_lr) * (epoch + 1) / float(warmup_epochs)

        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.lr = lr

    # ------------------- Train (ONE epoch) -------------------
    # ------------------- Train (ONE epoch) -------------------
    def train(self):
        epoch = int(self.meta_info.get('epoch', 0))

        # unwrap DataParallel/DistributedDataParallel once
        m = self.model.module if hasattr(self.model, "module") else self.model

        # always expose epoch to model (for TAG gate / ramp)
        setattr(m, "cur_epoch", epoch)

        # optional: if you implemented a hook
        if hasattr(m, "set_epoch"):
            try:
                m.set_epoch(epoch)
            except Exception as e:
                self.io.print_log(f'[warn] set_epoch failed: {e!r}')

        # legacy fallback: some TAGLayer uses tag._epoch
        if hasattr(m, "tag") and (m.tag is not None):
            try:
                setattr(m.tag, "cur_epoch", epoch)
            except Exception:
                pass
            if hasattr(m.tag, "_epoch"):
                try:
                    m.tag._epoch = epoch
                except Exception:
                    pass


        self.adjust_lr()
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []

        for batch in loader:
            # 兼容 (data, label) 或 (data, label, sid/extra...)
            if isinstance(batch, (tuple, list)):
                data, label = batch[0], batch[1]
                sid = batch[2] if len(batch) > 2 else None
            else:
                raise RuntimeError("Unexpected batch type from feeder")

            data = data.float().to(self.output_device, non_blocking=True)
            label = label.long().to(self.output_device, non_blocking=True)

            logits = self.model(data)
            if isinstance(logits, (tuple, list)):
                # 兼容老代码：如果模型意外返回 tuple，只取第一个当分类 logits
                logits = logits[0]

            # 数值保护
            if not torch.isfinite(logits).all():
                self.io.print_log(f'!!! non-finite logits: min={float(logits.min())} max={float(logits.max())}')
                logits = logits.clamp(-100, 100)

            # CE
            loss_ce = self.ce_loss_train(logits, label)

            # KD：用 teacher 的输出（如果启用）
            if self.use_kd and (self.teacher is not None) and (self.kd_alpha > 0):
                with torch.no_grad():
                    teacher_logits = self.teacher(data)
                    if isinstance(teacher_logits, (tuple, list)):
                        teacher_logits = teacher_logits[0]
                loss_kd = soft_ce_kd(logits, teacher_logits, T=self.kd_temperature)
                loss = (1.0 - self.kd_alpha) * loss_ce + self.kd_alpha * loss_kd
            else:
                loss_kd = torch.tensor(0.0, device=logits.device)
                loss = loss_ce

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # log iter：loss 用数值，另存 msg（避免奇怪回显）
            cur = float(loss.item())
            loss_value.append(cur)
            self.iter_info['loss'] = cur
            self.iter_info['loss_ce'] = float(loss_ce.item())
            self.iter_info['loss_kd'] = float(loss_kd.item())
            self.iter_info['lr'] = float(self.lr)

            self.iter_info['loss_msg'] = (
                f"{cur:.4f} (CE {self.iter_info['loss_ce']:.4f} / KD {self.iter_info['loss_kd']:.4f})"
            )

            self.show_iter_info()
            self.meta_info['iter'] += 1

        mean_loss = float(np.mean(loss_value)) if loss_value else 0.0
        self.epoch_info['mean_loss'] = mean_loss
        self.show_epoch_info()
        self.io.print_timer()
        self._log_epoch_metric('train', loss=mean_loss)


    # ------------------- Eval / Test -------------------
    def test(self, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        name_frag = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    data, label = batch[0], batch[1]
                    sid = batch[2] if len(batch) > 2 else None
                else:
                    raise RuntimeError("Unexpected batch type from feeder")

                data = data.float().to(self.output_device, non_blocking=True)
                label = label.long().to(self.output_device, non_blocking=True)

                logits = self.model(data)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                result_frag.append(logits.detach().cpu().numpy())
                if sid is not None:
                    # sid 可能是 list/tuple of strings
                    if isinstance(sid, (list, tuple)):
                        name_frag.extend([str(x) for x in sid])
                    else:
                        name_frag.append(str(sid))

                if evaluation:
                    loss = self.ce_loss_eval(logits, label)
                    loss_value.append(float(loss.item()))
                    label_frag.append(label.detach().cpu().numpy())

        self.result = np.concatenate(result_frag) if result_frag else np.zeros((0, 0), dtype=np.float32)

        if evaluation:
            self.label = np.concatenate(label_frag) if label_frag else np.array([], dtype=np.int64)
            val_loss = float(np.mean(loss_value)) if loss_value else 0.0
            self.epoch_info['mean_loss'] = val_loss
            self.show_epoch_info()

            # Top1 / Top5
            rank = self.result.argsort() if len(self.result) else np.zeros((0, 0), dtype=np.int64)
            labels = self.label
            n = len(labels)
            num_cls = int(self.result.shape[1]) if len(self.result) else 0

            hit1 = [l in rank[i, -1:] for i, l in enumerate(labels)] if n else []
            hit5 = [l in rank[i, -min(5, num_cls):] for i, l in enumerate(labels)] if n else []
            top1 = (sum(hit1) / n) if n else 0.0
            top5 = (sum(hit5) / n) if n else 0.0

            for k in self.arg.show_topk:
                self.show_topk(k)

            epoch_now = int(self.meta_info.get('epoch', 0))
            self._log_epoch_metric('val', loss=val_loss, top1=100.0 * top1, top5=100.0 * top5)

            # 保存 logits/labels/names（优先用 sid）
            names = name_frag if name_frag else None
            self._save_eval_arrays(epoch_now, logits_np=self.result.astype(np.float32),
                                labels_np=self.label.astype(np.int64), names=names)

            is_better = (top1 > self.best_top1) or (abs(top1 - self.best_top1) < 1e-12 and val_loss < self.best_loss)
            if is_better:
                self.best_top1 = top1
                self.best_loss = val_loss
                self.best_epoch = epoch_now
                self.io.save_model(self.model, 'best_model.pt')
                self.io.print_log(f"[BEST] epoch {self.best_epoch} | Top1={100*self.best_top1:.2f}% | loss={self.best_loss:.4f} -> saved to best_model.pt")

            if (epoch_now + 1) >= int(self.arg.num_epoch):
                self.io.print_log(
                    f"[BEST SUMMARY] epoch {self.best_epoch} | "
                    f"Top1={100*self.best_top1:.2f}% | loss={self.best_loss:.4f}"
                )

    # 旧接口：TopK 打印
    def show_topk(self, k: int):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        acc = (sum(hit_top_k) * 1.0 / len(hit_top_k)) if hit_top_k else 0.0
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * acc))

    # ------------------- Arg parser -------------------
    @staticmethod
    def get_parser(add_help=False):
        parent = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent],
            description='Spatial Temporal Graph Convolution Network (dual-head)'
        )
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim / lr
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='epochs to decay LR by 0.1')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='linear warmup epochs (optional)')
        parser.add_argument('--optimizer', default='SGD', help='SGD or Adam')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
        parser.add_argument('--label_smoothing', type=float, default=0.0, help='cross entropy label smoothing')
        # KD（可选，默认关闭）
        parser.add_argument('--use_kd', type=str2bool, default=False)
        parser.add_argument('--kd_alpha', type=float, default=0.0)
        parser.add_argument('--kd_temperature', type=float, default=4.0)
        parser.add_argument('--kd_teacher_model', type=str, default=None)
        parser.add_argument('--kd_teacher_args', type=dict, default=dict())
        parser.add_argument('--kd_teacher_weights', type=str, default=None)
        return parser
