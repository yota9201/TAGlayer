#!/usr/bin/env python
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import torchlight
from torchlight import str2bool
from torchlight import import_class

from .processor import Processor


def tal_collate_fn(batch):
    data = torch.stack([item[0] for item in batch], dim=0)
    target = [item[1] for item in batch]
    sid = [item[2] for item in batch]
    return data, target, sid


class TAL_Processor(Processor):
    def load_model(self):
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args).to(self.output_device)
        self.best_acc = -1.0
        self.work_dir = self.arg.work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = {}
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                drop_last=True,
                collate_fn=tal_collate_fn,
            )
        if self.arg.test_feeder_args:
            self.data_loader["test"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                collate_fn=tal_collate_fn,
            )

    def load_optimizer(self):
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError("Unknown optimizer")

    def adjust_lr(self):
        epoch = int(self.meta_info.get("epoch", 0))
        steps = self.arg.step if isinstance(self.arg.step, (list, tuple)) else [self.arg.step]
        steps = [int(s) for s in steps if s is not None]
        lr = float(self.arg.base_lr) * (0.1 ** sum(1 for s in steps if epoch >= s))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.lr = lr

    def _build_frame_targets(self, target, t_out, device):
        n = len(target)
        y = torch.full((n, t_out), int(self.arg.ignore_index), dtype=torch.long, device=device)
        mask = torch.zeros((n, t_out), dtype=torch.bool, device=device)

        for i, item in enumerate(target):
            t_in = max(1, int(item["T"]))
            scale = float(t_out) / float(max(1, t_in))
            segments = np.asarray(item["segments"], dtype=np.int64).reshape(-1, 3)
            for st, ed, cls in segments:
                st_ds = int(np.floor(float(st) * scale))
                ed_ds = int(np.ceil(float(ed) * scale))
                st_ds = max(0, min(t_out, st_ds))
                ed_ds = max(0, min(t_out, ed_ds))
                if ed_ds <= st_ds:
                    continue
                y[i, st_ds:ed_ds] = int(cls)
                mask[i, st_ds:ed_ds] = True

        return y, mask

    def _save_eval_arrays(self, epoch, logits_np, labels_np, sid_list):
        out_dir = os.path.join(self.work_dir, "eval_arrays")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"epoch{epoch:03d}_logits.npy"), logits_np)
        np.save(os.path.join(out_dir, f"epoch{epoch:03d}_labels.npy"), labels_np)
        with open(os.path.join(out_dir, f"epoch{epoch:03d}_sid.pkl"), "wb") as f:
            pickle.dump(sid_list, f)

    def train(self):
        epoch = int(self.meta_info.get("epoch", 0))
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_obj, "set_epoch"):
            model_obj.set_epoch(epoch)

        self.adjust_lr()
        self.model.train()
        loss_value = []

        for data, target, _sid in self.data_loader["train"]:
            data = data.float().to(self.output_device, non_blocking=True)
            logits = self.model(data)
            t_out = logits.size(-1)
            y_frame, mask = self._build_frame_targets(target, t_out=t_out, device=logits.device)

            ce = F.cross_entropy(logits, y_frame, ignore_index=int(self.arg.ignore_index), reduction="none")
            masked_loss = ce[mask].mean() if mask.any() else logits.sum() * 0.0

            self.optimizer.zero_grad(set_to_none=True)
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            loss_value.append(float(masked_loss.item()))
            self.iter_info["loss"] = float(masked_loss.item())
            self.iter_info["lr"] = float(self.lr)
            self.show_iter_info()
            self.meta_info["iter"] += 1

        self.epoch_info["mean_loss"] = float(np.mean(loss_value)) if loss_value else 0.0
        self.show_epoch_info()

    def test(self):
        self.model.eval()
        loss_value = []
        logits_frag = []
        labels_frag = []
        sid_frag = []
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target, sid in self.data_loader["test"]:
                data = data.float().to(self.output_device, non_blocking=True)
                logits = self.model(data)
                t_out = logits.size(-1)
                y_frame, mask = self._build_frame_targets(target, t_out=t_out, device=logits.device)

                ce = F.cross_entropy(logits, y_frame, ignore_index=int(self.arg.ignore_index), reduction="none")
                masked_loss = ce[mask].mean() if mask.any() else logits.sum() * 0.0
                loss_value.append(float(masked_loss.item()))

                pred = logits.argmax(dim=1)
                if mask.any():
                    correct += int((pred[mask] == y_frame[mask]).sum().item())
                    total += int(mask.sum().item())

                logits_frag.append(logits.detach().cpu().numpy())
                labels_frag.append(y_frame.detach().cpu().numpy())
                sid_frag.extend(list(sid))

        acc = (correct / total) if total > 0 else 0.0
        mean_loss = float(np.mean(loss_value)) if loss_value else 0.0
        self.epoch_info["mean_loss"] = mean_loss
        self.epoch_info["frame_acc"] = acc
        self.show_epoch_info()

        epoch = int(self.meta_info.get("epoch", 0))
        logits_np = np.concatenate(logits_frag, axis=0) if logits_frag else np.zeros((0, self.arg.num_class, 0), dtype=np.float32)
        labels_np = np.concatenate(labels_frag, axis=0) if labels_frag else np.zeros((0, 0), dtype=np.int64)
        self._save_eval_arrays(epoch, logits_np, labels_np, sid_frag)

        if acc > self.best_acc:
            self.best_acc = acc
            self.io.save_model(self.model, "best_model.pt")
            self.io.print_log(f"[BEST] epoch {epoch} | frame_acc={acc:.4f}")

    @staticmethod
    def get_parser(add_help=False):
        parent = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(add_help=add_help, parents=[parent], description="Frame-wise TAL Processor")
        parser.add_argument("--num_class", type=int, default=20)
        parser.add_argument("--in_channels", type=int, default=4)
        parser.add_argument("--max_frame", type=int, default=400)
        parser.add_argument("--ignore_index", type=int, default=-100)
        parser.add_argument("--downsample_rate", type=int, default=1)
        parser.add_argument("--base_lr", type=float, default=0.001)
        parser.add_argument("--step", type=int, default=[], nargs="+")
        parser.add_argument("--optimizer", default="AdamW")
        parser.add_argument("--nesterov", type=str2bool, default=True)
        parser.add_argument("--weight_decay", type=float, default=0.0005)
        return parser

