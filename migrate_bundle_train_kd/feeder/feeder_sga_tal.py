import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_path, split="train", mmap=True, debug=False):
        self.data_path = data_path
        self.split = split
        self.debug = debug
        self.mmap = mmap
        self.load_data()

    def load_data(self):
        data_file = os.path.join(self.data_path, f"{self.split}_data.npy")
        segments_file = os.path.join(self.data_path, f"{self.split}_segments.pkl")
        names_file = os.path.join(self.data_path, f"{self.split}_names.pkl")

        self.data = np.load(data_file, mmap_mode="r" if self.mmap else None)
        with open(segments_file, "rb") as f:
            self.segment_meta = pickle.load(f)
        with open(names_file, "rb") as f:
            self.sample_name = pickle.load(f)

        if self.debug:
            self.data = self.data[:100]
            self.segment_meta = self.segment_meta[:100]
            self.sample_name = self.sample_name[:100]

    def __len__(self):
        return len(self.segment_meta)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index], dtype=np.float32)
        meta = self.segment_meta[index]
        sid = meta.get("sid", self.sample_name[index])
        segments = np.asarray(meta.get("segments", []), dtype=np.int64).reshape(-1, 3)
        valid_t = int(meta.get("length", data_numpy.shape[1]))
        target = {
            "segments": segments,
            "T": np.int64(valid_t),
        }
        return torch.from_numpy(data_numpy), target, sid

