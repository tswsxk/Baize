# coding: utf-8
# 2022/4/25 @ tongshiwei

import multiprocess as mp
import torch
from tqdm import tqdm
from longling import print_time
from torch.utils.data import Dataset as _Dataset, DataLoader
from pseudo_data import pseudo_data, pseudo_seq_data
from baize.utils import pad_sequence


def test_pseudo_data():
    data = pseudo_data()

    class Dataset(_Dataset):
        def __getitem__(self, item):
            return data[item]

        def __len__(self):
            return len(data)

    dataset = Dataset()
    data_loader = DataLoader(dataset, 16)
    with print_time("single process"):
        for e in range(5):
            for _ in tqdm(data_loader, "loading %s" % e):
                pass

    data_loader = DataLoader(dataset, 16, num_workers=4)
    with print_time("multi process: %s" % 4):
        for e in range(5):
            for _ in tqdm(data_loader, "loading %s" % e):
                pass


def test_pseudo_seq_data():
    data = pseudo_seq_data()

    class Dataset(_Dataset):
        def __getitem__(self, item):
            return data[item]

        def __len__(self):
            return len(data)

    def padding_seq(batch):
        return torch.tensor(pad_sequence(batch))

    dataset = Dataset()
    data_loader = DataLoader(
        dataset,
        16,
        collate_fn=padding_seq
    )
    with print_time("single process"):
        for e in range(5):
            for _ in tqdm(data_loader, "loading %s" % e):
                pass

    data_loader = DataLoader(
        dataset,
        16,
        collate_fn=padding_seq
    )
    with print_time("single process"):
        for e in range(5):
            for _ in tqdm(data_loader, "loading %s" % e):
                pass

    data_loader = DataLoader(
        dataset, 16, num_workers=4, collate_fn=padding_seq,
    )
    with print_time("multi process: %s" % 4):
        for e in range(5):
            for _ in tqdm(data_loader, "loading %s" % e):
                pass


if __name__ == '__main__':
    # test_pseudo_data()
    test_pseudo_seq_data()
