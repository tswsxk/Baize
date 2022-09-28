# coding: utf-8
# 2022/4/25 @ tongshiwei

import torch
from pseudo_data import pseudo_data
from torch.utils.data import Dataset as _Dataset, DataLoader
from longling import print_time
from tqdm import tqdm
from baize import iterwrap


class Dataset(_Dataset):
    def __init__(self):
        self.data = pseudo_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    return torch.tensor(batch)


@iterwrap(level="p")
def batchify(dataset, batch_size=8):
    batch_data = []
    for d in dataset:
        batch_data.append(d)
        if len(batch_data) == batch_size:
            yield torch.tensor(batch_data)
    if batch_data:
        yield torch.tensor(batch_data)


def test_multi_process2():
    dataset = Dataset()
    data_loader = batchify(dataset)
    with print_time("multi process: %s" % 4):
        for e in range(5):
            for _ in tqdm(data_loader, "loading %s" % e):
                pass


def test_multi_process():
    dataset = Dataset()
    data_loader = DataLoader(dataset, 16, collate_fn=collate_fn, num_workers=2)
    with print_time("multi process: %s" % 4):
        for e in range(5):
            for _ in tqdm(data_loader, "loading %s" % e):
                pass


if __name__ == '__main__':
    test_multi_process2()
