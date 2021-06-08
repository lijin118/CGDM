import random
import torch.utils.data
from builtins import object


class PairedData(object):
    def __init__(self, data_loader_S, data_loader_T, max_dataset_size, flip):
        self.data_loader_S = data_loader_S
        self.data_loader_T = data_loader_T
        self.stop_S = False
        self.stop_T = False
        self.max_dataset_size = max_dataset_size
        self.flip = flip

    def __iter__(self):
        self.stop_S = False
        self.stop_T = False
        self.data_loader_S_iter = iter(self.data_loader_S)
        self.data_loader_T_iter = iter(self.data_loader_T)
        self.iter = 0
        return self

    def __next__(self):
        S, S_paths = None, None
        T, T_paths = None, None
        try:
            S, S_paths, S_indexes = next(self.data_loader_S_iter)
        except StopIteration:
            if S is None or S_paths is None:
                self.stop_S = True
                self.data_loader_S_iter = iter(self.data_loader_S)
                S, S_paths, S_indexes = next(self.data_loader_S_iter)

        try:
            T, T_paths, T_indexes = next(self.data_loader_T_iter)
        except StopIteration:
            if T is None or T_paths is None:
                self.stop_T = True
                self.data_loader_T_iter = iter(self.data_loader_T)
                T, T_paths, T_indexes = next(self.data_loader_T_iter)

        if (self.stop_S and self.stop_T) or self.iter > self.max_dataset_size:
            self.stop_S = False
            self.stop_T = False
            raise StopIteration()
        else:
            self.iter += 1
            if self.flip and random.random() < 0.5:
                idx = [i for i in range(S.size(3) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                S = S.index_select(3, idx)
                T = T.index_select(3, idx)
            return {'S': S, 'S_label': S_paths, 'S_index':S_indexes,
                    'T': T, 'T_label': T_paths, 'T_index':T_indexes}


class CVDataLoader(object):
    def initialize(self, dataset_S, dataset_T, batch_size, shuffle=True, drop_last=False):
        # normalize = transforms.Normalize(mean=mean_im,std=std_im)
        self.max_dataset_size = float("inf")  # infinite big
        data_loader_S = torch.utils.data.DataLoader(
            dataset_S,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4)
        data_loader_T = torch.utils.data.DataLoader(
            dataset_T,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4)
        self.dataset_S = dataset_S
        self.dataset_T = dataset_T
        flip = False
        self.paired_data = PairedData(data_loader_S, data_loader_T, self.max_dataset_size, flip)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_S), len(self.dataset_T)), self.max_dataset_size)
