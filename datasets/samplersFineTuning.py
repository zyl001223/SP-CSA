import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, label_dalle, n_batch, n_cls, n_per, dalle_shot=5):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per
        self.dalle_shot = dalle_shot

        label = np.array(label)  # all data label
        label_dalle = np.array(label_dalle) # all dalle data label
        if not np.array_equal(label, label_dalle):
            raise ValueError("The labels of the two datasets are not equal.")
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexes,e.g. 5
            
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per + self.dalle_shot]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).reshape(-1)
            # no .t() transpose (in contrast to 'permuted' sampler),
            # As such, the data and labels stay in the sequence of order 'aaaaabbbbbcccccdddddeeeee' after reshape,
            # instead of 'abcdeabcdeabcde'...
            yield batch