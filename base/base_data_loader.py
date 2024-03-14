import numpy as np
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class BalanceSampler(Sampler):
    # use when training with control posts
    def __init__(self, data_source, control_ratio=0.75) -> None:
        self.data_source = data_source
        self.control_ratio = control_ratio
        self.indexes_control = np.where(data_source.is_control == 1)[0]
        self.indexes_mental = np.where(data_source.is_control == 0)[0]
        self.len_control = len(self.indexes_control)
        self.len_mental = len(self.indexes_mental)

        np.random.shuffle(self.indexes_control)
        np.random.shuffle(self.indexes_mental)

        self.pointer_control = 0
        self.pointer_mental = 0
    
    def __iter__(self):
        for i in range(len(self.data_source)):
            if np.random.rand() < self.control_ratio:
                id0 = np.random.randint(self.pointer_control, self.len_control)
                sel_id = self.indexes_control[id0]
                self.indexes_control[id0], self.indexes_control[self.pointer_control] = self.indexes_control[self.pointer_control], self.indexes_control[id0]
                self.pointer_control += 1
                if self.pointer_control >= self.len_control:
                    self.pointer_control = 0
                    np.random.shuffle(self.indexes_control)
            else:
                id0 = np.random.randint(self.pointer_mental, self.len_mental)
                sel_id = self.indexes_mental[id0]
                self.indexes_mental[id0], self.indexes_mental[self.pointer_mental] = self.indexes_mental[self.pointer_mental], self.indexes_mental[id0]
                self.pointer_mental += 1
                if self.pointer_mental >= self.len_mental:
                    self.pointer_mental = 0
                    np.random.shuffle(self.indexes_mental)
            
            yield sel_id

    def __len__(self) -> int:
        return len(self.data_source)
        
class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, split, bal_sample, control_ratio, num_workers, collate_fn=default_collate):
        
        self.shuffle = shuffle
        self.split = split
        
        self.batch_idx = 0
        self.n_samples = len(dataset)
        
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        
        self.sampler = None
        if self.split == 'train' and bal_sample :
            self.sampler = BalanceSampler(dataset, control_ratio)
            self.init_kwargs['shuffle'] = False
            self.n_samples = len(self.sampler) # 이게 맞나? 
        
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    # def split_validation(self):
    #     if self.split == 'train' :
    #         self.init_kwargs['shuffle'] = self.shuffle
    #         return DataLoader(sampler=None, **self.init_kwargs)
    #     else:
    #         return None