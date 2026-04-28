import os
import lmdb 
import pickle 

from functools import lru_cache, partial
from tqdm import tqdm

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from torch_cluster import radius_graph

MAX_NMR_SHIFTS = 32

def pre_transform_func(data: dict) -> Data:
    
    pos = torch.as_tensor(data['pos'], dtype=torch.float32)
    edge_index = radius_graph(x=pos, r=5.0)
    
    if 'hnmr' in data and 'cnmr' in data:
        hnmr = torch.as_tensor(data['hnmr'], dtype=torch.float32)
        cnmr = torch.as_tensor(data['cnmr'], dtype=torch.float32)
        
        hnmr = torch.sort(hnmr, dim=0)[0]
        cnmr = torch.sort(cnmr, dim=0)[0]
        
        padded_hnmr = torch.zeros(MAX_NMR_SHIFTS, dtype=torch.float32)
        padded_hnmr[:len(hnmr)] = hnmr.flatten()
        
        padded_cnmr = torch.zeros(MAX_NMR_SHIFTS, dtype=torch.float32)
        padded_cnmr[:len(cnmr)] = cnmr.flatten()
    
    data_object = Data(
        pos=pos,
        edge_index=edge_index,
        x=torch.as_tensor(data['z'], dtype=torch.long),
        ir=torch.as_tensor(data['ir'], dtype=torch.float32).reshape(1, -1),
        raman=torch.as_tensor(data['raman'], dtype=torch.float32).reshape(1, -1),
        hnmr=padded_hnmr.reshape(1, -1) if 'hnmr' in data else None,
        cnmr=padded_cnmr.reshape(1, -1) if 'cnmr' in data else None,
        )
    
    return data_object

class Spec2ConfDataset(InMemoryDataset):
    def __init__(self, root, mode, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return [f'{self.mode}.lmdb']

    @property
    def processed_file_names(self):
        return [f'{self.mode}.pt']

    def process(self):
        
        # Read data into huge `Data` list.
        db = lmdb.open(f'{self.root}/{self.mode}.lmdb', subdir=False, lock=False, map_size=int(1e11)) 
        with db.begin() as txn: 
            data = list(txn.cursor())
        data_list = [pickle.loads(item[1]) for item in data]

        n_jobs = -1

        if self.pre_filter is not None:
            data_list = [data for data in tqdm(data_list) if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in tqdm(data_list)]
        
        if self.pre_transform is not None:
            data_list = Parallel(n_jobs=n_jobs)(
                delayed(self.pre_transform)(data) for data in tqdm(data_list, desc="Transforming")
            )
            
        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])
        


     
class Dataloader:
    def __init__(self, 
                 ds, 
                 data_dir='',
                 target_keys=None, 
                 collate_fn=None,
                 device='cpu',
                 force_reload=False):

        self.ds = ds
        self.target_keys = target_keys
        self.data_dir = data_dir
        self.collate_fn = collate_fn
        self.device = device
        self.force_reload = force_reload
        
    def generate_dataset(self, verbose=False):

        if verbose: 
            print(f'[train set] = {self.ds} | [target keys] = {self.target_keys}')
        
        self.dataset = Spec2ConfDataset(
            root=f'{self.data_dir}/{self.ds}', 
            mode=self.mode, 
            pre_transform=pre_transform_func,
            force_reload=self.force_reload,
            )
        
    def generate_dataloader(self,
                            mode='train',
                            batch_size=16, 
                            num_workers=0, 
                            ddp=False):
        
        self.mode = mode
        self.generate_dataset()
        shuffle = True if mode == 'train' else False
        
        if ddp:
            data_sampler = DistributedSampler(self.dataset, shuffle=shuffle)
            dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=data_sampler, pin_memory=True)
            return dataloader
        else:
            dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
            return dataloader
