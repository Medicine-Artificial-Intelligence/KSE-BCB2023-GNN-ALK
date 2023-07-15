import pandas as pd
from rdkit import Chem
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import deepchem as dc
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

"""
!!!
NOTE: This file was replaced by dataset_featurizer.py
but is kept to illustrate how to build a custom dataset in PyG.
!!!
"""
# Featurizing class
class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, valid = False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.valid = valid
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test == True:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.valid == True:
            return [f'data_valid_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(row["Canomicalsmiles"])
            f = featurizer._featurize(mol)
            #node features
            node_features = torch.tensor(f.node_features, dtype=torch.float)
            #edge index
            edge_index = torch.tensor(f.edge_index, dtype=torch.int64)
            #edge features
            edge_features = torch.tensor(f.edge_features, dtype=torch.float)

            label = self._get_labels(row["pChEMBL"])
            smiles = row["Canomicalsmiles"]
            data = Data(node_features=node_features, 
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        y = label,
                        smiles= row["Canomicalsmiles"]
                        ) 

            if self.test ==True:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            elif self.valid == True:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_valid_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))

    

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test == True :
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        elif self.valid == True:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_valid_{idx}.pt'))
            
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data