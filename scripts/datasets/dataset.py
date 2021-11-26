from os import listdir
from os.path import isfile, join
from typing import List

import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from datasets.vocab import Vocab

    
class NerDataset(Dataset):
    """Creates a pytorch dataloader given the data directory"""

    def __init__(self, root_dir: str, expand_vocab:bool = True):
        self.vocab = Vocab()
        self.expand_vocab = expand_vocab
        self.data = self.read_data(root_dir)
        
  
    def read_data(self, root_dir: str) -> List[List]:
        """
        Reads all csv files from the given directory path, updates vocab and 
        returns content of the files as list of list
        
        Args:
            root_dir: string of directory path containing csv files

        Returns:
            data: list of lists containing each row of data : [token, start_idx, end_idx, tag]
        """

        fnames = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f)) and f.endswith(".csv")]
        data = []
        
        for fname in fnames:
            contents = pd.read_csv(fname, index_col=None, header=0)
            contents = contents.values.tolist()
            for row in contents:
                data.append(row) 

                # expand_vocab is being used only for train data in this project
                if self.expand_vocab:
                    self.vocab.add_to_vocab(row[0], row[-1]) # Adding token and it's tag to the vocab
        
        return data
    
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return self.data[index]