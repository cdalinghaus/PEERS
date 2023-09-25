from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
import numpy as np

from io import StringIO
import pandas as pd

import os
import random

class EEGDataset:
    
    def __init__(self, block_size, data_path = "tmp/nanoGPT/"):
        
        self.train_files = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.endswith(".rhead")]
        
        self.tokenizer = Tokenizer.from_file("../bpe_tokenizer.json")
        
        self.chunk_size = 50000
        self.block_size = block_size
        self.device = "cpu"

    def encode(self, s):
        return self.tokenizer.encode(s).ids  # This will output a list of integers

    def decode(self, l):
        return self.tokenizer.decode(l).replace(" ", "")  # This will output a string

    def power_law_random_int(self, X, a=1.00001):
        """
        Generate a power-law distributed random integer between 0 and X.

        :param X: The upper bound for the random number.
        :param a: The shape parameter for the Zipf's law distribution. It should be > 1.
                  The larger 'a' is, the steeper the decay of the distribution.

        :return: A random integer.
        """

        while True:
            value = np.random.zipf(a)  # zipf starts from 1
            if 0 <= value < X:
                return value


    def read_random_chunk(self, file_path):
        # Get the total file size
        total_size = os.path.getsize(file_path)

        # Calculate a random start offset
        start_offset = random.randint(0, total_size - self.chunk_size)

        # Read a chunk of data
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(start_offset)  # Jump to the random offset
            text = f.read(self.chunk_size)  # Read the chunk

            # Adjust the text to start from the next newline character after the offset.
            # This ensures that the text doesn't start in the middle of a sentence.
            if start_offset != 0:
                newline_pos = text.find('\n')
                if newline_pos != -1:
                    text = text[newline_pos + 1:]

        t = StringIO(text)
        df = pd.read_csv(t, sep=",")

        df = df.drop(df.columns[0],axis=1)

        num_columns = len(df.columns)

        # Select a random number of columns to keep (between 1 and the total number of columns)
        num_to_select = random.randint(1, num_columns)
        num_to_select = self.power_law_random_int(num_columns)

        # Randomly sample columns
        selected_columns = random.sample(list(df.columns), num_to_select)

        df = df[selected_columns]
        string = df.to_csv()

        ids = torch.Tensor(self.encode(string))
        length = len(ids)

        if length < (self.block_size+1):
            #print(f"Chunk that was read from file was too short, trying again. You should probably increase chunk_size ({chunk_size}).")
            self.chunk_size += 1000
            return self.read_random_chunk(file_path)

        ids = ids[:self.block_size+1]
        return ids[:-1], ids[1:]
    
    
    def get_batch(self):
        self.chunk_size -= 1
        file_name = random.choice(self.train_files)
        
        X, y = self.read_random_chunk(file_name)
        X, y = X.long(), y.long()
        
        device = self.device
        if device == 'cuda':
            X, y = X.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            X, y = X.to(device), y.to(device)
        return X, y
    
    def __len__(self):
        return 1000000
    
    def __getitem__(self, idx):
        return self.get_batch()
