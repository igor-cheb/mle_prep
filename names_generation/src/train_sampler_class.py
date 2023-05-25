import numpy as np
import torch
from src.utilities import RAW_DATA_PATH

class TrainSampler():
    """Class for reading and sampling data for training"""
    def __init__(self, 
                 vocabulary: list, 
                 file_path: str=RAW_DATA_PATH):
        self.words = open(file_path, 'r').read().splitlines()
        self.vocabulary = vocabulary

    def _word2vec(self, word: str) -> list:
        """Function that transforms passed word into a vector of indicies using input vocabulary"""
        return [self.vocabulary.index(let) for let in word]

    def sample_train(self):
        """Function beaks down a random word from passed list into train and target samples"""
        word = self._word2vec(self.words[np.random.choice(range(len(self.words)))] + '.')
        X = []; y = []
        for n, ch in enumerate(word[:-1]):
            X.append(ch); y.append(word[n+1])
        return torch.tensor(X), torch.tensor(y).float()