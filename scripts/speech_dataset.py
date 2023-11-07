import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import *

class SpeechData(Dataset):
    def __init__(self):
        Xset, Yset = preprocess()
        self.x = torch.concat(Xset, dim=0).reshape(-1, 1, 384)
        self.y = torch.concat(Yset, dim=0).reshape(-1, 1, 1)
        self.n = self.x.shape[0]
        assert self.x.shape[0] == self.y.shape[0], "number of audio windows and frequency labels does not match"

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y

class voicedData(SpeechData):
    def __init__(self):
        super().__init__()
        self.y = torch.where(self.y > 0, 1.0, 0.0)

class audioPitchTranformerData(SpeechData):
    #zeroRate is the percentage of zero frequency data to keep
    def __init__(self, zeroRate=0.2):
        super(audioPitchTranformerData, self).__init__()
        # Determine indices to keep based on non-zero frequencies and amount of zero data to keep
        non_zero_indices = torch.where(self.y != 0)[0]
        zero_indices = torch.where(self.y == 0)[0]
        num_zero = len(zero_indices)
        num_nonZero = len(non_zero_indices)
        zeros_to_keep = torch.randperm(num_zero)[:int(zeroRate * len(zero_indices))]
        zeros_to_keep = zero_indices[zeros_to_keep]
        indices_to_keep, _ = torch.concat([non_zero_indices, zeros_to_keep], dim=0).sort()

        self.x = self.x[indices_to_keep]
        self.y = self.y[indices_to_keep]
        
        self.n = int(num_zero * zeroRate) + num_nonZero
        assert self.x.shape[0] == self.n, "length does not match for zero selection"