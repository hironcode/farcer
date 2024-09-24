import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Assuming the last column is the label
        label = row['Emotion']

        # Other columns are features
        features = row.loc[['Utterances', "image_path"]].values.astype('float32')
        
        sample = {'features': features, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
