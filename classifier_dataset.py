import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader

class ClassifierDataset(Dataset):
    def __init__(self, df:pd.DataFrame) -> None:
        X = df[['x1','x2']].to_numpy()
        y = df['digit'].to_numpy()
        self.n_samples = X.shape[0]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).to(torch.long)
    
    def __getitem__(self,index):
        return self.X[index],self.y[index]
    
    def __len__(self):
        return self.n_samples

def create_data_loader(dataset:Dataset, batch_size = 32):
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)




    

