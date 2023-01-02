import torch
import torch.nn as nn




class Classifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_features=2,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=10),
        )

    def forward(self,X):
        return self.nn(X)