import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from classifier import Classifier
from classifier_dataset import ClassifierDataset,create_data_loader

def latent_space_data_preprocessing(df:pd.DataFrame):
    train_dataloader = create_data_loader(ClassifierDataset(df=df),batch_size=32)
    return train_dataloader

def train(df:pd.DataFrame):

    # output console mesg
    print('Classifier training process is about to start ...')
    # Get data 
    train_dataloader = latent_space_data_preprocessing(df)
    # set hyperparams
    n_epochs = 100
    lr = 0.2
    # create a classifier instance 
    classifier = Classifier()
    # define a loss function and an optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr,weight_decay=1e-05)

    classifier.train()
    for epoch in range(n_epochs):
        train_loss = 0
        for i,(features,label) in enumerate(train_dataloader):
            # forward pass
            out = classifier(features)
            
            # loss calcul
            loss = criterion(out,label)
            train_loss += loss.item()

            optimizer.zero_grad()
            # backward pass
            loss.backward()
            
            #update weights
            optimizer.step()
        print(f'Epoch {epoch} : training loss {train_loss / len(train_dataloader)}')

    print('Classifier training process completed')
    torch.save(classifier,'classifier.pth')
            

