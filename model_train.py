import torch
import torch.nn as nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
import download_dataset as dataset
import utils as utils


def train(model:nn.Module, epochs:int = 30, learning_rate:int = 0.004, batch_size:int = 32, save_model:bool = True):

    # Output a console mesg
    print(f'The process of training is about to start...')

    # Get training and validation set
    train_dataloader,validation_dataloader,classes_names = dataset.create_train_validation_dataset(path='data',batch_size=batch_size,validation_rate=0.2)
    
    # Setup loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    start_time = timer()

    # 3. Loop for a number of epochs
    for epoch in tqdm(range(epochs)):

        model.train()
        train_loss = 0
        validation_loss = 0

        for features, _ in train_dataloader:

            # forward pass
            decoded_image = model(features)
            # Evaluate loss
            loss = criterion(decoded_image, features)
            train_loss += loss.item()

            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # update model weights
            optimizer.step()

        # print epoch training loss
        print(f' Epoch {epoch} : training loss {train_loss / len(train_dataloader)}')
        model.eval()
        with torch.inference_mode():
            for features,_ in validation_dataloader:
                # Do the forward pass
                decoded_image = model(features)

                loss = criterion(decoded_image,features)
                validation_loss += loss.item()

        print(f'Epoch {epoch} : Validation loss {validation_loss / len(validation_dataloader)}')

    end_time = timer()

    print(f'Total training time: {end_time-start_time:.3f} seconds')

    if save_model:
        # save the model
        utils.save_trained_auto_encoder(model=model)
        print(f'model saved !')



        


        