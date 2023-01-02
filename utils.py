import torch
from pathlib import Path
import download_dataset as dataset
import seaborn as sns
import pandas as pd
from classifier_train import train
import matplotlib.pyplot as plt
import numpy as np


def save_trained_auto_encoder(model:torch.nn.Module, directory:str = 'model', name:str = 'AE'):
    model_save_path = Path(directory) / (name + '.pth')
    torch.save(model,model_save_path)


def load_saved_auto_encoder(directory:str = 'model', name:str = 'AE'):
    model_load_path = Path(directory) / (name + '.pth')
    if not model_load_path.is_file():
        print('There is no trained auto-encoder to load !')
        return False
    return torch.load(f=model_load_path)

def load_saved_classifier(name:str = 'classifier'):
    model_load_path = Path(name + '.pth')
    if not model_load_path.is_file():
        print('There is no trained classifier to load !')
        return False
    return torch.load(f=model_load_path)


def create_auto_encoder_latent_space(visualize_latent_space:bool = False,train_classifier:bool = True):
    
    # load tranied auto_encoder
    auto_encoder = load_saved_auto_encoder()

    # load test dataloader
    test_dataloader,_ = dataset.create_test_dataset(path='data',batch_size=32)

    if auto_encoder:

        # extract the encoder part
        encoder = auto_encoder.encoder_cnn

        encoder.eval()
        with torch.inference_mode():
            for i,(features,label) in enumerate(test_dataloader):

                # calculate the latent features
                encoder_output = encoder(features)

                # Associate the latent features with a label
                encoder_output = torch.cat((encoder_output,torch.unsqueeze(label,1)),dim = -1)

                if i == 0:
                    latent_space = encoder_output
                else:
                    latent_space = torch.cat((latent_space,encoder_output),dim=0)

        df = pd.DataFrame(latent_space.detach().numpy(), columns=["x1", "x2" ,"digit"])
        df = df.astype({'digit':'int'})

        if visualize_latent_space:
            plt.figure()
            sns.scatterplot(data=df,x='x1',y='x2',hue='digit',palette='deep').set(title='Auto-encoder latent space')

        if train_classifier:
            # Train a classifier on the generated latent space
            train(df=df)

    return df

def get_image():
    test_dataloader,_ = dataset.create_test_dataset(path='data',batch_size=32)
    x = iter(test_dataloader)
    features,label = next(x)
    return features[0].reshape(1,1,28,28),label[0].item()


def predict_image_class(image:torch.tensor,auto_encoder:torch.nn.Module,classifier:torch.nn.Module,visualisation: bool = False):

    if auto_encoder and classifier:
        encoder = auto_encoder.encoder_cnn
        encoder.eval()
        classifier.eval()
        auto_encoder.eval()
        with torch.inference_mode():

            # decode the input
            decoded_image = auto_encoder(image)
            # Generate the 2D latent variable for the input image
            latent_variable = encoder(image)
            # run the classifier on the latent varibale
            y_logit = classifier(latent_variable)
            # convert model output to probabilities
            probabilities = torch.softmax(y_logit, dim=1)
            # get the predicted class
            pred_class = probabilities.argmax(dim=1)
            # get the corresponding probability
            probability,_ = probabilities.max(dim=1)

        if visualisation:
            fig = plt.figure()
            fig.text(x=0.35,y=0.2,s=f'The input image is {probability.item()*100:.3f}% "{pred_class.item()}"')
            fig.add_subplot(1,2,1)
            plt.title('Original input Image')
            plt.axis('off')
            plt.imshow(image.reshape(28,28).numpy(),cmap='gist_gray')
            fig.add_subplot(1,2,2)
            plt.title('Decoded Image')
            plt.axis('off')
            plt.imshow(decoded_image.reshape(28,28).numpy(),cmap='gist_gray')

        return pred_class,probability



def map_latent_space_probability_space(latent_space: pd.DataFrame,auto_encoder:torch.nn.Module,classifier:torch.nn.Module):
    
    # load test dataloader
    test_dataloader,_ = dataset.create_test_dataset(path='data',batch_size=32)

    # local variables setUp
    probabilities = np.array([])
    predicted_classes = np.array([])

    for i,(features,label) in enumerate(test_dataloader):
        classes,proba = predict_image_class(image=features,auto_encoder=auto_encoder,classifier=classifier)
        probabilities = np.append(probabilities,proba.numpy())
        predicted_classes = np.append(predicted_classes,classes.numpy())
    
    latent_space['predicted_class'] = predicted_classes
    latent_space['probability'] = probabilities

    return latent_space


def visualize_mapping(mapped_latent_space:pd.DataFrame, save:bool):
    ax = sns.jointplot(x=mapped_latent_space.x1, y=mapped_latent_space.x2,
              cut = 2, hue=mapped_latent_space.predicted_class,
              palette='deep',
              kind='kde', fill=True,
              height=10, ratio=6,
              joint_kws = dict(alpha=0.6),
              marginal_kws=dict(fill=True),
              legend=False)
    ax = sns.despine(ax=None, left=True, bottom=True)
    if save:
        plt.savefig('mapping.png')










    






    
    




            


                
            