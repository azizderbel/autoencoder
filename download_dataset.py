import torch
import torchvision
from torch.utils.data import DataLoader,random_split



def create_train_validation_dataset(path: str,batch_size: int,validation_rate: float):

    # Download training data from the torchvision repo
    train_data = torchvision.datasets.MNIST(
    root = path,
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor()
    )

    total_dataset_length = len(train_data)

    class_names = train_data.classes
    # Split the downloaded dataset into a training and a validation folds
    train_data,validation_data = random_split(train_data,[int(total_dataset_length * (1 - validation_rate)),int(total_dataset_length * validation_rate)])

    # Create Train and validation dataloaders
    train_loader = DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = True
    )

    validation_loader = DataLoader(
        dataset = validation_data,
        batch_size = batch_size,
        shuffle = True
    )

    print(f'{len(train_data)} training samples were reserved successfully')
    print(f'{len(validation_data)} validation samples were reserved successfully')

    return train_loader,validation_loader,class_names


def create_test_dataset(path: str,batch_size: int):

    # Download test data from the torchvision repo
    test_data = torchvision.datasets.MNIST(
    root = path,
    train = False,
    download = True,
    transform = torchvision.transforms.ToTensor()
    )

    class_names = test_data.classes
    # Create test dataloaders
    test_dataloader = DataLoader(
        dataset = test_data,
        batch_size = batch_size,
        shuffle = False
    )

    return test_dataloader,class_names

