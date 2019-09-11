import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

# Import Pytorch modules
import torch
from torchvision import datasets, transforms


def load_data(path):
    """Loads, transforms, and creates torch.utils.data.Dataloaders
       for data for model training.

    Args:
        path (str): filepath to data directories with sub-directory
                    structure:
                    path/train/..
                    path/valid/..
                    path/test/..

    Returns:
        dataloaders (dict): {'train': Dataloader(train_data),
                             'valid':, Dataloader(valid_data),
                             'test': Dataloader(test_data)}
    """
    # Training Images Details
    IMG_SIZE = 224  # Size of images used for training
    IMG_MEAN = [0.485, 0.456, 0.406]  # image normalization mean
    IMG_SDEV = [0.229, 0.224, 0.225]  # image normalization standard deviation

    # Training phases
    phases = ['train', 'valid', 'test']

    # Define data locations
    data_dir = {n: path + n for n in phases}

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_SDEV)]),
        'valid':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_SDEV)]),
        'test':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_SDEV)])
    }

    # Load the datasets
    image_datasets = {n: datasets.ImageFolder(
                            data_dir[n], transform=data_transforms[n])
                      for n in phases}

    # Create the PyTorch dataloaders
    dataloaders = {n: torch.utils.data.DataLoader(
                        image_datasets[n], batch_size=64, shuffle=True)
                    for n in phases}

    # mapping of classes to training indices
    class_to_idx = image_datasets['train'].class_to_idx

    return dataloaders, class_to_idx


def display_prediction(image_path, probabilities, predictions):
    """Displays classified image with top predicted class as title and
       horizontal bar chart of predicted probabilities of predicted top classes

    Args:
        image_path (str): path to image to be classified
        probabilities ([float]): list of predicted probabilities for
                                 topk classes
        class_idxs ([int]): list of predicted topk class indices
        class_names ([str]): list of predicted topk class names
    """
    top_class = predictions[0]

    # Setup plot gird and title
    fig = plt.figure(figsize=(4, 5.4))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    fig.suptitle(top_class.capitalize(), x=0.6, y=1, fontsize=16)

    # Display image
    ax1.imshow(Image.open(image_path))
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Display predicted classes and probabilities
    y = np.arange(len(predictions))  # setup y axis grid
    ax2.barh(y, probabilities)
    ax2.set_yticks(y)
    ax2.set_yticklabels(predictions)
    ax2.invert_yaxis()  # prediction with highest probability on top
    ax2.set_xlabel('Prediction probability')

    # Adjust layout
    fig.tight_layout()
    plt.subplots_adjust(top=0.93)

    plt.show()

def prediction_class_names(predictions, class_to_idx, category_names):
    """convert indeces to named classesself.

    Args:
        predictions ([int]): predicted class indices
        class_to_idx (dict): mapping of indices to classess
        cat_to_name (dict): mapping of numbered classes to class names

    Returns:
        class_names ([str]): list of class names for predictions
                             Returns empty list if category_names are not
                             provided
    """
    class_dict = {val: key for key, val in class_to_idx.items()}
    class_idxs = [class_dict[pred] for pred in predictions]

    if not category_names:
        class_names = class_idxs
    else:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[idx] for idx in class_idxs]

    return class_names

def process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

    Args:
        image_path : path to input PIL image

    Returns:
        image_tensor (Tensor): processed image as torch.FloatTensor
    """
    IMG_SIZE = 224  # Size of images used for training
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_SDEV = [0.229, 0.224, 0.225]

    # Load PIL image
    image = Image.open(image_path)

    # Resize to 256 max dim
    if image.size[0] >= image.size[1]:
        image.thumbnail((256, image.size[1] * 256 // image.size[0]))
    else:
        image.thumbnail((image.size[0] * 256 // image.size[1], 256))

    # Center crop
    image = image.crop((
            (image.size[0] - IMG_SIZE) // 2,
            (image.size[1] - IMG_SIZE) // 2,
            (image.size[0] + IMG_SIZE) // 2 ,
            (image.size[1] + IMG_SIZE) // 2))
    # Convert to np.array and rescale color channels to 0-1
    image = np.array(image) / 255
    # Normalize image
    image = (image - np.array(IMG_MEAN)) / np.array(IMG_SDEV)
    # Rearrange to make color channel first dimension
    image = image.transpose(2, 0, 1)
    # Convert to toch.FloatTensor
    image_tensor = torch.from_numpy(
            np.expand_dims(image, axis=0)).type(torch.FloatTensor)

    return image_tensor
