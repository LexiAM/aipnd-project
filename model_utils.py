import matplotlib.pyplot as plt
import time
import copy
import numpy as np
import os
import sys

from workspace_utils import active_session
from data_utils import prediction_class_names

# Import Pytorch modules
import torch
from torch import optim, nn
from torchvision import models


def  classify_image(image_tensor, model, top_k, category_names, gpu):
    """Classifies image

    Args:
        image_tensor (torch.FloatTensor): processed input image Tensor
        model (): trained torch model
        gpu (bool): True use GPU, use CPU otherwise

    Returns:
        probs ([float]): list of prediction probabilities
        preds ([int]): list of predicted class indices
    """
    device = select_device(gpu=gpu)
    print(f'\nClassifying image using {device}...')
    model.eval()
    image_tensor = image_tensor.to(device)  # Send image to device
    with torch.no_grad():
        output = torch.exp(model(image_tensor))
    probs, preds = output.topk(top_k, dim=1)
    probs, preds = probs.tolist()[0], preds.tolist()[0]

    # Provide real class names if mapping dictionary path is provided in the args
    class_names = prediction_class_names(
            predictions=preds,
            class_to_idx=model.class_to_idx,
            category_names=category_names)

    return probs, preds, class_names


def create_model(arch='vgg16', class_to_idx=None, hidden_units=[], drop_p=0.5):
    """Builds neural network with pre-trained torchvision model base and
       custom fully connected classifier final layer(s) with attached
       information: base torchvision architecture name, training class to index
       mapping dictionary, output size, hidden layer sizes, and dropout
       probability

        Args:
            arch (str): pre-trained torchvision model name.
                        Supported models:
                        ['alexnet', 'densenet161', 'resnet18', 'vgg16']
            output_size (int): size of output layer
            hidden_units ([int]): array holding integer sizes of hidden
                                  layers
            drop_p (float (0, 1)): dropout probability
            class_to_idx (dict): mapping of training classes to indices

        Returns:
            model (nn model): pre-trained model with custom classifier
                              Attributes:
                                  arch (str)
                                  class_to_idx (dict)
                                  output_size (int)
                                  hidden_layers ([int])
                                  drop_p (float)


    """
    print('\nBuilding model...')

    # Supported models dict(): keys = model name, values = classifier input size
    supported_models = {'alexnet': 9216,
                        'densenet161': 2208,
                        'resnet18': 512,
                        'vgg16': 25088}

    # Define model classifier input size to match pre-trained model architecture
    try:
        input_size = supported_models[arch]
    except KeyError:
        print(f'Exception: Architecture {arch} is not one of supported'
              f' model architectures: {list(supported_models.keys())}.')
        sys.exit(1)

    # Determine model output size by number of classes in class_to_idx
    output_size = len(class_to_idx)

    # Load pre-trained model
    model = getattr(models, arch)(pretrained=True)

    # Freeze pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create custom classifier
    ##########################

    # Combine all layer sizes
    layer_sizes = [input_size]
    if hidden_units:
        layer_sizes += hidden_units
    layer_sizes += [output_size]

    # Build layers
    classifier = nn.Sequential()
    for idx, (inp, out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        # Add FC linear layer
        classifier.add_module('fc' + str(idx), nn.Linear(inp, out))
        # Add ReLU and Dropout after FC layers except last FC layer
        if idx < (len(layer_sizes) - 2):
            classifier.add_module('relu' + str(idx), nn.ReLU())
            classifier.add_module('dropout' + str(idx), nn.Dropout(drop_p))
    # Return LogSoftMax output
    classifier.add_module('output', nn.LogSoftmax(dim=1))
    ##########################

    # Replace pre-trained model classifer layer(s) with custom classifier
    if arch == 'resnet18':
        model.fc = classifier
    else:
        model.classifier = classifier

    # Attach relevant model information for future reference
    model.arch = arch
    model.class_to_idx = class_to_idx
    model.output_size = output_size
    model.hidden_units = hidden_units
    model.drop_p = drop_p

    print('\nModel created:')
    print(model)

    return model


def create_optimizer(model, lr=0.001):
    """Returns Adam optimizer for provided model.
       Optimizes only final fc/classifier layer(s) parameters

    Args:
        model (): model created with create_model()
        lr (float): learning rate

    Returns:
        optimizer (torch.optim.Optimizer)
    """
    print('\nBuilding optimizer...')
    if model.arch == 'resnet18':
        params = model.fc.parameters()
    else:
        params = model.classifier.parameters()

    optimizer = optim.Adam(params, lr=lr)

    return  optimizer


def load_checkpoint(checkpoint_path, load_optimizer=False, gpu=False):
    """Loads checkpoint and rebuilds pretrained model and optinal
       optimizer on device specified by gpu(bool)

    Args:
        checkpoint_path (str): checkpoint file path
        load_optmizer (bool): True: creates an optimizer and loads state_dict
                              False: does not create an optimizer
        gpu (bool): True attempt to load on GPU. If GPU is not available, user is notified

    Returns:
        model (): rebuilt pre-trained model
        optimizer (optim.Optimizer, None): optimizer  if load_optimizer==True, None otherwise
        epoch (int): save epoch number
        history (dict): training history
    """

    print('\nLoading model checkpoint...')

    # select device and check if gpu is available if requested
    device = select_device(gpu)

    # load checkpoint.
    # IMPORTANT Assumes checkpoint is saved on CPU
    if gpu:
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    epoch = checkpoint['epoch']
    history = checkpoint['history']

    # Build model
    model = create_model(arch=checkpoint['arch'],
                         class_to_idx=checkpoint['class_to_idx'],
                         hidden_units=checkpoint['hidden_units'],
                         drop_p=checkpoint['drop_p'])

    # Load model weights
    print('\nLoading state dictionary...')
    model.load_state_dict(checkpoint['model_state_dict'])

    # sending model to device
    model.to(device)

    # Create optimizer
    if load_optimizer:
        optimizer = create_optimizer(model=model)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = None

    print(f'\nModel checkpoint epoch {epoch} successfully loaded.')

    return model, optimizer, epoch, history


# Select GPU or CPU device
def select_device(gpu):
    """Selects GPU or CPU device.
       If GPU is requested but not available throws AssertionError.

    Args:
        gpu (bool): selects GPU if True, otherwise CPU

    Returns:
        device (torch.device): 'cpu' or 'cuda:0'
    """
    # Check if GPU is requested it is available
    if gpu:
        assert torch.cuda.is_available(), ('Error: Requested GPU, '
                                           'but GPU is not available.')

    # Select device
    device = torch.device('cuda:0') if gpu else torch.device('cpu')

    return device


def train_model(dataloaders, model, optimizer, gpu=True,
                start_epoch=1, epochs=2, train_history=None):
    """Trains PyTorch model. At the end of training loads model.state_dict
       and optimizer.state_dict for the epoch with highest validation accuracy

    Args:
        model (PyTorch model)
        dataloaders (dict): keyes = ['train', 'valid', 'test'];
                            values = Dataloaders
        criterion (): training loss criterion
        optimizer (optim. Optimizer)
        gpu (bool): trains on GPU if True, otherwise CPU
        start_epoch (int): start epoch number
        epochs (int): number of epochs to train counting from start_epoch
        train_history (dict): training and validation losses and accuracies
                             history = {
                                'train': {'loss': [], 'acc': []},
                                'valid': {'loss': [], 'acc': []}}
                            If start_epoch <= len(history['..']['..'][]),
                            history truncates at the start_epoch to override
                            history of epochs after and including the
                            start_epoch

    Returns:
        history (dict(dict)): nested dictionary containing training and
                              validation losses and accuracies
        best_epoch (int): training epoch with highest validation accuracy
    """
    criterion = nn.NLLLoss()

    # Setup historical and best state tracking
    ##########################################

    # Track losses and accuracies
    if train_history is None:  # create new history if doesn't exist
        history = {
            'train': {'loss': [], 'acc': []},
            'valid': {'loss': [], 'acc': []}
        }
    else:
        history = train_history
        # truncate history at the start_epoch to override history of epochs
        # after and including start_epoch
        history['train']['loss'] = history['train']['loss'][0: start_epoch - 1]
        history['valid']['loss'] = history['valid']['loss'][0: start_epoch - 1]
        history['train']['acc'] = history['train']['acc'][0: start_epoch - 1]
        history['valid']['acc'] = history['valid']['acc'][0: start_epoch - 1]

    # Best validation accuracy and epoch
    # IMPORTANT: In case of checkpoint loading continuation of previous
    # training assumes that best (highest validation accuracy)
    # model/optimizer state is saved
    if not history['valid']['acc']:
        best_acc = 0  # intialize with 0 if there is no training history
        best_epoch = 0
    else:
        # initialize with highest historical validation acc and epoch
        best_acc = max(history['valid']['acc'])
        best_epoch = history['valid']['acc'].index(
            max(history['valid']['acc'])) + 1

    # Best model/optimizer states
    # IMPORTANT: In case of checkpoint loading continuation of previous
    # training assumes that best (highest validation accuracy)
    # model/optimizer state is saved
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    # Select training device
    device = select_device(gpu)
    print(f'\nStarting model training from epoch {start_epoch} on {device} ...')

    # Train Model
    ############################
    train_start = time.time()  # start training timer

    model.to(device)  # Send model to device

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f'\nEpoch {epoch}/{start_epoch + epochs - 1}:'
              f'\n---------------------')

        # Train and validate in each epoch
        for phase in ['train', 'valid']:
            # Start phase timer
            phase_start = time.time()

            # Set model mode
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Reset running statistics
            running_loss = 0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                # Send inputs, labels to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Reset parameter gradients for training
                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward propagation
                    logps = model(inputs)
                    # Caluclate loss
                    loss = criterion(logps, labels)

                    if phase == 'train':
                        # Back propagation
                        loss.backward()
                        # Update model parameters
                        optimizer.step()

                # Update running statistics
                running_loss += loss.item() * inputs.size(0)

                # Running count of correctly identified classes
                ps = torch.exp(logps)  # probabilities
                _, predictions = ps.topk(1, dim=1)   # top predictions
                # Number of correctly classified inputs
                equals = predictions == labels.view(*predictions.shape)
                running_corrects += torch.sum(
                    equals.type(torch.FloatTensor)).item()

            # Calculate phase statistics
            phase_loss = running_loss / len(dataloaders[phase].dataset)
            history[phase]['loss'].append(phase_loss)

            phase_acc = running_corrects / len(dataloaders[phase].dataset)
            history[phase]['acc'].append(phase_acc)

            # Save best model weights if accuracy improved
            if phase == 'valid' and phase_acc > best_acc:
                best_epoch = epoch
                best_acc = phase_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_optimizer_state_dict = copy.deepcopy(
                    optimizer.state_dict())

            # Display training updates for the epoch
            phase_duration = time.time() - phase_start
            print(f'{phase.upper()} completed in {phase_duration:.0f}s. '
                  f'Loss: {phase_loss:.4f}, Acc: {phase_acc:.4f}')

    # Set model/optimizer.state_dict to best_model/optimizer_state_dict
    model.load_state_dict(best_model_state_dict)
    optimizer.load_state_dict(best_optimizer_state_dict)

    # Display training results
    train_duration = time.time() - train_start
    print(f'\nTraining complete in {(train_duration // 60):.0f}m '
          f'{(train_duration % 60):.0f}s. '
          f'Best Validation Acc: {best_acc:.4f}, '
          f'achieved after epoch {best_epoch}')

    return history, best_epoch
    return device


def plot_history(history):
    """Plots training historical training and validation accuracies and losses

    Args:
        history (dict): training and validation losses and accuracies history.
                        {'train': {'loss': [], 'acc': []},
                         'valid': {'loss': [], 'acc': []}}
    """
    fig, ax1 = plt.subplots()

    # Correctly number epochs starting from 1
    epochs = np.arange(1, len(history['train']['loss']) + 1)

    # Plot losses
    ax1.plot(epochs, history['train']['loss'], 'g-')
    ax1.plot(epochs, history['valid']['loss'], 'b-')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Validation Loss'],bbox_to_anchor=(0.6,0.2))

    # Plot accuracies
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train']['acc'], 'y-')
    ax2.plot(epochs, history['valid']['acc'], 'r-')
    ax2.set_ylabel('Accuracy')
    ax2.legend(['Training Accuracy', 'Validation Accuracy'],
               bbox_to_anchor=(0.5,0.9))

    plt.legend(frameon=False)
    plt.show()


def save_checkpoint(save_dir, epoch, model, optimizer, history):
    """Saves PyTorch checkpoint on CPU with provided model, optimizer and
       training history

    Args:
        save_dir (str): checkpoint save directory
        epoch (int): training epoch being saved
        model: model for which checkpoint is being saved
        optimizer (torch.optim.Optimizer)
        history (dict(dict)): nested dictionary containing training and
                              validation losses and accuracies
    """
    print(f'\nSaving best training epoch {epoch} checkpoint...')

    # create checkpoint filepath
    #############################
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]

    # make save_dir if necessary
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = save_dir + '/checkpoint.pth'

    model.to('cpu')  # no need to save on gpu

    # Setup checkpoint
    checkpoint = {
        'arch': model.arch,
        'output_size': model.output_size,
        'class_to_idx': model.class_to_idx,
        'hidden_units': model.hidden_units,
        'drop_p': model.drop_p,
        'history': history,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    # Save
    torch.save(checkpoint, filepath)

    # Notify user
    file_size = os.path.getsize(filepath)
    print(f'\nCheckpoint saved: {(file_size / 1e9):.2f}Gb\n')


def test_model(dataloader, model, gpu=False):
    """Tests model performance on a data from dataloader and prints accuracy.

    Args:
        dataloader (DataLoader)
        model (torchvision model)
        gpu (bool): Use GPU if True, otherwise CPU

    Returns:
        test_acc (float): model prediction accuracy
    """
    print('\nEvaluating model performance on a Test data set...')
    # Set model evaluation mode and send to torch.device
    model.eval()
    device = select_device(gpu)
    model.to(device)

    # setup loss
    criterion = nn.NLLLoss()

    # Run validation on TEST data
    running_corrects = 0
    for inputs, labels in dataloader:
        # send inputs, labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            # forward propagation
            logps = model(inputs)
            # caluclate loss
            loss = criterion(logps, labels)

        # Running accuracy
        ps = torch.exp(logps)  # probabilities
        _, predictions = ps.topk(1, dim=1)   # top predictions
        equals = predictions == labels.view(*predictions.shape)
        running_corrects += torch.sum(equals.type(torch.FloatTensor)).item()

    # Calculate accuracy
    test_acc = running_corrects / len(dataloader.dataset)

    return test_acc
