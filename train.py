import argparse

from data_utils import load_data
from model_utils import (
        create_model,  # builds model based on selected architecture
        create_optimizer,   # creates optimizer for model's final Classifier layers
        load_checkpoint,  # loads model from checkpoint
        plot_history,  # plots training loss and accuracy history
        save_checkpoint,   # saves model checkpoint
        train_model,  # trains model
        test_model)  # test model performance on a test data set


# Parse command line input
###########################
parser = argparse.ArgumentParser(
        description='Image classification model training application. '
        'Classification model training.')
parser.add_argument(
        'data_directory', action='store', type=str, default='flowers/',
        help='Specify directory containing training, validation, and test data.'
        ' default="flowers/"')
parser.add_argument(
        '--save_path', action='store', type=str,
        default='checkpoints/checkpoint.pth',
        help='Provide save checkpoint file path. '
        'default="checkpoints/checkpoint.pth"')
parser.add_argument(
        '--resume_checkpoint', action='store', type=str, default='',
        help='Provide filepath for model checkpoint for continued training. '
        'Model will be loaded from the provided checkpoint path.\n'
        'If resume_checkpoint argument is not empty, model hyperparameter '
        'arguments --arch, --learning_rate, --hidden_units, and --drop_p '
        'are ignored.')
parser.add_argument(
        '--arch', action='store', type=str, default='vgg16',
        choices=['alexnet', 'densenet161', 'resnet18', 'vgg16'],
        help='Choose pre-trained model architecture from the following '
        'supported options: alexnet, densenet161, resnet18, vgg16. '
        'default=vgg16.')
parser.add_argument(
        '--learning_rate', action='store', type=float, default=0.01,
        help='Set model training learning rate.')
parser.add_argument('--hidden_units', nargs='*', type=int,
        help='Enter number of hidden units for final fc classifier layers. '
        'Empty argument is allowed and is set by default. Separate units for'
        ' mutliple hidden layers by space. Example:\n'
        '"--hidden_units 1024 512 256"')
parser.add_argument(
        '--drop_p', action='store', type=float, default=0.5,
        help='Set Droupout probability. default=0.5')
parser.add_argument(
        '--epochs', action='store', type=int, default=2,
        help='Set number of training epochs. default=2')
parser.add_argument(
        '--gpu', action='store_true', help='Use GPU for training.')
args = parser.parse_args()

gpu = True if args.gpu else False

# Load data
###########################
(dataloaders, class_to_idx) = load_data(args.data_directory)

# Create model
###########################
if args.resume_checkpoint:  # resume_checkpoint path is provided
    # load checkpoint
    (model, optimizer, epoch, history) = load_checkpoint(
             checkpoint_path=args.resume_checkpoint,
             load_optimizer=True, gpu=gpu)
    start_epoch = epoch + 1
else:
    # create new model and optimizer
    model = create_model(
            arch=args.arch, class_to_idx=class_to_idx,
            hidden_units=args.hidden_units, drop_p=args.drop_p)
    optimizer = create_optimizer(model=model, lr=args.learning_rate)
    start_epoch = 1
    history = None

# Train model
###########################
history, best_epoch = train_model(
        dataloaders=dataloaders, model=model,
        optimizer=optimizer, gpu=gpu, start_epoch=start_epoch,
        epochs=args.epochs, train_history=history)

# Check performance on test data set
# test_acc = test_model(
#         dataloader=dataloaders['test'], model=model, gpu=gpu)
# print(f'\nModel achieved accuracy of {(test_acc * 100):.2f}% on Test data set.')

# Plot training history
plot_history(history)
# NOTE: plot_history() is currently not working on Udacity workspace because
# display device is not available

# Save checkpoint
###########################
save_checkpoint(
        save_path=args.save_path, epoch=best_epoch, model=model,
        optimizer=optimizer, history=history)
