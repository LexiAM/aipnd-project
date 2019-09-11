import argparse

from data_utils import display_prediction,  process_image
from model_utils import load_checkpoint, classify_image


# Parse arguments
############################################
parser = argparse.ArgumentParser(
        description='Image classification model training application. '
        'Image class prediction.')
parser.add_argument(
        'image_path', action='store', type=str,
        help='Specify path to image for class prediction.')
parser.add_argument(
        'checkpoint', action='store', type=str,
        help='Classification model checkpoint file path.')
parser.add_argument(
        '--top_k', action='store', type=int, default=5,
        help='Return top_k most likely classes. dafault=5')
parser.add_argument(
        '--category_names', action='store', default='',
        help='Path to file with mapping of categories to class names '
        'in .json format')
parser.add_argument(
        '--gpu', action='store_true',
        help='Use GPU for training. default=True.')
args = parser.parse_args()

gpu = True if args.gpu else False

# Load, process and convert image to Tensor
############################################
image_tensor = process_image(image_path=args.image_path)

# load model
# model moved to device specified by gpu(bool) on load
############################################
model, _, _, _ = load_checkpoint(
        checkpoint_path=args.checkpoint, load_optimizer=False, gpu=gpu)

# Classify image
############################################
probabilities, predictions = classify_image(
        image_tensor=image_tensor, model=model, top_k=args.top_k,
        category_names=args.category_names, gpu=gpu)

# Show results
############################################
top_class = predictions[0]
top_prob = probabilities[0]
top_k = args.top_k
print(f'\nTop predicted class is "{top_class.capitalize()}" with '
      f'probability {top_prob:.4f}')
print(f'\nPredicted top {top_k} classes {predictions} with '
      f'probabilities {probabilities}')

# viasualize image with predicted classes and probabilities
# NOTE: Currently not working because display device is not available
display_prediction(
        image_path=args.image_path,
        probabilities=probabilities,
        predictions=predictions)
