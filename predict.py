import argparse
import json
import numpy as np
import torch

from PIL import Image
from torch.autograd import Variable
from utils import load_checkpoint


def main():
    args = get_input_args()
    with open(args.category_names, 'r') as f:
        class_mappings = json.load(f)

    model = load_checkpoint(args.checkpoint, args.gpu)
    # Moves buffers and parameters to the GPU or GPU
    model.cuda() if args.gpu else model.cpu()

    probabilities, classes = predict(args.input, model, int(args.top_k), args.gpu)

    for probability, class_index in zip(probabilities, classes):
        print("{}: {}".format(class_mappings[class_index], probability))


def get_input_args():
    """
    Parse command line arguments
    :returns: Argument object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('input')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', default=5,
                        help='Number of top K most likely classes')
    parser.add_argument('--category_names', default='class_mappings.json',
                        help='Category names file path')
    parser.add_argument('--gpu', dest='gpu', default=False,
                        help='Use GPU for inference', action='store_true')

    return parser.parse_args()


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model
    :param image: Image
    :returns: np.array
    """
    resize_px = 256
    crop_px = 225
    size = resize_px, resize_px
    image.thumbnail(size, Image.ANTIALIAS)

    left = (resize_px - crop_px) / 2
    top = (resize_px - crop_px) / 2
    right = (resize_px + crop_px) / 2
    bottom = (resize_px + crop_px) / 2

    image_array = np.array(image.crop((left, top, right, bottom))) / 255
    normalized = (image_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return np.transpose(normalized, (2, 0, 1))


def predict(image_path, model, limit=5, gpu=False):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    :param image_path: string
    :param model: model
    :param limit: int Top K results that should be returned
    :param gpu: bool
    :returns: np.array, list
    """
    model.eval()

    image = torch.FloatTensor([process_image(Image.open(image_path))])
    image = Variable(image, volatile=True)
    if gpu:
        image = image.cuda()

    output = model.forward(image)
    probabilities, idx = torch.exp(output).topk(limit)
    labels = [model.idx_to_class[label] for label in idx.data[0]]
    top_probabilities = [p for p in probabilities.data[0]]

    return top_probabilities, labels


main()
