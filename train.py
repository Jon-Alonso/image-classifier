import argparse
from collections import OrderedDict

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models

from utils import get_data_loaders, save_checkpoint
from .trainer import Trainer


def main():
    args = get_input_args()
    model = build_model(args.arch, int(args.hidden_units))
    image_datasets, data_loaders = get_data_loaders(args.data_path)

    # Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.lr))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trainer = Trainer(scheduler, optimizer, criterion)
    # Print stats every epoch
    trainer.debug = True

    # Moves parameters and buffers to the GPU
    if args.gpu:
        model = model.cuda()

    for epoch in range(args.epochs):
        trainer.train(model, data_loaders['train'], len(image_datasets['train']))
        trainer.validate(model, data_loaders['val'], len(image_datasets['val']))

    save_checkpoint(model, optimizer, data_loaders['train'], args.save_dir)


def get_input_args():
    """
    Parse command line arguments
    :returns: Argument object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', action="store")
    parser.add_argument("--save_dir", default=".",
                        help="Checkpoint directory path")
    parser.add_argument("--arch", default="vgg13", help="Model architecture")
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate",
                        action="store", dest="lr", )
    parser.add_argument("--hidden_units", default=512,
                        help="Number of hidden units")
    parser.add_argument("--epochs", default=5, help="Number of epochs")
    parser.add_argument("--gpu", default=False,
                        help="Use GPU for training", action='store_true')

    return parser.parse_args()


def build_model(model_name, hidden_units):
    """
    Create a Torch Vision model. https://pytorch.org/docs/master/torchvision/models.html
    :param model_name: string
    :param hidden_units: int
    :returns: model
    """
    try:
        model = getattr(models, model_name)(pretrained=True)
    except:
        raise ValueError('Invalid model name' + model_name)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = create_classifier(
        model.classifier[0].in_features, hidden_units)

    return model


def create_classifier(input_units, hidden_units):
    """
    Create a classifier for a model with ReLU activation function,
    dropout and the specified units.
    :param input_units: int
    :param hidden_units: int
    :returns: Tensor
    """
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))


main()
