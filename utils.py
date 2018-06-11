import os
import torch
from torchvision import datasets, transforms

def get_data_loaders(data_path):
    """
    Get the data-loaders pointing to a specific path.
    The folder should contain a train, validation and test folder with images inside.
    :returns: image datasets, data loaders
    """
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'

    data_transforms = get_data_transforms()

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(valid_dir, transform=data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    data_loaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    return image_datasets, data_loaders


def get_data_transforms():
    """
    Get the data transformations.
    Means and standard deviations are provided by the data source.
    :returns: dict
    """
    means = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]

    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(means, sd)
    ]

    return {
        'train': transforms.Compose([transforms.RandomRotation(35),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(224),
                                     ] + common_transforms
                                    ),
        'val': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   ] + common_transforms
                                  ),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    ] + common_transforms
                                   )
    }


def save_checkpoint(model, optimizer, data_loader, path):
    """
    Save a model checkpoint in a path.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoint_path = os.path.join(path, 'checkpoint.pth')
    model.class_to_idx = data_loader.dataset.class_to_idx

    checkpoint = {
        'model': model,
        'output_size': 102,
        'input_size': [3, 224, 224],
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_dict': optimizer.state_dict(),
        'batch_size': data_loader.batch_size,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, gpu):
    """
    Load a saved model from a path.
    :returns: model
    """
    if gpu:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.idx_to_class = dict([[v, k] for k, v in model.class_to_idx.items()])

    return model
