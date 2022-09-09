from itertools import product
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils import data
from tqdm import tqdm
import wandb

from datasets.blastocyst import ClinicDataset, SFUDataset
from models.unet import UNet
from utils.trainer import Trainer
from utils.losses import DicePlusBCELoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_SIZE = 272
TRANSFORM = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE)])
SFU_MASK = 'combined'
CLINIC_MASK = 'random'
MASK = SFU_MASK
CHANNELS_OUT = 1

# clinic data
clinic1 = Path('/datasets/clinic1')
clinic2 = Path('/datasets/clinic2')
clinic3 = Path('/datasets/clinic3')
clinic4 = Path('/datasets/clinic4')
root_dirs = [clinic1, clinic2, clinic3, clinic4]
# sfu data
sfu = Path('/datasets/sfu/BlastsOnline')


def get_dataloader(type: str, use_augmentations: bool,
                   batch_size: int, shuffle: bool, sfu_only: bool):
    assert type in ['train', 'test', 'valid']

    sfu_dataset = SFUDataset(sfu.joinpath(f'{type}'), use_augmentations=use_augmentations,
                                   mask=SFU_MASK, transform=TRANSFORM)
    datasets = [sfu_dataset]
    if not sfu_only:
        for root_dir in root_dirs:
            dataset = ClinicDataset(root_dir.joinpath(f'{type}'),
                                      use_augmentations=use_augmentations,
                                      mask=CLINIC_MASK, transform=TRANSFORM)
            datasets.append(dataset)
    return data.DataLoader(dataset=data.ConcatDataset(datasets),
                           batch_size=batch_size, shuffle=shuffle)


def get_loss_fn(loss: str, dice_weight: float=None, smooth: float=None):
    if loss == 'BCE':
        return nn.BCEWithLogitsLoss()
    if loss == 'dice_loss+BCE':
        return DicePlusBCELoss(dice_weight, smooth)


def main(batch_size: int, n_epochs: int, n_blocks: int, start_filters: int, lr: float,
         loss: str, dice_weight: float=None, smooth: float=None, sfu_only: bool=True):
    # setup weights and biases
    config = {
        'learning_rate': lr,
        'epochs': n_epochs,
        'batch_size': batch_size,
        'channels_out': CHANNELS_OUT,
        'n_blocks': n_blocks,
        'start_filters': start_filters,
        'dice_weight': dice_weight,
        'smooth': smooth,
        'mask': MASK,
        'loss': loss,
    }
    wandb.init(project='PLACEHOLDER', entity='PLACEHOLDER', group=f'PLACEHOLDER',
               config=config, reinit=True)

    # get dataloaders
    dataloader_train = get_dataloader('train', use_augmentations=True, batch_size=batch_size,
                                      shuffle=True, sfu_only=sfu_only)
    dataloader_valid = get_dataloader('valid', use_augmentations=False, batch_size=batch_size,
                                      shuffle=False, sfu_only=sfu_only)
    dataloader_test = get_dataloader('test', use_augmentations=False, batch_size=batch_size,
                                      shuffle=False, sfu_only=sfu_only)

    # initialize model and learning environment
    unet = UNet(1, CHANNELS_OUT, n_blocks=n_blocks, start_filters=start_filters, sigmoid=True)
    unet.to(device)
    unet = nn.DataParallel(unet)

    loss_fn = get_loss_fn(loss, dice_weight, smooth)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    trainer = Trainer(unet, loss_fn, optimizer, n_epochs, device,
                      dataloader_train, dataloader_valid, dataloader_test, notebook=False)

    # train the model
    trainer.model_train(log=True)

    # find scores for validation dataset
    scores_valid = trainer.model_eval(data_type='valid', plot=False)

    wandb.log({'iou': np.mean(scores_valid)})


if __name__ == '__main__':
    params = {
        'lr': [5e-4],
        'n_epochs': [35, 70, 150],
        'batch_size': [30],
        'n_blocks': [5],
        'start_filters': [16, 24, 32],
        'dice_weight': [0.5, 1., 1.5, 2., 5.],
        'smooth': [1., 5.],
        'loss': ['dice_loss+BCE'],
    }
    keys, values = zip(*params.items())
    params_combinations = [dict(zip(keys, v)) for v in product(*values)]

    for params_comb in tqdm(params_combinations):
        main(**params_comb, sfu_only=True)

    # the best combination of parameters was found to be batch_size = 30,
    # n_epochs = 35, n_blocks = 5, lr = 5e-4
