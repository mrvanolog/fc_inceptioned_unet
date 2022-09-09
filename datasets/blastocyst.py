import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import ImageFile
from skimage.draw import disk
from skimage.exposure import equalize_adapthist
from skimage.io import imread
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SFUDataset(data.Dataset):
    def __init__(self, dataset_dir: str, use_augmentations: bool=False,
                 mask: str='combined', add_masks: bool=False, transform: T=None):
        self.dataset_dir = Path(dataset_dir)
        self.use_augmentations = use_augmentations
        # options for mask: 'all', 'combined', 'te+zp', 'icm', 'te', 'zp'
        assert mask in ['all', 'te&zp', 'combined', 'te+zp', 'icm', 'te', 'zp']
        self.mask = mask
        self.add_masks = add_masks
        self.transform = transform

        self.img_dir = self.dataset_dir.joinpath('Images')
        self.mask_icm_dir = self.dataset_dir.joinpath('GT_ICM')
        self.mask_te_dir = self.dataset_dir.joinpath('GT_TE')
        self.mask_zp_dir = self.dataset_dir.joinpath('GT_ZP')
        self.f_names = [f.stem for f in self.img_dir.iterdir()]

        self.augmentations = [A.RandomRotate90(p=1),
                              A.Flip(p=1),
                              A.GaussNoise(var_limit=(0.05, 0.005), p=1),
                              A.CoarseDropout(max_holes=30, max_height=20, max_width=20, min_holes=10, p=1),
                              A.Affine(shear=(-20, 20), p=1),
                              A.Affine(scale=(1, 1.3), translate_percent=(0, 0.3), p=1),
                             ]
        self.to_grayscale = T.Grayscale(num_output_channels=1)
        self.normalize = T.Normalize(0, 1)

    def __len__(self):
        return len(self.f_names)

    def __getitem__(self, indx: int):
        # select sample
        name = self.f_names[indx]
        # compile file names
        img_name = f'{name}.BMP'
        icm_name = f'{name} ICM_Mask.bmp'
        te_name = f'{name} TE_Mask.bmp'
        zp_name = f'{name} ZP_Mask.bmp'

        # load input and targets
        x = imread(self.img_dir.joinpath(img_name))
        y = [
            imread(self.mask_icm_dir.joinpath(icm_name)),
            imread(self.mask_te_dir.joinpath(te_name)),
            imread(self.mask_zp_dir.joinpath(zp_name)),
        ]
        # apply histogram equalization to image
        x = equalize_adapthist(x)

        # create background and blastocell masks
        if self.add_masks:
            bgrnd = self._create_background(y[2])
            bcell = self._create_blastocoel(y[0], y[1], y[2], bgrnd)
            y.append(bgrnd)
            y.append(bcell)

        if self.use_augmentations:
            x, y = self._augment(x, y)

        x, y = self._process_data(x, y)

        # return a desired mask, if mask == 'all' then return a 3 channel tensor with all masks
        if self.mask == 'all':
            y = torch.cat(y, 0)
        if self.mask == 'te&zp':
            y = torch.cat([y[1], y[2]], 0)
        if self.mask == 'combined':
            comb_y = 0
            for _y in y:
                comb_y += _y
            y = comb_y
        if self.mask == 'te+zp':
            y = y[1] + y[2]
        if self.mask == 'icm':
            y = y[0]
        if self.mask == 'te':
            y = y[1]
        if self.mask == 'zp':
            y = y[2]

        # convert y values to binary (1, 0)
        y = (y > 0).float() * 1

        return x, y, name

    def _create_background(self, zp_mask):
        # create template for background mask
        h = zp_mask.shape[0]
        w = zp_mask.shape[1]
        bgrnd = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(zp_mask, bgrnd, (0, 0), 1)

        return bgrnd

    def _create_blastocoel(self, icm_mask, th_mask, zp_mask, bgrnd_mask):
        # create template for blastocoel mask
        cropped = bgrnd_mask[1:-1, 1:-1]
        combined_mask = icm_mask + th_mask + zp_mask + cropped
        h = combined_mask.shape[0]
        w = combined_mask.shape[1]
        bcell = np.zeros((h, w), np.uint8)

        # find any co-ord with blastocoel
        for x in range(0, h):
            for y in range(0, w):
                if combined_mask[x][y] == 0:
                    bcell[x][y] = 1

        return bcell

    def _augment(self, img, masks):
        # choose augmentations randomly
        aug = A.Compose(
            random.sample(self.augmentations, random.randint(0, len(self.augmentations)))
        )
        # Combine all the masks into a single image so we can run augmentation on it
        augmented = aug(image=img, masks=masks)
        return augmented['image'], augmented['masks']

    def _process_data(self, x, y):
        # typecasting
        x = TF.to_tensor(x).float()
        if x.shape[0] == 3:
            x = self.to_grayscale(x)  # handle occasional images with 3 channels
        y = [TF.to_tensor(_y) for _y in y]
        # y = [self.to_grayscale(_y) for _y in y]  # convert 3 channel .png to 1 channel

        # apply transformation
        if self.transform is not None:
            x = self.transform(x)
            y = [self.transform(_y).type(torch.float) for _y in y]

        # apply normalization
        x = self.normalize(x)

        return x, y


class ClinicDataset(data.Dataset):
    def __init__(self, dataset_dir: str, use_augmentations: bool=False,
                 mask: str='all', transform: T=None):
        self.dataset_dir = Path(dataset_dir)
        self.use_augmentations = use_augmentations
        # options for mask: 'all', '1', '2', '3', 'random'
        assert mask in ['all', '1', '2', '3', 'random']
        self.mask = mask
        self.transform = transform

        self.img_dir = self.dataset_dir.joinpath('Images')
        self.mask1_dir = self.dataset_dir.joinpath('Mask1')
        self.mask2_dir = self.dataset_dir.joinpath('Mask2')
        self.mask3_dir = self.dataset_dir.joinpath('Mask3')
        self.img_files = [f for f in self.img_dir.iterdir()]
        self.img_ext = self.img_files[0].suffix
        self.mask_ext = '.png'
        self.f_names = self._filter_f_names()
        self.clinic3 = False
        if self.dataset_dir.parent.name == 'clinic3':
            self.clinic3 = True

        self.augmentations = [A.RandomRotate90(p=1),
                              A.Flip(p=1),
                              A.GaussNoise(var_limit=0.005, p=1),
                              A.CoarseDropout(max_holes=30, max_height=20, max_width=20, min_holes=10, p=1)]
        self.to_grayscale = T.Grayscale(num_output_channels=1)
        self.normalize = T.Normalize(0, 1)

    def __len__(self):
        return len(self.f_names)

    def __getitem__(self, indx: int):
        # select sample
        name = self.f_names[indx]
        # compile file names
        img_name = f'{name}{self.img_ext}'
        mask_name = f'{name}{self.mask_ext}'

        # load input and targets
        x = imread(self.img_dir.joinpath(img_name))
        if self.clinic3:
            x = cv2.imread(str(self.img_dir.joinpath(img_name)), cv2.IMREAD_GRAYSCALE)
        y = [
            imread(self.mask1_dir.joinpath(mask_name)),
            imread(self.mask2_dir.joinpath(mask_name)),
            imread(self.mask3_dir.joinpath(mask_name)),
        ]
        # cut out the well from the image
        x, y = self._apply_circle_mask(x, y)

        # apply histogram equalization to image
        x = equalize_adapthist(x)

        if self.use_augmentations:
            x, y = self._augment(x, y)

        x, y = self._process_data(x, y)

        # return a desired mask, if mask == 'all' then return a 3 channel tensor with all masks
        if self.mask == 'all':
            y = torch.cat(y, 0)
        if self.mask == 'random':
            i = random.randint(0, 2)
            y = y[i]
        if self.mask == '1':
            y = y[0]
        if self.mask == '2':
            y = y[1]
        if self.mask == '3':
            y = y[2]

        # convert y values to binary (1, 0)
        y = (y > 0).float() * 1

        return x, y, name

    def _augment(self, img, masks):
        # choose augmentations randomly
        aug = A.Compose(
            random.sample(self.augmentations, random.randint(0, len(self.augmentations)))
        )
        # Combine all the masks into a single image so we can run augmentation on it
        augmented = aug(image=img, masks=masks)
        return augmented['image'], augmented['masks']

    def _process_data(self, x, y):
        # typecasting
        x = TF.to_tensor(x).float()
        if x.shape[0] == 3:
            x = self.to_grayscale(x)  # handle occasional images with 3 channels
        y = [TF.to_tensor(_y) for _y in y]
        y = [self.to_grayscale(_y) for _y in y]  # convert 3 channel .png to 1 channel

        # apply transformation
        if self.transform is not None:
            x = self.transform(x)
            y = [self.transform(_y).type(torch.float) for _y in y]

        # apply normalization
        x = self.normalize(x)

        return x, y

    def _filter_f_names(self) -> list:
        """Return a list of file names that are present
        in all Images, Mask1, Mask2 and Mask3 folders.
        """
        mask1_files = [f.stem for f in self.mask1_dir.iterdir()]
        mask2_files = [f.stem for f in self.mask2_dir.iterdir()]
        mask3_files = [f.stem for f in self.mask3_dir.iterdir()]
        f_names = []
        for f in self.img_files:
            f_name = f.stem
            if f_name in mask1_files and f_name in mask2_files and f_name in mask3_files:
                f_names.append(f_name)

        return f_names

    def _apply_circle_mask(self, x: np.ndarray, y: list):
        """Return an image and masks with applied circle mask.
        """
        # prepare circle mask
        img_shape = x.shape
        radius = img_shape[0]/2.5
        rr, cc = disk((img_shape[0]/2, img_shape[1]/2), radius)
        circle_mask = np.zeros(img_shape)
        circle_mask[rr, cc] = 1
        # apply mask to the image
        x = (x * circle_mask).astype(int)
        # crop image to get rid of zeros around circle mask
        aug = A.CenterCrop(int(radius*2), int(radius*2))
        cropped = aug(image=x, masks=y)

        return cropped['image'], cropped['masks']


class ClinicMultifocalDataset(data.Dataset):
    def __init__(self, dataset_dir: str, use_augmentations: bool=False,
                 mask: str='all', transform: T=None, use_circle_mask: bool=False):
        self.dataset_dir = Path(dataset_dir)
        self.use_augmentations = use_augmentations
        # options for mask: 'all'
        assert mask in ['all']
        self.mask = mask
        self.transform = transform
        self.use_circle_mask = use_circle_mask

        self.dir_paths = [d for d in self.dataset_dir.iterdir()]
        self.focals = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]
        self.to_grayscale = T.Grayscale(num_output_channels=1)
        self.normalize = T.Normalize(0, 1)

    def __len__(self):
        return len(self.dir_paths)

    def __getitem__(self, indx: int):
        # select sample
        dir_path = self.dir_paths[indx]
        # compile file names
        img_name = [n.name for n in dir_path.iterdir()][0]
        img_base = '_'.join(img_name.split('_')[:-1])
        file_names = [dir_path.joinpath(f'{img_base}_{f}.jpg') for f in self.focals]
        # I do it this way, bcause I need focal distances sorted in the correct oreder

        # load input and targets
        img_list = []
        for f in file_names:
            img_list.append(imread(f))
        x = np.stack(img_list, axis=-1)

        # cut out the well from the image
        if self.use_circle_mask:
            x = self._apply_circle_mask(x)

        # apply histogram equalization to image
        x = equalize_adapthist(x)

        x = self._process_data(x)

        return x, img_base

    def _apply_circle_mask(self, x: np.ndarray):
        """Return an image and masks with applied circle mask.
        """
        # prepare circle mask
        img_shape = x.shape
        radius = img_shape[0]/2.5
        rr, cc = disk((img_shape[0]/2, img_shape[1]/2), radius)
        circle_mask = np.zeros(img_shape)
        circle_mask[rr, cc] = 1
        # apply mask to the image
        x = (x * circle_mask).astype(int)
        # crop image to get rid of zeros around circle mask
        aug = A.CenterCrop(int(radius*2), int(radius*2))
        cropped = aug(image=x)

        return cropped['image']

    def _process_data(self, x):
        # typecasting
        x = TF.to_tensor(x).float()
        if x.shape[0] == 3:
            x = self.to_grayscale(x)  # handle occasional images with 3 channels

        # apply transformation
        if self.transform is not None:
            x = self.transform(x)

        # apply normalization
        x = self.normalize(x)

        return x
