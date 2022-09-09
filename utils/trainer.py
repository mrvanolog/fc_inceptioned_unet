from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import wandb


class Trainer:
    """Helper class that contains all necessary methods
    for model training and evaluation.
    """
    def __init__(
        self, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, n_epochs: int,
        device: str, train_dataloader: data.DataLoader, valid_dataloader: data.DataLoader,
        test_dataloader: data.DataLoader=None, scheduler: optim.lr_scheduler.StepLR=None,
        thresh: float=None, notebook: bool=False,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.device = device
        self.thresh = thresh
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.metrics = ['IoU', 'DSC', 'VS', 'RVD']
        self.metric_fn_dict = {
            'iou': self.calc_iou,
            'dsc': self.calc_dsc,
            'vs': self.calc_vs,
            'rvd': self.calc_rvd,
        }

        # colour maps for data visualization
        c_white = colors.colorConverter.to_rgba('white', alpha=0)
        c_red = colors.colorConverter.to_rgba('red', alpha=1)
        c_green = colors.colorConverter.to_rgba('green', alpha=1)
        c_blue = colors.colorConverter.to_rgba('blue', alpha=1)
        c_magenta = colors.colorConverter.to_rgba('magenta', alpha=1)
        c_khaki = colors.colorConverter.to_rgba('khaki', alpha = 1)
        cmap_red = colors.LinearSegmentedColormap.from_list('rb_cmap', [c_white,c_red], 512)
        cmap_green = colors.LinearSegmentedColormap.from_list('rb_cmap', [c_white,c_green], 512)
        cmap_blue = colors.LinearSegmentedColormap.from_list('rb_cmap', [c_white,c_blue], 512)
        cmap_magenta = colors.LinearSegmentedColormap.from_list('rb_cmap', [c_white,c_magenta], 512)
        cmap_khaki = colors.LinearSegmentedColormap.from_list('rb_cmap', [c_white,c_khaki], 512)
        self.colormaps = [cmap_red, cmap_green, cmap_blue, cmap_magenta, cmap_khaki]

        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm

    def model_train(self, log: bool=False) -> None:
        """Train the model.
        """
        progress = self.tqdm(range(self.n_epochs), leave=False)
        for _ in progress:
            train_loss = self._train(self.train_dataloader, self.loss_fn, self.optimizer)
            valid_loss = self._test(self.valid_dataloader, self.loss_fn)

            if log:
                wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss})
            if self.scheduler is not None:
                self.scheduler.step()

            progress.set_description(f'Training loss: {np.round(train_loss, 4)}')

    def model_eval(self, metric: str='iou', data_type: str='valid', plot: bool=False,
                   display_name: bool=False, mean_score: bool=True) -> List[float]:
        """Evaluate the model with a metric of choice.

        Args:
            metric (str, optional): Metric to use. Options are: 'iou'. Defaults to 'iou'
            data_type (str, optional): Which dataset to use. Options are: 'valid', 'test'.
                Defaults to 'valid'
            plot (bool, optional): Whether or not to visualise predictions. Defaults to False
            display_name (bool, optional): Whether or not to print the name of a file. Defaults to False
            mean_score (bool, optional): Whether or not to display metric for each channel or
                the mean value for all channels. Defaults to True

        Returns:
            List[float]: List of scores for each datapoint in the dataset
        """
        if metric == 'iou':
            metric_fn = self.calc_iou

        if data_type == 'valid':
            data = self.valid_dataloader
        if data_type == 'test':
            data = self.test_dataloader

        self.model.eval()

        total_scores = {}
        for metric in self.metrics:
            total_scores[metric] = []
        for x, y, name in data:
            with torch.no_grad():
                input, target = x.to(self.device), y.to(self.device)
                preds = self.model(input)
                preds_thresh, thresh = self._get_preds_thresh(preds, y)

                for i in range(len(x)):
                    for metric in self.metrics:
                        metric_fn = self.metric_fn_dict[metric.lower()]
                        score = metric_fn(target[i], preds_thresh[i], mean_score).cpu()
                        total_scores[metric].append(score)

                    if not plot:
                        continue

                    f, axarr = plt.subplots(1, 4, figsize = (11, 3))
                    axarr[0].imshow(x[i].permute(1, 2, 0))
                    if y.shape[1] not in [1, 3] or preds.shape[1] not in [1, 3]:
                        for c in range(y.shape[1]):
                            axarr[1].imshow(y[i][c], cmap=self.colormaps[c])
                        for c in range(preds.shape[1]):
                            axarr[2].imshow(preds[i][c].cpu(), cmap=self.colormaps[c])
                            axarr[3].imshow(preds_thresh[i][c].cpu(), self.colormaps[c])
                    else:
                        axarr[1].imshow(y[i].permute(1, 2, 0))
                        axarr[2].imshow(preds[i].cpu().permute(1, 2, 0))
                        axarr[3].imshow(preds_thresh[i].cpu().permute(1, 2, 0))

                    title = ''
                    if display_name:
                        title = f'{name[i]}'
                    f.suptitle(title)
                    axarr[0].set_title('Image')
                    axarr[1].set_title('Ground Truth')
                    axarr[2].set_title('Prediction')
                    axarr[3].set_title(f'Prediction with Threshold ({self.thresh or thresh.cpu().numpy()})')
                    plt.show()
                    for metric in self.metrics:
                        print(f'{metric}: {total_scores[metric][-1].round(decimals=4)}')

                    print('-----------' * 4)

        return total_scores

    @staticmethod
    def calc_threshold(data: torch.Tensor, thresh_type: str='mean') -> float:
        if thresh_type == 'mean':
            agr_fn = torch.mean
        if thresh_type == 'median':
            agr_fn = torch.median

        return agr_fn(data, (0, 2, 3))

    @staticmethod
    def tp_tn_fp_fn(gt: torch.Tensor, pred: torch.Tensor) -> tuple:
        """Returns the numpber of true positives, true negatives,
        false positives, false negatives for each of the channels.
        """
        pred = pred.contiguous()
        gt = gt.contiguous()

        if gt.shape[0] == 1:
            tensors = [gt]
            tensors.extend([torch.zeros_like(gt) for _ in range(pred.shape[0] - 1)])
            gt = torch.cat(tensors, dim=0)

        tp = (pred * gt).sum(dim=(1, 2))
        fp = pred.sum(dim=(1, 2)) - tp
        fn = gt.sum(dim=(1, 2)) - tp
        tn = pred.shape[-2] * pred.shape[-1] - (tp + fp + fn)

        return tp, tn, fp, fn

    @staticmethod
    def calc_iou(gt: torch.Tensor, pred: torch.Tensor, mean: bool=False) -> torch.Tensor:
        """Calculates Intersection over Union (IOU) or Jaccard Index.

        Args:
            gt (torch.Tensor): Ground truth mask
            pred (torch.Tensor): Prediction mask
            mean (bool): return data for each channel or the mean value. Defaults to False

        Returns:
            torch.Tensor: Value/values betwen 0 and 1
        """
        tp, _, fp, fn = Trainer.tp_tn_fp_fn(gt, pred)
        iou = tp / (tp + fp + fn)
        if mean:
            iou = iou.mean()

        return iou

    @staticmethod
    def calc_dsc(gt: torch.Tensor, pred: torch.Tensor, mean: bool=False) -> float:
        """Calculates Dice Coefficient (DSC).

        Args:
            gt (torch.Tensor): Ground truth mask
            pred (torch.Tensor): Prediction mask
            mean (bool): return data for each channel or the mean value. Defaults to False

        Returns:
            torch.Tensor: Value/values betwen 0 and 1
        """
        tp, _, fp, fn = Trainer.tp_tn_fp_fn(gt, pred)
        dice_score = (2. * tp) / (2. * tp + fp + fn)
        if mean:
            dice_score = dice_score.mean()

        return dice_score

    @staticmethod
    def calc_vs(gt: torch.Tensor, pred: torch.Tensor, mean: bool=False) -> float:
        """Calculates Volumetric Similarity (VS).

        Args:
            gt (torch.Tensor): Ground truth mask
            pred (torch.Tensor): Prediction mask
            mean (bool): return data for each channel or the mean value. Defaults to False

        Returns:
            torch.Tensor: Value/values betwen 0 and 1
        """
        tp, _, fp, fn = Trainer.tp_tn_fp_fn(gt, pred)
        v_sim = 1 - torch.abs(fn - fp) / (2. * tp + fp + fn)
        if mean:
            v_sim = v_sim.mean()

        return v_sim

    @staticmethod
    def calc_rvd(gt: torch.Tensor, pred: torch.Tensor, mean: bool=False) -> float:
        """Calculates Relative Volume Difference (RVD).

        Args:
            gt (torch.Tensor): Ground truth mask
            pred (torch.Tensor): Prediction mask
            mean (bool): return data for each channel or the mean value. Defaults to False

        Returns:
            torch.Tensor: Value/values betwen 0 and 1
        """
        tp, _, fp, fn = Trainer.tp_tn_fp_fn(gt, pred)
        rvd = torch.abs(fn - fp) / (fn + tp)
        if mean:
            rvd = rvd.mean()

        return rvd

    def _train(self, dataloader, loss_fn, optimizer) -> float:
        self.model.train()
        losses = []

        for x, y, _ in dataloader:
            input, target = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            preds = self.model(input)
            loss = loss_fn(preds, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        return np.mean(losses)

    def _test(self, dataloader, loss_fn) -> float:
        self.model.eval()
        losses = []

        for x, y, _ in dataloader:
            input, target = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                preds = self.model(input)
                loss = loss_fn(preds, target)
                losses.append(loss.item())

        return np.mean(losses)

    def _get_preds_thresh(self, preds, y) -> tuple:
        if self.thresh is not None:
            thresh = self.thresh
            preds_thresh = (preds > self.thresh).float() * 1
        else:
            preds_thresh = torch.zeros_like(preds)
            thresh = self.calc_threshold(preds)
            for ch in range(y.shape[1]):
                preds_thresh[:,ch,:,:] = (preds[:,ch,:,:] > thresh[ch]).float() * 1

        return preds_thresh, thresh