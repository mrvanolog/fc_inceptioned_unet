import torch
from torch import nn


class LossBase():

    def __init__(self, thresh: float=0.5):
        self.thresh = thresh

    def tp_tn_fp_fn(self, gt: torch.Tensor, pred: torch.Tensor) -> tuple:
        """Returns the numpber of true positives, true negatives,
        false positives, false negatives for each of the channels.
        """
        pred = pred.contiguous()
        pred_thresh = (pred > self.thresh).float() * 1
        gt = gt.contiguous()

        if gt.shape[0] == 1:
            gt = torch.cat((gt, torch.zeros_like(gt), torch.zeros_like(gt)), dim=0)

        tp = (pred * gt).sum(dim=(1, 2))
        fp = pred.sum(dim=(1, 2)) - tp
        fn = gt.sum(dim=(1, 2)) - tp
        tn = pred.shape[-2] * pred.shape[-1] - (tp + fp + fn)

        return tp, tn, fp, fn

    def calc_iou(self, gt: torch.Tensor, pred: torch.Tensor, mean: bool=False) -> torch.Tensor:
        """Calculates Intersection over Union (IOU) or Jaccard Index.

        Args:
            gt (torch.Tensor): Ground truth mask
            pred (torch.Tensor): Prediction mask
            mean (bool): return data for each channel or the mean value. Defaults to False

        Returns:
            torch.Tensor: Value/values betwen 0 and 1
        """
        tp, _, fp, fn = self.tp_tn_fp_fn(gt, pred)
        iou = tp / (tp + fp + fn)
        if mean:
            iou = iou.mean()

        return iou

    def calc_dsc(self, gt: torch.Tensor, pred: torch.Tensor, mean: bool=False) -> float:
        """Calculates Dice Coefficient (DSC).

        Args:
            gt (torch.Tensor): Ground truth mask
            pred (torch.Tensor): Prediction mask
            mean (bool): return data for each channel or the mean value. Defaults to False

        Returns:
            torch.Tensor: Value/values betwen 0 and 1
        """
        tp, _, fp, fn = self.tp_tn_fp_fn(gt, pred)
        dice_score = (2. * tp) / (2. * tp + fp + fn)
        if mean:
            dice_score = dice_score.mean()

        return dice_score

    def calc_vs(self, gt: torch.Tensor, pred: torch.Tensor, mean: bool=False) -> float:
        """Calculates Volumetric Similarity (VS).

        Args:
            gt (torch.Tensor): Ground truth mask
            pred (torch.Tensor): Prediction mask
            mean (bool): return data for each channel or the mean value. Defaults to False

        Returns:
            torch.Tensor: Value/values betwen 0 and 1
        """
        tp, _, fp, fn = self.tp_tn_fp_fn(gt, pred)
        v_sim = 1 - torch.abs(fn - fp) / (2. * tp + fp + fn)
        if mean:
            v_sim = v_sim.mean()

        return v_sim

    def calc_rvd(self, gt: torch.Tensor, pred: torch.Tensor, mean: bool=False) -> float:
        """Calculates Relative Volume Difference (RVD).

        Args:
            gt (torch.Tensor): Ground truth mask
            pred (torch.Tensor): Prediction mask
            mean (bool): return data for each channel or the mean value. Defaults to False

        Returns:
            torch.Tensor: Value/values betwen 0 and 1
        """
        tp, _, fp, fn = self.tp_tn_fp_fn(gt, pred)
        rvd = torch.abs(fn - fp) / (fn + tp)
        if mean:
            rvd = rvd.mean()

        return rvd


class DicePlusBCELoss(LossBase):

    def __init__(self, dice_weight: float=None, smooth: float=None, rescale: bool=False):
        """Dice loss plus Binary Cross Entropy Loss.

        Args:
            dice_weight (float, optional): Weight of the dice loss in a sum with BCE loss.
                BCE loss weight is always equal to 1. Defaults to 1.0.
            smooth (float, optional): Smoothing parameter for dice loss.
                Added to both numerator and denomenator. Defaults to 1.0.
            rescale (bool, optional): Whether to rescale predictions between 0 and 1.
                Defaults to False.
        """
        super().__init__()

        self.dice_weight = dice_weight or 1.0
        self.smooth = smooth or 1.0
        self.rescale = rescale

        self.bce_loss = nn.BCELoss()

    def __call__(self, pred, target):
        if self.rescale:
            pred1 = pred - pred.min(2, keepdim=True)[0]
            pred2 = pred1 / pred1.max(2, keepdim=True)[0]
            pred = pred2

        return self.dice_weight * self._dice_loss(pred, target) + self.bce_loss(pred, target)

    def _dice_loss(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=0).sum(dim=0)
        union = (pred + target).sum(dim=0).sum(dim=0)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        soft_dice_loss = 1 - dice_score

        return soft_dice_loss.mean()


class DicePlusVSPlusBCELoss(LossBase):

    def __init__(self, thresh: float=0.5, bce_weight: float=None, dice_weight: float=None,
                 vs_weight: float=None, rescale: bool=False):
        super().__init__(thresh)

        self.bce_w = bce_weight or 1.0
        self.dice_w = dice_weight or 1.0
        self.vs_w = vs_weight or 1.0
        self.rescale = rescale

        self.bce_loss = nn.BCELoss()

    def __call__(self, pred, target):
        if self.rescale:
            pred = pred - pred.min(2, keepdim=True)[0]
            pred = pred / pred.max(2, keepdim=True)[0]

        dice = 1 - self.calc_dsc(target, pred, mean=True)
        vs = 1 - self.calc_vs(target, pred, mean=True)

        return self.dice_w * dice + self.vs_w * vs + self.bce_w * self.bce_loss(pred, target)
