"""
Early stopping tracker.
Author: JiaWei Jiang

This file contains the definition of early stopping to prevent overfit
or boost modeling efficiency.
"""
from typing import Optional


class EarlyStopping(object):
    """Monitor whether the specified metric improves or not. If metric
    doesn't improve for the `patience` epochs, then the training and
    evaluation processes will stop early.

    Parameters:
        patience: tolerance for number of epochs when model can't
                  improve the specified score (e.g., loss, metric)
        mode: performance determination mode, the choices can be:
            {'min', 'max'}
        tr_loss_thres: stop training immediately once training loss
            reaches this threshold
    """
    
    best_score: float
    stop: bool
    wait_count: int
    
    def __init__(
        self, 
        patience: int = 10, 
        mode: str = "min", 
        tr_loss_thres: Optional[float] = None
    ):
        self.patience = patience
        self.mode = mode
        self.tr_loss_thres = tr_loss_thres
        self._setup()

    def step(self, score: float) -> None:
        """Update states of es tracker.
        
        Parameters:
            score: specified score in the current epoch
        
        Return:
            None
        """
        if self.tr_loss_thres is not None:
            if score <= self.tr_loss_thres:
                self.stop = True
        else:
            score_adj = score if self.mode == "min" else -score
            if score_adj < self.best_score:
                self.best_score = score_adj
                self.wait_count = 0
            else:
                self.wait_count += 1

            if self.wait_count >= self.patience:
                self.stop = True

    def _setup(self) -> None:
        """Setup es tracker."""
        if self.mode == "min":
            self.best_score = 1e18
        elif self.mode == "max":
            self.best_score = -1 * 1e-18
        self.stop = False
        self.wait_count = 0
