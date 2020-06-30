import numpy as np
import torch
import json

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, log_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.log_path = log_path
        self.log_entries = {}

        if log_path:
            config = {"patience": patience, "delta": delta}
            self.log_entries["config"] = config
            
            with open(log_path, 'w') as json_file:
                json.dump(self.log_entries, json_file)



    def __call__(self, val_loss, epoch):
        score = -val_loss
        save = False

        if self.best_score is None:
            self.best_score = score
            save = True
            if self.verbose:
                print('Validation loss decreased ({0:.6f} --> {1:.6f}).'.format(self.val_loss_min, val_loss))
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save = True
            if self.verbose:
                print('Validation loss decreased ({0:.6f} --> {1:.6f}).'.format(self.val_loss_min, val_loss))
            self.val_loss_min = val_loss
            self.counter = 0

        if self.log_path:
        		entry = {"counter": self.counter, "val_loss_min": self.val_loss_min, "val_loss": val_loss}
        		self.log_entries["{:03d}".format(epoch)] = entry

                with open(self.log_path, 'w') as json_file:
                    json.dump(self.log_entries, json_file)

        return save
