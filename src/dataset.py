import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    """
    Custom PyTorch Dataset for data handling both
    continuous numerical features and categorical features.
    """
    def __init__(self, X_processed, y, num_col_count, target_dtype=torch.float32):
        """
        Parameters:
        -----------
        X_processed : np.ndarray
          The combined processed feature array returned by Column Transformer.
        y : array-like or pd.Series, optional
          The target values/labels.
        num_col_count : int
          The number of continuous numerical columns.
        target_dtype : torch.dtype, default=torch.float32
          Tensor type for target (use torch.long for multi-class classification,
          torch.float32 for regression or binary tasks.)
        """

        # We first split combined feature array back into numerical and 
        # categorical parts.
        X_num = X_processed[:, :num_col_count]
        X_cat = X_processed[:, num_col_count:]

        # Next we convert numerical features to float tensors.
        self.num_features = torch.tensor(X_num, dtype=torch.float32)

        # We then handle categorical features and shift indices.
        # Recall that our OrdinalEncoder from preprocessing.py maps unknown values
        # to -1. These next few lines of code ensure that our range is now [0. N]
        # as opposed to [-1, N-1].
        if X_cat.shape[1] > 0:
            X_cat_shifted = X_cat + 1
            self.cat_features = torch.tensor(X_cat_shifted, dtype=torch.long)
        else:
            self.cat_features = torch.empty((X_processed.shape[0], 0), 
                                              dtype=torch.long)

        # If target values/labels are provided, we finally convert them into
        # tensors.
        if y is not None:
            self.y = torch.tensor(np.array(y), dtype=target_dtype)
        else:
            self.y = None

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.num_features)

    def __getitem__(self, idx):
        """Retrieves a single sample dictionary and its corresponding target."""
        features = {
            'num': self.num_features[idx],
            'cat': self.cat_features[idx]
        }

        if self.y is not None:
            return features, self.y[idx]
        else:
            return features

