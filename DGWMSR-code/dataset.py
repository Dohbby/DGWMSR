import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
cv2.setNumThreads(1)




class H5Dataset(Dataset):
    def __init__(self, h5_file):
        super(H5Dataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr=np.expand_dims(f['input'][idx], 0)
            lr=np.clip(lr, 0.0, 1.0)
            left=np.expand_dims(f['linput'][idx], 0)
            right=np.expand_dims(f['rinput'][idx], 0)
            gt=np.expand_dims(f['label'][idx], 0)
            return lr,left,right,gt  # 1,H,W

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['input'])