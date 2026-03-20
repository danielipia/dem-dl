import torch
import h5py
import numpy as np
import os
import zarr
from torch.utils.data import Dataset

class DEMdata(Dataset):
    def __init__(self, data_path, split="train", args=None):
        file_name = f"{split}.h5"
        self.file_path = os.path.join(data_path, file_name)
        with h5py.File(self.file_path, 'r') as f:
            self.len = f['AIAPatches'].shape[0]
        self.args = args
        self.dataset = None

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')

        aia_patch = self.dataset['AIAPatches'][idx]
        dem_patch = self.dataset['DEMPatches'][idx]

        if self.args.preprocess == 'sqrt':
            aia_patch = np.maximum(0, aia_patch)**0.5
            dem_patch = np.maximum(0, dem_patch)**0.5
        elif self.args.preprocess == 'max':
            aia_patch = np.maximum(0, aia_patch)
            # make everything positive or 0 (there might be some small negative values really close to 0, this is good for numerical stability)
            dem_patch = np.maximum(0, dem_patch)
        elif self.args.preprocess == "":
            aia_patch = aia_patch
            dem_patch = dem_patch

        aia_patch = torch.from_numpy(aia_patch).float()
        dem_patch = torch.from_numpy(dem_patch).float()
        return aia_patch, dem_patch

    def __len__(self):
        return self.len
    
class ZarrAIAData(Dataset):
    def __init__(self, data_path, split="train", args=None):
        self.zarr_path = os.path.join(data_path, f"{split}.zarr")
        self.args = args
        self.dataset = None  # will open later
        self.len = None

    def __getitem__(self, idx):
        if self.dataset is None:
            # open once per worker
            self.dataset = zarr.open(self.zarr_path, mode='r')['AIAData']
        aia_patch = self.dataset['AIAPatches'][idx]
        aia_error = self.dataset['AIAErrors'][idx]

        if self.args.preprocess == 'sqrt':
            aia_patch = np.maximum(0, aia_patch)**0.5
            aia_error = np.maximum(0, aia_error)**0.5
        elif self.args.preprocess == 'max':
            aia_patch = np.maximum(0, aia_patch)
            aia_error = np.maximum(0, aia_error)
        # if preprocess is empty or None, leave as is

        aia_patch = torch.from_numpy(aia_patch).float()
        aia_error = torch.from_numpy(aia_error).float()
        return aia_patch, aia_error

    def __len__(self):
        if self.len is None:
            ds = zarr.open(self.zarr_path, mode='r')['AIAData']
            self.len = ds['AIAPatches'].shape[0]
        return self.len
    
class ZarrDEMData(Dataset):
    def __init__(self, data_path, split="train", args=None):
        self.zarr_path = os.path.join(data_path, f"{split}.zarr")
        self.args = args
        self.dataset = None # will open later
        self.aia_ds = None
        self.dem_ds = None
        self.len = None

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = zarr.open(self.zarr_path, mode='r')
            self.aia_ds = self.dataset['AIAData']
            self.dem_ds = self.dataset['DEMData']
            
        aia_patch = self.aia_ds['AIAPatches'][idx]
        dem_patch = self.dem_ds['DEMPatches'][idx]

        if self.args.preprocess == 'sqrt':
            aia_patch = np.maximum(0, aia_patch)**0.5
            dem_patch = np.maximum(0, dem_patch)**0.5
        elif self.args.preprocess == 'max':
            aia_patch = np.maximum(0, aia_patch)
            dem_patch = np.maximum(0, dem_patch)
        # if preprocess is empty or None, leave as is

        aia_patch = torch.from_numpy(aia_patch).float()
        dem_patch = torch.from_numpy(dem_patch).float()
        return aia_patch, dem_patch

    def __len__(self):
        if self.len is None:
            ds = zarr.open(self.zarr_path, mode='r')['AIAData']
            self.len = ds['AIAPatches'].shape[0]
        return self.len
    
class SimpleAIAData(Dataset):
    def __init__(self, data):
        self.len = None
        self.aia_ds = data[0]
        self.err_ds = data[1]

    def __getitem__(self, idx):

        aia_patch = self.aia_ds[idx]
        aia_error = self.err_ds[idx]

        aia_patch = torch.from_numpy(aia_patch).float()
        aia_error = torch.from_numpy(aia_error).float()
        return aia_patch, aia_error

    def __len__(self):
        if self.len is None:
            self.len = len(self.aia_ds)
        return self.len

class zarrDataset(Dataset):
    def __init__(self, data_path, split="train", args=None):
        self.name = "zarrDataset"
        self.data_path = data_path
        self.split = split
        self.args = args
        self.dataset = None  # will open later
        self.x = None
        self.y = None
        self.len = None
        self.mean = None
        self.std = None

    def _initialize_dataset(self):
        """lazy initialization"""
        if self.dataset is None:
            x_path = os.path.join(self.data_path, f"{self.split}_x.zarr")
            y_path = os.path.join(self.data_path, f"{self.split}_y.zarr")
            
            self.x = zarr.open(x_path, mode='r')
            self.y = zarr.open(y_path, mode='r')
            
            # validate shapes of x and y
            for i in range(1, 4):
                assert self.x.shape[i] == self.y.shape[i], f"Shape mismatch at dimension {i}"
            
            # placeholder normalization
            if self.mean is None:
                self.mean = np.zeros(self.x.shape[0]).astype(np.float32)
            if self.std is None:
                self.std = np.ones(self.x.shape[0]).astype(np.float32)
            
            self.dataset = True  # mark as initialized

    def __getitem__(self, idx):
        self._initialize_dataset()
        
        dataX = self.x[:, :, :, idx]
        dataY = self.y[:, :, :, idx]
        
        # normalization
        dataX = (dataX - self.mean[:, None, None]) / self.std[:, None, None]
        
        # preprocessing if specified
        if self.args and hasattr(self.args, 'preprocess'):
            if self.args.preprocess == 'sqrt':
                dataX = np.maximum(0, dataX)**0.5
                dataY = np.maximum(0, dataY)**0.5
            elif self.args.preprocess == 'max':
                dataX = np.maximum(0, dataX)
                dataY = np.maximum(0, dataY)
        
        dataX = torch.from_numpy(dataX).float()
        dataY = torch.from_numpy(dataY).float()

        # # zero out extended logT bins (19-25) that have extreme values from notrunc basis
        # if dataY.shape[0] > 18:
        #     dataY[19:] = 0.0

        # change infs to 0s
        dataX[torch.isinf(dataX)] = 0.0
        dataY[torch.isinf(dataY)] = 0.0
        
        return dataX, dataY

    def __len__(self):
        if self.len is None:
            x_path = os.path.join(self.data_path, f"{self.split}_x.zarr")
            x = zarr.open(x_path, mode='r')
            self.len = x.shape[3]  # N dimension
        return self.len

    def setNormalization(self, mean, std):
        """Set normalization parameters"""
        self.mean = mean.astype(np.float32).copy()
        self.std = std.astype(np.float32).copy()