from utils.ply import write_ply, read_ply
import numpy as np

from torch.utils.data import Dataset
import glob

from sklearn.neighbors import KDTree
import torch

class PlyDataset(Dataset):

    def __init__(self, path, mode = "train", n_points = 512):
        super().__init__()
        self.path = path
        self.mode = mode
        self.n_points = n_points

        self.folder = self.path + "/" + self.mode
        clouds_filename = glob.glob(self.folder + "/*.ply")
        clouds = [read_ply(filename) for filename in clouds_filename]
        self.points = [np.vstack((cloud['x'], cloud['y'], cloud['z'])).T for cloud in clouds]
        if mode == "test":
            self.labels = [np.zeros(cloud.shape[0]) for cloud in self.points]
        else:
            self.labels = [cloud['class'] for cloud in clouds]


        self.trees = [KDTree(cloud) for cloud in self.points]

        self.cloud_sizes = [cloud.shape[0] for cloud in self.points]
        self.size = sum(self.cloud_sizes)

    def __len__(self):
        return self.size
    
    def __getitem__(self, id):
        for cloud, label, tree in zip(self.points, self.labels, self.trees):
            if id < cloud.shape[0]:
                pt = cloud[id]
                closest = tree.query(pt.reshape(1,3), self.n_points, return_distance=False)
                return torch.tensor(cloud[closest[0]]), label[id]
            else:
                id -= cloud.shape[0]
        raise ValueError("Item index out of bound")