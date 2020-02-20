import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from modelnet import ModelNet
from model import Model, compute_loss
from dgl.data.utils import download, get_download_dir

import numpy as np

from plydataset import PlyDataset

from functools import partial
import tqdm
import urllib
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
# data_filename = 'modelnet40-sampled-2048.h5'
# local_path = args.dataset_path or os.path.join(get_download_dir(), data_filename)

# if not os.path.exists(local_path):
#     download('https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/modelnet40-sampled-2048.h5', local_path)

CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        )



def evaluate(model, test_loader, dev):
    model.eval()

    fullpred = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label in tq:
                data, label = data.to(dev), label.to(dev).squeeze().long()
                logits = model(data)
                _, preds = logits.max(1)

                preds = preds.cpu().detach().numpy()
                fullpred.append(preds.reshape(-1,1))
                
    return np.vstack(fullpred)


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(20, [64, 64, 128, 256], [512, 512, 256], 40)
model = model.to(dev)
if args.load_model_path:
    model.load_state_dict(torch.load(args.load_model_path, map_location=dev))

print("Loading point clouds... ", end="")
test_dataset = PlyDataset("../data/MiniChallenge", "test", 128)
print("Done!")
test_loader = CustomDataLoader(test_dataset)

pred = evaluate(model, test_loader, dev)

outfile = "MiniDijon9.txt"

np.save(outfile + ".npy", pred)

np.savetxt('MiniDijon9.txt', pred, fmt='%d')
