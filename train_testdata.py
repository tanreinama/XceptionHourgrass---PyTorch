import os
import gc
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from sklearn.model_selection import train_test_split

from xceptionhourgrass import XceptionHourglass

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='train_files', help='input directory')
parser.add_argument('--val', type=float, default=0.01, help='validation ratio')
parser.add_argument('--epochs', default=20, type=int, help='max epochs')
parser.add_argument('--workers', default=8, type=int, help='num of threads')
args = parser.parse_args()

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
if not os.path.isdir('validation'):
    os.mkdir('validation')

filelist = os.listdir(args.dir+'/imgs')
train_files, valid_files = train_test_split(filelist, test_size=args.val)

class MyDataset(object):
    def __init__(self, filelist, valid=False):
        self.filelist = filelist
        self.valid = valid

    def __getitem__(self, idx):
        img = Image.open(args.dir + '/imgs/' + self.filelist[idx])
        msk = Image.open(args.dir + '/mask/' + self.filelist[idx])
        img, msk = np.array(img), np.array(msk)
        img = img.transpose((2,0,1)).astype(np.float32) / 255.
        msk = (msk != 0).reshape((1,msk.shape[0],msk.shape[1])).astype(np.float32)
        img, msk = torch.tensor(img), torch.tensor(msk)
        return img, msk

    def __len__(self):
        return len(self.filelist)

dataset = MyDataset(train_files)
dataset_v = MyDataset(valid_files, valid=True)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=args.workers)
data_loader_v = torch.utils.data.DataLoader(
    dataset_v, batch_size=1, shuffle=False, num_workers=args.workers)

model = XceptionHourglass()
model.cuda()
loss = nn.BCELoss()

dp = torch.nn.DataParallel(model)
params = [p for p in dp.parameters() if p.requires_grad]
optimizer = torch.optim.RMSprop(params, lr=2.5e-4,  momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=6,
                                               gamma=0.9)

def plot_sample(r, x, y, filename):
    r = r[0].detach().cpu().numpy()[0]
    x = x[0].detach().cpu().numpy().transpose((1,2,0))
    y = y[0].detach().cpu().numpy()[0]
    fig = plt.figure(figsize=(9, 3), facecolor="w")
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x)
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y, cmap='gist_gray', vmin=0, vmax=1)
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(r, cmap='gist_gray', vmin=0, vmax=1)
    plt.savefig(filename)
    plt.clf()
    plt.close()

loss_plot = [[],[]]

for epoch in range(args.epochs):
    total_loss = []
    prog = tqdm(data_loader, total=len(data_loader))
    for X, y in prog:
        X = X.cuda()
        y = y.cuda()

        losses = loss(dp(X), y)

        prog.set_description("loss:%05f"%losses)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss.append(losses.detach().cpu().numpy())

    prog, X, y, losses = None, None, None, None,
    torch.cuda.empty_cache()
    gc.collect()
    total_loss_v = []
    prog = tqdm(data_loader_v, total=len(data_loader_v))
    for i, (X, y) in enumerate(prog):
        X = X.cuda()
        y = y.cuda()

        res = model(X)
        losses = loss(res, y)

        prog.set_description("loss:%05f"%losses)

        total_loss_v.append(losses.detach().cpu().numpy())
        plot_sample(res, X, y, 'validation/%d.png'%i)

    prog, X, y, res, losses = None, None, None, None, None
    torch.cuda.empty_cache()
    gc.collect()

    lr_scheduler.step()

    loss_plot[0].append(np.mean(total_loss))
    loss_plot[1].append(np.mean(total_loss_v))
    print("epoch:",epoch,"train_loss:",loss_plot[0][-1],"valid_loss:",loss_plot[1][-1])

    torch.save(model.state_dict(), "checkpoint/checkpoint-%d.model"%epoch)

plt.plot(loss_plot[0])
plt.plot(loss_plot[1])
plt.savefig("validation/loss.png")
