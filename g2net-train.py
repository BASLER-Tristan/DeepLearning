#!/usr/bin/env python
# coding: utf-8
# In[3]:


import os
import numpy as np
import pandas as pd
import h5py
import timm
import matplotlib.pyplot as plt
import seaborn
import time

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from timm.scheduler import CosineLRScheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
di = "D:train_cleaned"


# In[4]:


torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


train_labels = pd.read_csv("DATA/train_labels.csv")
submission = pd.read_csv("DATA/sample_submission.csv")


# In[6]:


train_labels.head()


# In[7]:


train_labels["target"].value_counts()


# In[8]:


# Removing the negative labels
train_labels = train_labels[train_labels.target >= 0]
train_labels.target.value_counts()


# In[9]:


submission.head()


# In[11]:


count = 0
train_files = []
test_files = []

for dirname, _, filenames in os.walk("D:train_cleaned/"):
    print(filenames)
    for filename in filenames:
        count += 1
        path = os.path.join(dirname, filename)

        if "test" in dirname:
            test_files.append(path)

        if "train" in dirname:
            train_files.append(path)

        if count % 1000 == 0:
            print(count, "data files loaded")


# In[23]:


train_files[0]


# In[24]:


# defining a configuration
class CFG:
    model_name = "tf_efficientnet_b4_ns"
    target_size = 1
    transform = True
    flip_rate = 0.5
    fre_shift_rate = 1.0
    time_mask_num = 1
    freq_mask_num = 2
    nfold = 5
    is_cross_validate = True
    batch_size = 8
    epochs = 25
    num_workers = 0
    lr = 1e-3
    weight_decay = 1e-6
    train = True
    seed = 42
    score_method = "roc_auc_score"
    scheduler_type = "CosineLRScheduler"
    optimizer_type = "Adam"
    loss_type = "BCEWithLogitsLoss"
    max_grad_norm = 1000
    lr_max = 4e-4
    epochs_warmup = 1.0


# In[25]:


def get_criterion():
    if CFG.loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    if CFG.loss_type == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()


# In[26]:


def get_optimizer(model):
    if CFG.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr_max, weight_decay=CFG.weight_decay, amsgrad=False)
    if CFG.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr_max, weight_decay=CFG.weight_decay)
    return optimizer


# In[27]:


def get_scheduler(optimizer, warmup, nsteps):
    if CFG.scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, step_size=2, gamma=0.1, verbose=True)
    if CFG.scheduler_type == "CosineLRScheduler":
        scheduler = CosineLRScheduler(
            optimizer, warmup_t=warmup, warmup_lr_init=0.0, warmup_prefix=True, t_initial=(nsteps - warmup), lr_min=1e-6
        )
    return scheduler


# In[28]:


def get_score(y_true, y_pred):
    if CFG.score_method == "roc_auc_score":
        score = roc_auc_score(y_true, y_pred)
    if CFG.score_method == "accuracy_score":
        score = accuracy_score(y_true, y_pred)
    return score


# In[29]:


def transform(img):
    transforms_time_mask = nn.Sequential(torchaudio.transforms.TimeMasking(time_mask_param=10),)
    transforms_freq_mask = nn.Sequential(torchaudio.transforms.FrequencyMasking(freq_mask_param=10),)
    if np.random.rand() <= CFG.flip_rate:  # horizontal flip
        img = np.flip(img, axis=1).copy()
    if np.random.rand() <= CFG.flip_rate:  # vertical flip
        img = np.flip(img, axis=2).copy()
    if np.random.rand() <= CFG.fre_shift_rate:  # vertical shift
        img = np.roll(img, np.random.randint(low=0, high=img.shape[1]), axis=1)

    img = torch.from_numpy(img)

    for _ in range(CFG.time_mask_num):  # tima masking
        img = transforms_time_mask(img)
    for _ in range(CFG.freq_mask_num):  # frequency masking
        img = transforms_freq_mask(img)

    return img


# In[30]:


class Dataset(torch.utils.data.Dataset):
    """
    dataset = Dataset(data_type, df)

    img, y = dataset[i]
      img (np.float32): 2 x 360 x 128
      y (np.float32): label 0 or 1
    """

    def __init__(self, data_type, df):
        self.data_type = data_type
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id

        img = np.empty((2, 360, 128), dtype=np.float32)
        filename = "%s/%s.hdf5" % (di, file_id)

        with h5py.File(filename, "r") as f:
            g = f[file_id]

            for ch, s in enumerate(["H1", "L1"]):
                a = g[s]["SFTs"][:, :4096] * 1e22  # Fourier coefficient complex64

                p = a.real ** 2 + a.imag ** 2  # power
                p /= np.mean(p)  # normalize
                p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128

                img[ch] = p

        if CFG.transform:
            img = transform(img)

        return img, y


# In[32]:


file_id = "02887d232"


# In[33]:


dataset = Dataset("train", train_labels)
img, y = dataset[10]

plt.figure(figsize=(8, 3))
plt.title("Spectrogram")
plt.xlabel("time")
plt.ylabel("frequency")
plt.imshow(img[0, 300:360])  # zooming in for dataset[10]
plt.colorbar()
plt.show()


# In[34]:


dataset


# In[35]:


len(dataset)


# In[36]:


class Model(nn.Module):
    def __init__(self, name, *, pretrained=False):
        """
        name (str): timm model name, e.g. tf_efficientnet_b2_ns
        """
        super().__init__()

        # Use timm
        model = timm.create_model(name, pretrained=pretrained, in_chans=2)

        clsf = model.default_cfg["classifier"]
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.fc = nn.Linear(n_features, 1)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


# In[37]:


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()  # switch to training mode
    nbatch = len(train_loader)
    running_loss = 0
    count = 0
    tb = time.time()

    pbar = tqdm(train_loader, total=len(train_loader))
    pbar.set_description(f"[{epoch+1}/{CFG.epochs}] Train")

    for ibatch, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)
        running_loss += loss.item() * labels.shape[0]
        count += labels.shape[0]

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        optimizer.step()
        scheduler.step(epoch * nbatch + ibatch + 1)
        optimizer.zero_grad()

    lr_now = optimizer.param_groups[0]["lr"]
    dt = (time.time() - tb) / 60
    train_dict = {"loss": running_loss / count, "lr_now": lr_now, "time": dt}

    return train_dict


def valid_fn(valid_loader, model, criterion, device, compute_score=True):

    tb = time.time()
    model.eval()  # switch to evaluation mode
    preds = []
    y_all = []
    running_loss = 0
    count = 0

    pbar = tqdm(valid_loader, total=len(valid_loader))
    pbar.set_description("Validation")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)
        running_loss += loss.item() * labels.shape[0]
        count += labels.shape[0]
        # record accuracy
        y_all.append(labels.cpu().detach().numpy())
        preds.append(y_preds.sigmoid().to("cpu").numpy())

    del loss, images, labels, y_preds

    y_ground = np.concatenate(y_all)
    y_pred = np.concatenate(preds)
    score = get_score(y_ground, y_pred) if compute_score else None
    val_loss = running_loss / count

    val_dict = {"loss": val_loss, "score": score, "y": y, "y_pred": y_pred, "time": (time.time() - tb) / 60}

    return val_dict


# In[ ]:


# Train loop
def train_loop(data):

    if CFG.is_cross_validate:

        kfold = StratifiedKFold(n_splits=CFG.nfold, random_state=42, shuffle=True)
        for ifold, (idx_train, idx_test) in enumerate(kfold.split(data["id"], data["target"])):

            print("Fold %d/%d" % (ifold, CFG.nfold))
            torch.manual_seed(CFG.seed + ifold + 1)
            # create dataset
            train_dataset = Dataset("train", data.iloc[idx_train])
            valid_dataset = Dataset("train", data.iloc[idx_test])

            # create dataloaider
            train_loader = DataLoader(
                train_dataset,
                batch_size=CFG.batch_size,
                shuffle=True,
                num_workers=CFG.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
                pin_memory=True,
                drop_last=False,
            )

            # create model and transfer to device

            model = Model(CFG.model_name, pretrained=True)
            model.to(device)

            # select optimizer, scheduler and criterion
            optimizer = get_optimizer(model)
            nbatch = len(train_loader)
            warmup = CFG.epochs_warmup * nbatch
            nsteps = CFG.epochs * nbatch
            scheduler = get_scheduler(optimizer, warmup, nsteps)
            criterion = get_criterion()

            time_val = 0.0
            tb = time.time()
            # start training
            for epoch in range(CFG.epochs):
                # train
                train_dict = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
                # validation
                val_dict = valid_fn(valid_loader, model, criterion, device)

                time_val += val_dict["time"]

                print(
                    "Epoch = %d train_loss = %.4f val_loss = %.4f val_score = %.4f lr = %.2e time = %.2f min"
                    % (
                        epoch + 1,
                        train_dict["loss"],
                        val_dict["loss"],
                        val_dict["score"],
                        train_dict["lr_now"],
                        train_dict["time"],
                    )
                )
            dt = (time.time() - tb) / 60
            print("Training done %.2f min total, %.2f min val" % (dt, time_val))

            output_file = "model%d.pytorch" % ifold
            torch.save(model.state_dict(), output_file)
            print(output_file, "written")


# In[ ]:


# main
def main():
    if CFG.train:
        # train
        train_loop(train_labels)


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:


# train the model on whole data and save it
