import os
import sys
import random
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.cuda.amp import autocast
from Model import coxloss, c_index
import math


def read_split_data(root: str, seed: int, ALL_cohort, timedata, val_rate: float = 0.3):
    random.seed(seed)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_images_path = []
    train_images_fustat = []
    train_images_futime = []
    val_images_path = []
    val_images_fustat = []
    val_images_futime = []
    for model_cohort in ALL_cohort:
        root2 = os.path.join(root, str(model_cohort))
        images = [os.path.join(root2, i) for i in os.listdir(root2)]
        images = [name.replace('Smax+0', 'Smax') for name in images]
        images = [name.replace('Smax+1', 'Smax') for name in images]
        images = [name.replace('Smax-1', 'Smax') for name in images]
        images = list(set(images))
        images.sort()
        val_path_initial = random.sample(images, k=int(len(images) * val_rate))
        val_path_1 = [name.replace('Smax', 'Smax+0') for name in val_path_initial]
        val_path_2 = [name.replace('Smax', 'Smax+1') for name in val_path_initial]
        val_path_3 = [name.replace('Smax', 'Smax-1') for name in val_path_initial]
        images_1 = [name.replace('Smax', 'Smax+0') for name in images]
        images_2 = [name.replace('Smax', 'Smax+1') for name in images]
        images_3 = [name.replace('Smax', 'Smax-1') for name in images]
        val_path = val_path_1 + val_path_2 + val_path_3
        images_else = images_1 + images_2 + images_3
        images_finally = [item for item in images_else if item not in val_path]
        for img_path in val_path:
            name = img_path.split('/')[-1]
            val_images_path.append(os.path.join(img_path))
            val_images_fustat.append(timedata.loc[name, "fustat"])
            val_images_futime.append(timedata.loc[name, "futime"])
        for img_path in images_finally:
            name = img_path.split('/')[-1]
            train_images_path.append(os.path.join(img_path))
            train_images_fustat.append(timedata.loc[name, "fustat"])
            train_images_futime.append(timedata.loc[name, "futime"])

    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    Train_data = pd.DataFrame(
        {'images': train_images_path, 'fustat': train_images_fustat, 'futime': train_images_futime})
    Train_data.to_csv(root + "/" + str(seed) + "train_results.csv", encoding='gbk', index=False)

    Val_data = pd.DataFrame({'images': val_images_path, 'fustat': val_images_fustat, 'futime': val_images_futime})
    Val_data.to_csv(root + "/" + str(seed) + "val_results.csv", encoding='gbk', index=False)

    return train_images_path, train_images_fustat, train_images_futime, val_images_path, val_images_fustat, val_images_futime


def make_predict_data(root, predict_cohort, timedata):
    test_images_path = []
    test_images_fustat = []
    test_images_futime = []
    for part in predict_cohort:
        root2 = os.path.join(root, part)
        images = [os.path.join(root2, i) for i in os.listdir(root2)]
        for img_path in images:
            if True:
                name = img_path.split('/')[-1]
                test_images_path.append(os.path.join(img_path))
                test_images_fustat.append(timedata.loc[name, "fustat"])
                test_images_futime.append(timedata.loc[name, "futime"])

    print("{} images for testing.".format(len(test_images_path)))
    assert len(test_images_path) > 0, "number of test images must greater than 0."
    return test_images_path, test_images_fustat, test_images_futime


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, scheduler):
    model.train()
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)

    Cidxfunction = c_index()
    sum_cidx = torch.zeros(1).to(device)
    lossfunction = coxloss()
    sum_loss = torch.zeros(1).to(device)
    criterion = nn.CrossEntropyLoss()
    list_data = pd.DataFrame(columns=["images_path", "risk_pred", 'fustats', 'futimes'])
    for step, data in enumerate(data_loader):
        images_path, images, fustats, futimes = data
        with autocast():
            risk_pred = model(images.to(device))
        list_new_data = pd.DataFrame({"images_path": images_path,
                                      "risk_pred0": risk_pred.cpu().detach().numpy()[:, 0],
                                      "fustat": fustats,
                                      "futime": futimes})
        list_data = list_data._append(list_new_data)

        risk_pred0 = torch.tensor(list_data['risk_pred0'].astype(float).values, dtype=torch.float32)
        fustat = torch.tensor(list_data['fustat'].astype(float).values, dtype=torch.long)
        futime = torch.tensor(list_data['futime'].astype(float).values, dtype=torch.float32)
        sum_cidx = Cidxfunction(risk_pred0, futime, fustat)
        loss = lossfunction(risk_pred.to(device), futimes.to(device), fustats.to(device), model)
        re = "Cindex"
        sum_loss += loss
        scaler.scale(loss).backward()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, {}: {:.3f}, lr*10^3: {:.5f}".format(epoch,
                                                                                               sum_loss.item() / (
                                                                                                       step + 1),
                                                                                               re,
                                                                                               sum_cidx,
                                                                                               optimizer.param_groups[
                                                                                                   0]["lr"] * 1000)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            break
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    return sum_loss.item(), sum_cidx.item()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)

    Cidxfunction = c_index()
    sum_cidx = torch.zeros(1).to(device)
    lossfunction = coxloss()
    sum_loss = torch.zeros(1).to(device)
    criterion = nn.CrossEntropyLoss()
    list_data = pd.DataFrame(columns=["images_path", "risk_pred", 'fustats', 'futimes'])
    for step, data in enumerate(data_loader):
        images_path, images, fustats, futimes = data
        with autocast():
            risk_pred = model(images.to(device))
        list_new_data = pd.DataFrame({"images_path": images_path,
                                      "risk_pred0": risk_pred[:, 0].cpu().detach().numpy(),
                                      "fustats": fustats,
                                      "futimes": futimes})
        list_data = list_data._append(list_new_data)

        risk_pred0 = torch.tensor(list_data['risk_pred0'].astype(float).values, dtype=torch.float32)
        fustats = torch.tensor(list_data['fustats'].astype(float).values, dtype=torch.long)
        futimes = torch.tensor(list_data['futimes'].astype(float).values, dtype=torch.float32)
        re = "Cindex"
        sum_loss = lossfunction(risk_pred0, futimes, fustats, model)
        sum_cidx = Cidxfunction(risk_pred0, futimes, fustats)

        data_loader.desc = "[Valid epoch {}] loss: {:.3f}, {}: {:.3f}, lr*10^3: {:.5f}".format(
            epoch,
            sum_loss.item() / (step + 1),
            re,
            sum_cidx,
            0)

    return sum_loss.item(), sum_cidx.item()


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
