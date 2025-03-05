import os
import argparse
import pandas as pd
import torch
import warnings
import torch.optim as optim
from Dataset import MyDataSet
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from Model import swin_tiny_patch4_window7_224 as create_model
from utils_II_CRC import read_split_data, train_one_epoch, evaluate

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main(args):
    global train_loss, train_cidx, val_loss, val_cidx
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    (train_images_path, train_images_fustat, train_images_futime,
     val_images_path, val_images_fustat, val_images_futime) = read_split_data(args.data_path,
                                                                              args.seed,
                                                                              args.ALL_cohort,
                                                                              args.timedata)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224, 224)),
                                     transforms.ColorJitter(brightness=0.25, contrast=0.25),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                     ]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((224, 224)),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_fustat=train_images_fustat,
                              images_futime=train_images_futime,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_fustat=val_images_fustat,
                            images_futime=val_images_futime,
                            transform=data_transform["val"])
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs / 4, eta_min=0.0000001,
                                                           last_epoch=-1)

    for epoch in range(args.epochs):
        train_loss, train_cidx = train_one_epoch(model=model,
                                                 optimizer=optimizer,
                                                 data_loader=train_loader,
                                                 device=device,
                                                 epoch=epoch,
                                                 scaler=scaler,
                                                 scheduler=scheduler)

        val_loss, val_cidx = evaluate(model=model,
                                      data_loader=val_loader,
                                      device=device,
                                      epoch=epoch)
        tags = ["train_loss", "train_cidx", "val_loss", "val_cidx", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_cidx, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_cidx, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        torch.save(model.state_dict(),
                   "H:/II_CRC/weight/CRC-20241123-seed{}-tr{}-va{}-model-{}.pth".
                   format(args.seed,
                          round(train_cidx, 3),
                          round(val_cidx, 3),
                          epoch))
    return train_loss, train_cidx, val_loss, val_cidx


if __name__ == '__main__':
    data_list = pd.DataFrame()
    Dir = r'H:/II_CRC'
    timedata = pd.read_csv(os.path.join(Dir, "Match_II_CRC_DATA.csv"), index_col=0, encoding='gbk')
    num_class = 1
    timedata = timedata.iloc[:, :2]

    ALL_cohort = [['Center_I', 'Center_II', 'Center_III', 'Center_IV']]

    for seed in [1]:
        for epoch in [60]:
            for lr in [4e-6]:
                print('------------------ seed = ', seed, '------------------')
                print('------------------ lr = ', lr, '------------------')
                parser = argparse.ArgumentParser()
                parser.add_argument('--num_classes', type=int, default=num_class)
                parser.add_argument('--epochs', type=int, default=epoch)
                parser.add_argument('--batch-size', type=int, default=32)
                parser.add_argument('--lr', type=float, default=lr)
                parser.add_argument('--seed', type=float, default=seed)
                parser.add_argument('--ALL_cohort', type=float, default=ALL_cohort)
                parser.add_argument('--timedata', type=float, default=timedata)
                parser.add_argument('--data-path', type=str, default=Dir)

                parser.add_argument('--weights', type=str,
                                    default=r"weights/swin_tiny_patch4_window7_224.pth",
                                    help='initial weights path')
                parser.add_argument('--freeze-layers', type=bool, default=False)
                parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

                opt = parser.parse_args()
                train_loss, train_cidx, val_loss, val_cidx = main(opt)
                new_list = pd.DataFrame({"lr": [lr],
                                         "seed": [seed],
                                         "train_loss": [train_loss],
                                         "train_cidx": [train_cidx],
                                         "val_loss": [val_loss],
                                         "val_cidx": [val_cidx]
                                         })

                data_list = pd.concat([data_list, new_list], axis=0)
                data_list.to_csv(os.path.join(Dir, "results.csv"), encoding='gbk', index=False)
