import os
import time
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable

from utils.data import get_loader
from utils.func import label_edge_prediction, AvgMeter
from model.EFNet import EFNet

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='batch size')
parser.add_argument('--trainsize', type=int, default=352, help='input size')
parser.add_argument('--trainsets', type=str, nargs='+', default='DUTS-TRAIN', help='training  dataset')
parser.add_argument('--data_root', type=str, default='data', help='training  dataset root path')
parser.add_argument('--save_dir', type=str, default='log', help='training  dataset root path')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha number')
opt = parser.parse_args()

print(opt)

save_path = os.path.join(opt.save_dir, str(int(time.time())))

if not os.path.exists(save_path):
    os.makedirs(save_path)

os.system('cp -r model {}'.format(save_path))
os.system('cp -r utils {}'.format(save_path))
os.system('cp -r train_EFNet.py {}'.format(save_path))
os.system('cp -r test_EFNet.py {}'.format(save_path))

LOG_FOUT = open(os.path.join(save_path, 'log_train.txt'), 'w')
LOG_FOUT.write(str(opt) + '\n')
LOG_FOUT.flush()


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


trainfolder = {
    "DUT":
        {"image_root": 'train_data/train_images/',
         "gt_root": "train_data/train_masks/",
         "depth_root": "train_data/depth_re/",
         "anno": None},
    "NJU2000":
        {"image_root": "RGB/",
         "gt_root": "GT/",
         "depth_root": "depth/",
         "anno": "NJU2K_train.txt",
         },
    "NLPR":
        {"image_root": "RGB/",
         "gt_root": "GT/",
         "depth_root": "depth_re/",
         "anno": "NLPR_train.txt",
         },
    "RGBT":
        {"image_root": "RGB/",
         "gt_root": "GT/",
         "depth_root": "thermal/",
         "T_root": "thermal/",
         "anno": "RGBT-train.txt",
         },
    "RGB-T1000":
        {"image_root": "RGB/",
         "gt_root": "GT/",
         "depth_root": "T/",
         "T_root": "T/",
         "anno": "RGBT1000-train.txt",
         }

}

# data preparing, set your own data path here
data_root = opt.data_root
trainsets = opt.trainsets
image_paths = []
depth_paths = []
gt_paths = []
anno_paths = []
for trainset in trainsets:
    image_root = trainfolder[trainset]["image_root"]
    gt_root = trainfolder[trainset]["gt_root"]
    depth_root = trainfolder[trainset]["depth_root"]
    anno = trainfolder[trainset]["anno"]
    image_path = os.path.join(data_root, trainset, image_root)
    gt_path = os.path.join(data_root, trainset, gt_root)
    depth_path = os.path.join(data_root, trainset, depth_root)
    if anno:
        anno_path = os.path.join(data_root, trainset, anno)
    else:
        anno_path = None
    image_paths += [image_path]
    depth_paths += [depth_path]
    gt_paths += [gt_path]
    anno_paths += [anno_path]
train_loader = get_loader(image_paths, gt_paths, depth_paths, anno_paths, batchsize=opt.batchsize,
                          trainsize=opt.trainsize)
total_step = len(train_loader)

# build models
model = EFNet().cuda()
params = model.parameters()
optimizer = torch.optim.SGD(params, opt.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
CE = torch.nn.BCEWithLogitsLoss()
size_rates = [0.75, 1, 1.25]  # multi-scale training

# training
for epoch in range(0, opt.epoch):
    scheduler.step()
    model.train()
    loss_record1 = AvgMeter()
    loss_record2 = AvgMeter()
    loss_record3 = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts, depths = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            depths = Variable(depths).cuda()
            gt_edges = label_edge_prediction(gts)

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gt_edges = F.upsample(gt_edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # forward
            pred_s, pred_e, pred_ds = model(images, depths)

            loss1 = CE(pred_s, gts)
            loss2 = CE(pred_ds, gts)
            loss3 = CE(pred_e, gt_edges)
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            log_string('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                              loss_record1.show(), loss_record2.show(), loss_record3.show()))

    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, "epoch_{}_ckpt.pth".format(epoch)))
        log_string("model saved {}!".format(os.path.join(save_path, "epoch_{}_ckpt.pth".format(epoch))))

torch.save(model.state_dict(), os.path.join(save_path, 'final_ckpt.pth'))
