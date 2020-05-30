import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import misc
from utils.data import test_dataset
from model.EFNet import EFNet

parser = argparse.ArgumentParser()
parser.add_argument('--testset', type=str, nargs='+', default='NJU2000', )
parser.add_argument('--data_root', type=str, default='data', )
parser.add_argument('--ckpts', type=str, default='')

opt = parser.parse_args()
testfolder = {
    "DUT-RGBD":
        {"image_root": 'test_data/test_images/',
         "gt_root": "test_data/test_masks/",
         "depth_root": "test_data/plt4/",
         "anno": None,
         },
    "NJU2000":
        {"image_root": "RGB/",
         "gt_root": "GT/",
         "depth_root": "depth/",
         "anno": "NJU2K_test.txt",
         },
    "NLPR":
        {"image_root": "RGB/",
         "gt_root": "GT/",
         "depth_root": "depth_re/",
         "anno": "NLPR_test.txt",
         },
    "STERE":
        {"image_root": "RGB/",
         "gt_root": "GT/",
         "depth_root": "depth_re/",
         "anno": None,
         },
    "SIP":
        {"image_root": "RGB/",
         "gt_root": "GT/",
         "depth_root": "depth_re/",
         "anno": None,
         }
}
data_root = opt.data_root
ckpt_path = opt.ckpts
valset = opt.testset

CE = torch.nn.BCEWithLogitsLoss()

model = EFNet(alpha=opt.alpha).cuda()
model.load_state_dict(torch.load(ckpt_path))
model.eval()
for dataset in valset:
    root_path = './'
    save_gt_path = root_path + 'gt/' + dataset + '/'
    save_pred_path = root_path + 'pred/temp/' + dataset + '/'
    if not os.path.exists(save_gt_path):
        os.makedirs(save_gt_path)

    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    image_root = testfolder[dataset]["image_root"]
    gt_root = testfolder[dataset]["gt_root"]
    depth_root = testfolder[dataset]["depth_root"]
    anno = testfolder[dataset]["anno"]
    image_path = os.path.join(data_root, dataset, image_root)
    gt_path = os.path.join(data_root, dataset, gt_root)
    depth_path = os.path.join(data_root, dataset, depth_root)
    if anno:
        anno_path = os.path.join(data_root, dataset, anno)
    else:
        anno_path = None
    test_loader = test_dataset(image_path, gt_path, depth_path, anno_path, testsize=352)

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, depth, name = test_loader.load_data()
            gt = np.array(gt).astype('float')
            gt = gt / (gt.max())

            image = Variable(image).cuda()
            depth = Variable(depth).cuda()
            with torch.no_grad():
                pred_s, pred_e, pred_ds = model(image, depth)

            res = F.upsample(pred_ds, size=gt.shape, mode='bilinear', align_corners=True)
            res_sig = res.sigmoid().data.cpu().numpy().squeeze()
            misc.imsave(save_pred_path + name + '.png', res_sig)
            print("{} saved!".format(save_pred_path + name + '.png'))
