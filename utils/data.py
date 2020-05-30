import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class SalObjDataset(data.Dataset):
    def __init__(self, image_roots, gt_roots, depth_roots, anno_paths, trainsize):
        self.trainsize = trainsize
        self.images = []
        self.gts = []
        self.depths = []
        for image_root, gt_root, depth_root, anno_path in zip(image_roots, gt_roots, depth_roots, anno_paths):
            if anno_path is None:
                self.images += [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
                self.gts += [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
                self.depths += [depth_root + f for f in os.listdir(depth_root) if f.endswith('bmp')]
            else:
                with open(anno_path, 'r') as f:
                    filelist = list(f)
                    filelist = [fn[:-1] for fn in filelist]
                self.images += [image_root + fn + '.jpg' for fn in filelist]
                self.gts += [gt_root + fn + '.png' for fn in filelist]
                self.depths += [depth_root + fn + '.bmp' for fn in filelist]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.point_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depth_transform(depth)

        return image, gt, depth

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def point_loader(self, path):
        point = np.load(path)
        point = point[:3, ]
        return point

    def __len__(self):
        return self.size


def get_loader(image_roots, gt_roots, depth_roots, anno_paths, batchsize, trainsize, shuffle=True, num_workers=0,
               pin_memory=True):
    dataset = SalObjDataset(image_roots, gt_roots, depth_roots, anno_paths, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, anno_path, testsize):
        self.testsize = testsize
        if anno_path is None:
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
            self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')]
        else:
            with open(anno_path, 'r') as f:
                filelist = list(f)
                filelist = [fn[:-1] for fn in filelist]
            self.images = [image_root + fn + '.jpg' for fn in filelist]
            self.gts = [gt_root + fn + '.jpg' for fn in filelist]
            self.depths = [depth_root + fn + '.jpg' for fn in filelist]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.gt_transform = transforms.ToTensor()
        self.point_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize))])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        # set_trace()
        image = self.rgb_loader(self.images[self.index])
        t_image = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        name = self.images[self.index].split('/')[-1][0:-4]
        t_depth = self.depth_transform(depth).unsqueeze(0)
        self.index += 1
        return t_image, gt, t_depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def point_loader(self, path):
        pillar = np.load(path)
        pillar = torch.from_numpy(pillar)
        pillar = pillar[:3, ]
        return pillar

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
