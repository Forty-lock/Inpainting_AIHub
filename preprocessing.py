from torch.utils.data import Dataset
import torchvision.transforms as trans
import numpy as np
import cv2
import random
import glob
import torch

class CustomDataset(Dataset):
    def __init__(self, img_root, train=True):

        self.train = train

        self.image_list = glob.glob(img_root + '/img/*')
        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        self.msk_transform = trans.ToTensor()

    def __getitem__(self, idx):
        path_image = self.image_list[idx]
        path_gt = path_image.replace('img', 'gt')
        path_mask = path_image.replace('img', 'mask')

        name_image = path_image.split('/')[-1]

        Loadimage = cv2.imread(path_image)[:, :, ::-1]
        Loadgt = cv2.imread(path_gt)[:, :, ::-1]
        Loadmask = (cv2.imread(path_mask) > 125)

        Loadmask = Loadmask[:, :, 0:1].astype('float64')
        h, w, c = Loadimage.shape

        if self.train:
            if np.random.choice([True, False]):
                Loadimage = cv2.flip(Loadimage, 1)
                Loadgt = cv2.flip(Loadgt, 1)
                Loadmask = cv2.flip(Loadmask, 1)

            resize = np.random.uniform(1., 1.5)
            Loadimage = cv2.resize(Loadimage, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
            Loadgt = cv2.resize(Loadgt, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
            Loadmask = cv2.resize(Loadmask, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)

            h2, w2, c2 = Loadimage.shape

            rh = random.randint(0, h2 - h)
            rw = random.randint(0, w2 - w)

            Loadimage = Loadimage[rh:rh + h, rw:rw + w, :]
            Loadgt = Loadgt[rh:rh + h, rw:rw + w, :]
            Loadmask = Loadmask[rh:rh + h, rw:rw + w]

            img = self.img_transform(Loadimage.copy())
            gt = self.img_transform(Loadgt.copy())
            msk = self.msk_transform(Loadmask.copy()).float()

            return img, gt, msk
        else:

            img = self.img_transform(Loadimage.copy())
            gt = self.img_transform(Loadgt.copy())
            msk = self.msk_transform(Loadmask.copy()).float()

            return img, gt, msk, name_image

    def __len__(self):
        return len(self.image_list)

def collate(batch):
    paths = []
    imgs = []
    gts = []
    msks = []
    for sample in batch:
        imgs.append(sample[0])
        gts.append(sample[1])
        msks.append(sample[2])
        paths.append(sample[3])
    return torch.stack(imgs, 0), torch.stack(gts, 0), torch.stack(msks, 0), paths
