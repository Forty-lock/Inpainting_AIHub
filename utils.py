import torch
import torchvision
import numpy as np
import os
import math

def save_img(img, gt, msk, prediction, save_path, isave):

    save_path_img = save_path + '/img'
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)

    Bigpaper = torch.cat([img, img*msk, gt, prediction], 0)

    name = save_path_img + '/img_%04d.png' % isave
    torchvision.utils.save_image(Bigpaper, name, normalize=True, nrow=2)

def eval(test_loader, model, save_path):
    model.eval()

    temp1 = []
    temp2 = []
    temp3 = []
    with torch.no_grad():
        for i, (img, gt, msk, _) in enumerate(test_loader):

            Height_test, Width_test = img.shape[2:]
            if Height_test % 8 != 0 or Width_test % 8 != 0:
                ht = Height_test // 8 * 8
                wt = Width_test // 8 * 8
                img = img[:, :, :ht, :wt]
                gt = gt[:, :, :ht, :wt]
                msk = msk[:, :, :ht, :wt]

            prediction = model(img.cuda()*msk.cuda(), msk.cuda()).cpu()

            save_img(img, gt, msk, prediction, save_path, i)

            gt = (gt + 1) * 255 / 2
            prediction = (prediction + 1) * 255 / 2
            mse = torch.mean((prediction - gt) ** 2).item()
            psnr = 20 * math.log10(255.0 / math.sqrt(mse + 1e-10))

            rate = torch.mean(1-msk) * 100
            if rate < 10:
                temp1.append(psnr)
            elif rate < 20:
                temp2.append(psnr)
            elif rate < 30:
                temp3.append(psnr)

    model.train()
    return np.mean(temp1), np.mean(temp2), np.mean(temp3)

def save_model(G, D, optimizer_g, optimizer_d, model_path):

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    Checkpoint = model_path + '/PEPSI.pth'

    torch.save({
        'G': G.state_dict(),
        'D': D.state_dict(),
        'opt_G': optimizer_g.state_dict(),
        'opt_D': optimizer_d.state_dict(),
    }, Checkpoint)
    torch.cuda.empty_cache()
