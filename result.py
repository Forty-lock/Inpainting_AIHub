import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocessing import CustomDataset
import utils
import net as mm
import csv
import time

save_path = './results'
Checkpoint = './model/PEPSI.pth'

def main():
    with open(save_path + '/eval.csv', 'w', newline='') as data_file:
        wr = csv.writer(data_file)
        wr.writerow(['ID', 'Time', 'Size', 'PSNR'])

    # --------- Data
    print('Data Load')
    dPath_data = './test/'
    custom_data = CustomDataset(dPath_data, train=False)
    data_loader = DataLoader(custom_data, batch_size=1)

    # --------- Model
    print('Model build')
    network = mm.Generator().cuda()

    print('Weight Restoring.....')
    network.load_state_dict(torch.load(Checkpoint)['G'])
    torch.cuda.empty_cache()
    print('Weight Restoring Finish!')

    # --------- Test
    network.eval()
    for isave, (img, gt, msk) in enumerate(tqdm(data_loader)):
        _, c_test, Height_test, Width_test = img.shape
        if Height_test % 8 != 0 or Width_test % 8 != 0:
            ht = Height_test // 8 * 8
            wt = Width_test // 8 * 8
            img = img[:, :, :ht, :wt]
            gt = gt[:, :, :ht, :wt]
            msk = msk[:, :, :ht, :wt]
            _, c_test, Height_test, Width_test = img.shape

        prediction = network(img.cuda() * msk.cuda(), msk.cuda()).cpu()

        utils.save_img(img, gt, msk, prediction, save_path, isave)

        gt = (gt + 1) * 255 / 2
        prediction = (prediction + 1) * 255 / 2
        mse = torch.mean((prediction - gt) ** 2)
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse + 1e-10)).item()

        stat = ['img_%04d.png' % isave]
        stat.append(time.ctime())
        hole = (1 - msk.numpy()).mean() * 100
        stat.append(hole)
        stat.append(psnr)

        with open(save_path + '/eval.csv', 'a', newline='') as data_file:
            wr = csv.writer(data_file)
            wr.writerow(stat)

if __name__ == '__main__':
    main()