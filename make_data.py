import numpy as np
import glob
import cv2
import os
from tqdm import tqdm
import json

d_Path = 'D:/dataset/inpainting/'
width = 480
height = 270

def main():
    image_set_list = glob.glob(d_Path + '*/jpg/*')
    np.random.shuffle(image_set_list)

    for i, di in enumerate(tqdm(image_set_list)):

        if 'train' in image_set_list:
            save_path = './train'
        elif 'test' in image_set_list:
            save_path = './test'
        else:
            save_path = './val'

        image_list = glob.glob(di + '/*.jpg')
        image_list.sort()
        if len(image_list) == 0:
            print(di + '\thas no images!!!!')
            continue
        path_gt = image_list[-1]
        if not path_gt.split('_')[-1].startswith('G'):
            print(di + '\thas no GT!!!!')
            continue
        image_gt = cv2.imread(path_gt)
        image_list = image_list[:-1]

        for path_image in tqdm(image_list, mininterval=10, desc='%s\t' % di):
            path_mask = path_image.replace('\\jpg', '\\M').replace('.jpg', '_M.json')
            name_image = path_image.split('\\')[-1].split('.')[0]

            Loadimage = cv2.imread(path_image)
            h, w, c = Loadimage.shape

            # load json file
            with open(path_mask, encoding='UTF8') as data_file:
                dic = json.load(data_file)['Learning_Data_Info.']['Annotation']

            Loadmask = np.zeros((h, w), np.uint8)

            Loadgt = Loadimage.copy()

            for dic_ann in dic:

                polys = dic_ann['segmentation']
                if len(polys) < 6:
                    continue
                polys = np.array(polys).reshape(-1, 2)
                mask_temp = cv2.fillPoly(np.zeros((h, w), np.uint8), [polys], 1)

                if mask_temp.sum() <= 1:
                    continue

                (x, y, bw, bh) = cv2.boundingRect(polys)

                cX = x + bw//2
                cY = y + bh//2
                Loadmask = mask_temp + Loadmask * (1 - mask_temp)

                temp = cv2.copyMakeBorder(image_gt, 10, 10, 10, 10, cv2.BORDER_REFLECT)
                temp[10:-10, 10:-10, :] = Loadgt
                mask_temp = cv2.copyMakeBorder(mask_temp, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                dst = cv2.inpaint(temp, mask_temp, 3, cv2.INPAINT_NS)
                dst = dst[10:-10, 10:-10, :]
                mask_temp = mask_temp[10:-10, 10:-10]

                Loadgt = cv2.seamlessClone(image_gt, dst, mask_temp*255, (cX, cY), cv2.NORMAL_CLONE)

            Loadmask = 255-Loadmask * 255
            Loadgt = cv2.resize(Loadgt, dsize=(width, height), interpolation=cv2.INTER_AREA)
            Loadimage = cv2.resize(Loadimage, dsize=(width, height), interpolation=cv2.INTER_AREA)
            Loadmask = cv2.resize(Loadmask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = save_path + '/img'
            if not os.path.exists(save_name):
                os.makedirs(save_name)

            cv2.imwrite(save_name + '/%s.png' % name_image, Loadimage)

            save_name = save_path + '/gt'
            if not os.path.exists(save_name):
                os.makedirs(save_name)

            cv2.imwrite(save_name + '/%s.png' % name_image, Loadgt)

            save_name = save_path + '/mask'
            if not os.path.exists(save_name):
                os.makedirs(save_name)

            cv2.imwrite(save_name + '/%s.png' % name_image, Loadmask)

if __name__ == '__main__':
    main()