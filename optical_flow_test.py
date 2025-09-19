from os import path
import os
import numpy as np
import cv2
import time

import pandas
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import torch
from vit_pytorch import SimpleViT
from vit_pytorch.crossformer import CrossFormer
from Model import HTNet
# from facenest import Fusionmodel
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def get_whole_u_v_os():
    df = pandas.read_csv('cas(me)3_part_A_edited.csv')
    m, n = df.shape
    base_data_src = '/workspace/part_A'
    total_emotion=0
    image_size_u_v = 224
    whole_u_v_os_images = {}

    for i in tqdm(range(0, m)): # すべての動画に対するループ
        # print(df['Subject'][i], df['Filename'][i])
        if df['imagename_apex'][i].split('/')[0] != 'spNO.138':
            img_path_apex = base_data_src + '/'+df['imagename_apex'][i].split('/')[0]+'/'+df['imagename_apex'][i]
            img_path_onset = base_data_src + '/' + df['imagename_onset'][i].split('/')[0]+'/'+df['imagename_onset'][i]
            train_face_image_apex = cv2.imread(img_path_apex)
            cv2.imwrite('result/train_face_image_apex.png', train_face_image_apex)
            train_face_image_apex = cv2.cvtColor(train_face_image_apex, cv2.COLOR_BGR2RGB)
            train_face_image_apex = Image.fromarray(train_face_image_apex)

            train_face_image_onset = cv2.imread(img_path_onset)
            cv2.imwrite('result/train_face_image_onset.png', train_face_image_onset)
            train_face_image_onset = cv2.cvtColor(train_face_image_onset, cv2.COLOR_BGR2RGB)
            train_face_image_onset = Image.fromarray(train_face_image_onset)
            # get face and bounding box
            mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
            face_apex = mtcnn(train_face_image_apex) #(3,28,28)
            face_apex = np.array(face_apex.permute(1, 2, 0).int().numpy()).astype('uint8') # (28,28,3)
            image_u_v_os_temp = np.zeros([image_size_u_v, image_size_u_v, 3], dtype=np.uint8)

            face_onset = mtcnn(train_face_image_onset)
            face_onset = np.array(face_onset.permute(1, 2, 0).int().numpy()).astype('uint8')
            pre_face_onset = cv2.cvtColor(face_onset, cv2.COLOR_BGR2GRAY)
            next_face_apex = cv2.cvtColor(face_apex, cv2.COLOR_BGR2GRAY)
            # pre_face_onset = cv2.cvtColor(face_onset, cv2.COLOR_RGB2GRAY)
            # next_face_apex = cv2.cvtColor(face_apex, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(pre_face_onset, next_face_apex, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            u = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
            v = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
            magnitude = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            image_u_v_os_temp[:, :, 0] = u
            image_u_v_os_temp[:, :, 1] = v
            image_u_v_os_temp[:, :, 2] = magnitude
            result_image = image_u_v_os_temp

            result_image_28 = cv2.resize(result_image, (28, 28), interpolation=cv2.INTER_AREA)

            cv2.imwrite('result/result.png', result_image)
            cv2.imwrite('result/result_image_28.png', result_image_28)

            # print(np.shape(image_u_v_os_temp))

            if face_onset is not None:
                total_emotion = total_emotion + 1
    # print(np.shape(whole_u_v_os_images))
        break
    #
    # print(total_emotion)
    return whole_u_v_os_images

if __name__=='__main__':
    get_whole_u_v_os()
