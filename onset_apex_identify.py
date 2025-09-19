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
import matplotlib.pyplot as plt


def _create_tvl1():
    # OpenCV のビルドにより関数名が違うことがある
    if hasattr(cv, "optflow"):
        if hasattr(cv.optflow, "DualTVL1OpticalFlow_create"):
            return cv.optflow.DualTVL1OpticalFlow_create()
        if hasattr(cv.optflow, "createOptFlow_DualTVL1"):
            return cv.optflow.createOptFlow_DualTVL1()
    # どれも無ければエラー
    raise RuntimeError("TV-L1 optical flow is unavailable. Install opencv-contrib-python.")

def calculate_optical_flow_tvl1(image_1, image_2):
    image_size_u_v = 224

    # 画像読み込み→RGB→PIL（MTCNN用）
    face_img1 = cv.imread(image_1)
    face_img1 = cv.cvtColor(face_img1, cv.COLOR_BGR2RGB)
    face_img1 = Image.fromarray(face_img1)

    face_img2 = cv.imread(image_2)
    face_img2 = cv.cvtColor(face_img2, cv.COLOR_BGR2RGB)
    face_img2 = Image.fromarray(face_img2)

    # 顔抽出（224x224, 後段でグレースケールに）
    mtcnn = MTCNN(margin=0, image_size=image_size_u_v,
                  select_largest=True, post_process=False, device='cuda:0')
    face_1 = mtcnn(face_img1)  # (3,H,W) tensor
    face_2 = mtcnn(face_img2)

    face_1 = np.array(face_1.permute(1, 2, 0).int().cpu().numpy()).astype('uint8')
    face_2 = np.array(face_2.permute(1, 2, 0).int().cpu().numpy()).astype('uint8')

    # TV-L1 は float32, 0..1 推奨
    prev_gray = cv.cvtColor(face_2, cv.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    next_gray = cv.cvtColor(face_1, cv.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # --- TV-L1 光フロー ---
    tvl1 = _create_tvl1()
    # 必要ならパラメータ調整（デフォルトで十分ならコメントのままでOK）
    # tvl1.setTau(0.25)
    # tvl1.setLambda(0.15)      # 正則化（大きいほど平滑）
    # tvl1.setTheta(0.3)
    # tvl1.setScalesNumber(5)
    # tvl1.setWarpingsNumber(5)
    # tvl1.setEpsilon(0.01)
    # tvl1.setOuterIterations(10)
    # tvl1.setInnerIterations(30)
    # tvl1.setGamma(0.0)
    # tvl1.setScaleStep(0.5)
    flow = tvl1.calc(prev_gray, next_gray, None)  # H×W×2, float32（画素あたりの変位/フレーム）

    # 大きさ・ベクトル分解
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.sqrt(u*u + v*v)

    # スカラー要約（明るさ変化に比較的頑健：中央値）
    motion_scalar = float(np.median(magnitude))

    # 可視化用 0..255 uint8 3ch（u, v, |flow|）
    # 符号付き u,v は 0..255 に正規化して保存しやすくする
    u_vis = cv.normalize(u, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    v_vis = cv.normalize(v, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    mag_vis = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    result_image = np.dstack([u_vis, v_vis, mag_vis])  # (H,W,3) uint8

    # 例：保存したい場合
    # cv.imwrite('result_z/result_tvl1.png', result_image)

    return motion_scalar


def calculate_optical_flow_Farneback(image_1, image_2):
    image_size_u_v = 224
    train_face_image_1 = cv2.imread(image_1)
    # cv2.imwrite('result_z/image1.png', train_face_image_1)
    train_face_image_1 = cv2.cvtColor(train_face_image_1, cv2.COLOR_BGR2RGB)
    train_face_image_1 = Image.fromarray(train_face_image_1)

    train_face_image_2 = cv2.imread(image_2)
    # cv2.imwrite('result_z/image2.png', train_face_image_2)
    train_face_image_2 = cv2.cvtColor(train_face_image_2, cv2.COLOR_BGR2RGB)
    train_face_image_2 = Image.fromarray(train_face_image_2)
    # get face and bounding box
    mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
    face_1 = mtcnn(train_face_image_1) #(3,28,28)
    face_1 = np.array(face_1.permute(1, 2, 0).int().numpy()).astype('uint8') # (28,28,3)
    image_u_v_os_temp = np.zeros([image_size_u_v, image_size_u_v, 3], dtype=np.uint8)

    face_2 = mtcnn(train_face_image_2)
    face_2 = np.array(face_2.permute(1, 2, 0).int().numpy()).astype('uint8')
    pre_face_2 = cv2.cvtColor(face_2, cv2.COLOR_BGR2GRAY)
    next_face_1 = cv2.cvtColor(face_1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pre_face_2, next_face_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    u = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
    v = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
    magnitude = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    image_u_v_os_temp[:, :, 0] = u
    image_u_v_os_temp[:, :, 1] = v
    image_u_v_os_temp[:, :, 2] = magnitude
    roi_mag = image_u_v_os_temp[:,:,2].ravel()
    motion_scalar = np.median(roi_mag)

    result_image = image_u_v_os_temp

    # cv2.imwrite(f'result_z/result_{motion_scalar}.png', result_image)

    return motion_scalar

def motion_curve(base_data_src):
    base_data_src = '/workspace/part_A/spNO.1/spNO.1/a/color'
    files = os.listdir(base_data_src)
    files_sorted = sorted(files, key=lambda x: int(x.split('.')[0]))
    motion_scalar_lst = []
    
    for i in tqdm(range(len(files_sorted) -1)):
        image_1 = base_data_src + '/' + files_sorted[i]
        image_2 = base_data_src + '/' + files_sorted[i+1]

        # motion_scalar = calculate_optical_flow_Farneback(image_1, image_2)
        motion_scalar = calculate_optical_flow_tvl1(image_1, image_2)
        motion_scalar_lst.append(motion_scalar)
    

    frames = list(range(len(motion_scalar_lst)))
    plt.figure(figsize=(10,5))
    plt.plot(frames, motion_scalar_lst, marker="o", label="Motion magnitude")
    plt.xlabel("Frame index")
    plt.ylabel("Motion (scalar)")
    plt.title("Optical Flow Motion Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig("motion_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__=='__main__':
    base_data_src = '/workspace/part_A/spNO.1/spNO.1/a/color'
    motion_curve(base_data_src)