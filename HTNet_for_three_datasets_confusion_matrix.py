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
    result_image = cv2.resize(result_image, (28, 28), interpolation=cv2.INTER_AREA)
    # 例：保存したい場合
    # cv.imwrite('result_z/result_tvl1.png', result_image)

    return result_image

# class STSTNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super(STSTNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels=3, kernel_size=3, padding=2)
#         self.conv2 = nn.Conv2d(in_channels, out_channels=5, kernel_size=3, padding=2)
#         self.conv3 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, padding=2)
#         self.relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm2d(3)
#         self.bn2 = nn.BatchNorm2d(5)
#         self.bn3 = nn.BatchNorm2d(8)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc = nn.Linear(in_features=5 * 5 * 16, out_features=out_channels)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x1 = self.relu(x1)
#         x1 = self.bn1(x1)
#         x1 = self.maxpool(x1)
#         x1 = self.dropout(x1)
#         x2 = self.conv2(x)
#         x2 = self.relu(x2)
#         x2 = self.bn2(x2)
#         x2 = self.maxpool(x2)
#         x2 = self.dropout(x2)
#         x3 = self.conv3(x)
#         x3 = self.relu(x3)
#         x3 = self.bn3(x3)
#         x3 = self.maxpool(x3)
#         x3 = self.dropout(x3)
#         x = torch.cat((x1, x2, x3), 1)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}

    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''

def calculate_optical_flow_Farneback(img_path_apex, img_path_onset):
    image_size_u_v = 224
    train_face_image_apex = cv2.imread(img_path_apex)
    train_face_image_apex = cv2.cvtColor(train_face_image_apex, cv2.COLOR_BGR2RGB)
    train_face_image_apex = Image.fromarray(train_face_image_apex)

    train_face_image_onset = cv2.imread(img_path_onset)
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
    flow = cv2.calcOpticalFlowFarneback(pre_face_onset, next_face_apex, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    u = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
    v = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
    magnitude = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    image_u_v_os_temp[:, :, 0] = u
    image_u_v_os_temp[:, :, 1] = v
    image_u_v_os_temp[:, :, 2] = magnitude
    image_u_v_os_temp = cv2.resize(image_u_v_os_temp, (28, 28), interpolation=cv2.INTER_AREA)

    return image_u_v_os_temp, face_onset
    
def get_whole_u_v_os():
    df = pandas.read_csv('cas(me)3_part_A_edited.csv')
    m, n = df.shape
    base_data_src = '/workspace/part_A'
    total_emotion=0
    whole_u_v_os_images = {}

    for i in tqdm(range(0, m)): # すべての動画に対するループ
        # print(df['Subject'][i], df['Filename'][i])
        if df['imagename_apex'][i].split('/')[0] != 'spNO.138':
            img_path_apex = base_data_src + '/'+df['imagename_apex'][i].split('/')[0]+'/'+df['imagename_apex'][i]
            img_path_onset = base_data_src + '/' + df['imagename_onset'][i].split('/')[0]+'/'+df['imagename_onset'][i]

            # image_u_v_os_temp, face_onset = calculate_optical_flow_Farneback(img_path_apex, img_path_onset) # 元のアルゴリズム
            image_u_v_os_temp = calculate_optical_flow_tvl1(img_path_apex, img_path_onset)
            whole_u_v_os_images[i] = image_u_v_os_temp
            # print(np.shape(image_u_v_os_temp))

    return whole_u_v_os_images

def get_single_u_v_os(image_onset_url, image_apex_url):
    base_data_src = '/home/qixuan/Documents/part_A/data/part_A/'
    image_onset_url =  base_data_src + image_onset_url
    image_apex_url =  base_data_src + image_apex_url
    image_size_u_v = 28
    train_face_image_apex = cv2.imread(image_apex_url)
    train_face_image_apex = cv2.cvtColor(train_face_image_apex, cv2.COLOR_BGR2RGB)
    train_face_image_apex = Image.fromarray(train_face_image_apex)

    train_face_image_onset = cv2.imread(image_onset_url)
    train_face_image_onset = cv2.cvtColor(train_face_image_onset, cv2.COLOR_BGR2RGB)
    train_face_image_onset = Image.fromarray(train_face_image_onset)
    # get face and bounding box
    mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
    face_apex = mtcnn(train_face_image_apex)  # (3,28,28)
    face_apex = np.array(face_apex.permute(1, 2, 0).int().numpy()).astype('uint8')  # (28,28,3)
    image_u_v_os_temp = np.zeros([image_size_u_v, image_size_u_v, 3],dtype=np.uint8)

    face_onset = mtcnn(train_face_image_onset)
    face_onset = np.array(face_onset.permute(1, 2, 0).int().numpy()).astype('uint8')
    pre_face_onset = cv2.cvtColor(face_onset, cv2.COLOR_BGR2GRAY)
    next_face_apex = cv2.cvtColor(face_apex, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pre_face_onset, next_face_apex, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    u = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
    v = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
    magnitude = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    image_u_v_os_temp[:, :, 0] = u
    image_u_v_os_temp[:, :, 1] = v
    image_u_v_os_temp[:, :, 2] = magnitude
    # print(np.shape(image_u_v_os_temp), image_u_v_os_temp[:, :, 0])
    return image_u_v_os_temp

def create_norm_u_v_os_train_test():
    df = pandas.read_csv('cas(me)3_part_A_edited.csv')
    m, n = df.shape
    sub_names = df['Subject']
    base_destination_folder= 'created_opticalflow/'
    whole_u_v_os_Arr = get_whole_u_v_os()

    print('finish get')
    print('creating file...')

    for subname in tqdm(sub_names):
        # subname = spNO.1, spNO.10...

        for label in range(0,3):
            test_destination_folder = base_destination_folder + '/' + str(subname)  + '/u_test/'+str(label)
            train_destination_folder = base_destination_folder + '/' + str(subname) + '/u_train/'+str(label)
            try:
                os.makedirs(test_destination_folder, exist_ok=True)
            except OSError as error:
                print()
            try:
                os.makedirs(train_destination_folder, exist_ok=True)
            except OSError as error:
                print()

    print('finish create file')

    print('saving file...')
    for i in tqdm(range(0, m)):
        save_filename = str(df['Subject'][i]) + '_'+str(df['Filename'][i])+'_'+str(df['Apex'][i])+'.png' # 全体のoptical_flow画像を保存
        if i in whole_u_v_os_Arr: # spNO.138は存在しないため存在するものだけ保存
            u_v_os_image = whole_u_v_os_Arr[i]
            os.makedirs('whole_optical', exist_ok=True)
            status = cv2.imwrite(os.path.join('whole_optical', save_filename), u_v_os_image)
        for j in range(0, m):
            if df['Subject'][i] == df['Subject'][j]:
                label_for_three = df['label_for_3'][i]
                saved_filename = str(df['Subject'][i]) + '_' + str(df['Filename'][i]) + '_' + str(
                    df['Apex'][i]) + '.png'
                test_image_path =  base_destination_folder + str(df['Subject'][i]) +'/'+'u_test/'+str(label_for_three)+'/'+saved_filename

                if i in whole_u_v_os_Arr: # spNO.138は存在しないため存在するものだけ保存
                    u_v_os_image = whole_u_v_os_Arr[i]
                    status  = cv2.imwrite(test_image_path, u_v_os_image)
            else:
                label_for_three = df['label_for_3'][j]
                saved_filename = str(df['Subject'][j]) + '_'+str(df['Filename'][j])+'_'+str(df['Apex'][j])+'.png'
                test_image_path = base_destination_folder + str(df['Subject'][i]) + '/' + 'u_train/' + str(label_for_three)+'/'+saved_filename
                # print(i, j, test_image_path)

                if j in whole_u_v_os_Arr:
                    u_v_os_image = whole_u_v_os_Arr[j]
                    cv2.imwrite(test_image_path, u_v_os_image)

# 1. get the whole face block coordinates
def whole_face_block_coordinates(): # apexの画像を分割できるように書き換える
    base_data_src = '/workspace/HTNet/whole_optical_Farneback'
    original_data_src = '/workspace/part_A'
    total_emotion = 0
    image_size_u_v = 28
    # get the block center coordinates
    face_block_coordinates = {}

    print('Apex画像から顔領域を分割')
    for image_name in os.listdir(base_data_src):
        base_name = image_name.split('_')[2].split('.')[0] + '.jpg'
        img_path_apex = os.path.join(original_data_src, image_name.split('_')[0], image_name.split('_')[0], image_name.split('_')[1], 'color', base_name)
        train_face_image_apex = cv2.imread(img_path_apex) # (444, 533, 3)
        face_apex = cv2.resize(train_face_image_apex, (28,28), interpolation=cv2.INTER_AREA)
        # get face and bounding box
        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
        # if not detecting face
        if batch_landmarks is None:
            batch_landmarks = np.array([[[9.528073, 11.062551]
                                            , [21.396168, 10.919773]
                                            , [15.380184, 17.380562]
                                            , [10.255435, 22.121233]
                                            , [20.583706, 22.25584]]])
        row_n, col_n = np.shape(batch_landmarks[0])
        # print(batch_landmarks[0])
        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21
        batch_landmarks = batch_landmarks.astype(int)
        # print(batch_landmarks[0])
        # get the block center coordinates
        face_block_coordinates[image_name] = batch_landmarks[0]
    # print(len(face_block_coordinates))
    return face_block_coordinates

# 2. crop the 28*28-> 14*14 according to i5 image centers
def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()
    # print(len(face_block_coordinates_dict))
    # Get train dataset
    whole_optical_flow_path = '/workspace/HTNet/whole_optical_Farneback'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    five_parts_optical_flow_imgs = {}
    # print(whole_optical_flow_imgs[0]) #spNO.189_f_150.png
    for n_img in whole_optical_flow_imgs:
        five_parts_optical_flow_imgs[n_img]=[]
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        five_part_coordinates = face_block_coordinates_dict[n_img]
        l_eye = flow_image[five_part_coordinates[0][0]-7:five_part_coordinates[0][0]+7,
                five_part_coordinates[0][1]-7: five_part_coordinates[0][1]+7]
        l_lips = flow_image[five_part_coordinates[1][0] - 7:five_part_coordinates[1][0] + 7,
                five_part_coordinates[1][1] - 7: five_part_coordinates[1][1] + 7]
        nose = flow_image[five_part_coordinates[2][0] - 7:five_part_coordinates[2][0] + 7,
                five_part_coordinates[2][1] - 7: five_part_coordinates[2][1] + 7]
        r_eye = flow_image[five_part_coordinates[3][0] - 7:five_part_coordinates[3][0] + 7,
                five_part_coordinates[3][1] - 7: five_part_coordinates[3][1] + 7]
        r_lips = flow_image[five_part_coordinates[4][0] - 7:five_part_coordinates[4][0] + 7,
                five_part_coordinates[4][1] - 7: five_part_coordinates[4][1] + 7]
        five_parts_optical_flow_imgs[n_img].append(l_eye)
        five_parts_optical_flow_imgs[n_img].append(l_lips)
        five_parts_optical_flow_imgs[n_img].append(nose)
        five_parts_optical_flow_imgs[n_img].append(r_eye)
        five_parts_optical_flow_imgs[n_img].append(r_lips)
        # print(np.shape(l_eye))
    # print((five_parts_optical_flow_imgs['spNO.189_f_150.png'][0]))->(14,14,3)
    # print(len(five_parts_optical_flow_imgs))
    return five_parts_optical_flow_imgs

class Fusionmodel(nn.Module):
  def __init__(self):
    #  extend from original
    super(Fusionmodel,self).__init__()
    # self.num_whole_features = num_whole_features
    # self.num_l_eye_features = num_l_eye_features
    # self.num_l_lips_features = num_l_lips_features # add pose
    # self.num_nose_features = num_nose_features
    # self.num_r_eye_features = num_r_eye_features # 1000
    # self.number_r_lips_features =  number_r_lips_features #
    # input number = 512+512+512+1000，output number = 256
    self.fc1 = nn.Linear(15, 3) # 15->3
    self.bn1 = nn.BatchNorm1d(3)
    self.d1 = nn.Dropout(p=0.5)
    # Linear 256 to 26
    self.fc_2 = nn.Linear(6, 2) # 6->3
    # self.fc_cont = nn.Linear(256, 3)
    self.relu = nn.ReLU()

    # forward layers is to use these layers above
  def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
    # whole_feature = whole_feature.view(-1, self.num_whole_features)
    # l_eye_feature = l_eye_feature.view(-1, self.num_l_eye_features)
    # l_lips_feature = l_lips_feature.view(-1, self.num_l_lips_features)
    # nose_feature = nose_feature.view(-1, self.num_nose_features)
    # r_eye_feature = r_eye_feature.view(-1, self.num_r_eye_features)
    # r_lips_feature = r_lips_feature.view(-1, self.number_r_lips_features) # vit transformer
    #  cat features
    fuse_five_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
    # nn.linear - fc
    fuse_out = self.fc1(fuse_five_features)
    # fuse_out = self.bn1(fuse_out)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out) # drop out
    #
    fuse_whole_five_parts = torch.cat(
        (whole_feature,fuse_out), 0)
    # fuse_whole_five_parts = self.bn1(fuse_whole_five_parts)
    fuse_whole_five_parts = self.relu(fuse_whole_five_parts)
    fuse_whole_five_parts = self.d1(fuse_whole_five_parts)  # drop out
    out = self.fc_2(fuse_whole_five_parts)
    return out

def main(config):
    learning_rate = 0.00005
    batch_size = 128
    epochs = 250

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    loss_fn = nn.CrossEntropyLoss()

    if (config.train):
        if not path.exists('HTNet_Weights'):
            os.mkdir('HTNet_Weights')

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []
    best_total_pred = []

    t = time.time()

    main_path = '/workspace/HTNet/created_opticalflow_Farneback' # ここを変えてオプティカルフローのアルゴリズムを変える
    subName = os.listdir(main_path)
    print(subName)

    all_five_parts_optical_flow = crop_optical_flow_block()

    # tempdict = {}
    # subname = 'sub01'
    # accuracydict = {}
    # accuracydict['pred'] = [0, 1]
    # accuracydict['truth'] = [1, 0]
    # tempdict[subname] = accuracydict
    # print(tempdict['sub01']['pred'])
    for n_subName in subName: # 各被験者ごと
        print('Subject:', n_subName)

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        # five face parts
        l_eye_train = []
        l_lips_train = []
        nose_train = []
        r_eye_train = []
        r_lips_train = []
        four_parts_train = []
        l_eye_test = []
        l_lips_test = []
        nose_test = []
        r_eye_test = []
        r_lips_test = []
        four_parts_test = []

        # Get train dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)

            for n_img in img:
                y_train.append(int(n_expression))
                # X_train.append(cv2.imread(main_path + '/' + n_subName + '/u_train/' + n_expression + '/' + n_img))
                # get all five parts of optical flow
                # l_eye_train.append(all_five_parts_optical_flow[n_img][0])
                # l_lips_train.append(all_five_parts_optical_flow[n_img][1])
                # nose_train.append(all_five_parts_optical_flow[n_img][2])
                # r_eye_train.append(all_five_parts_optical_flow[n_img][3])
                # r_lips_train.append(all_five_parts_optical_flow[n_img][4])
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips  =  cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_train.append(lr_eye_lips)


        # Get test dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)

            for n_img in img:
                y_test.append(int(n_expression))
                # X_test.append(cv2.imread(main_path + '/' + n_subName + '/u_test/' + n_expression + '/' + n_img))
                # get all five parts of optical flow
                # l_eye_test.append(all_five_parts_optical_flow[n_img][0])
                # l_lips_test.append(all_five_parts_optical_flow[n_img][1])
                # nose_test.append(all_five_parts_optical_flow[n_img][2])
                # r_eye_test.append(all_five_parts_optical_flow[n_img][3])
                # r_lips_test.append(all_five_parts_optical_flow[n_img][4])
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_test.append(lr_eye_lips)

        weight_dir = os.path.join('ourmodel_threedatasets_weights-best')
        os.makedirs(weight_dir, exist_ok=True)
        weight_path = weight_dir + '/' + n_subName + '.pth'

        # Reset or load model weigts
        # model = STSTNet().to(device)
        model = HTNet(
            image_size=28,
            patch_size=7,
            dim=256,  # 96, 56-66.9, 192-71.33
            heads=3,  # 3 -  72, 6-71.35
            num_hierarchies=3,  # number of hierarchies
            block_repeats=(2, 2, 8),
            # the number of transformer blocks at each heirarchy, starting from the bottom(2,2,20) - 70.74
            num_classes=3
        )


        # fusionmodel = Fusionmodel().to(device)

        model = model.to(device)
        # five_parts_model =  five_parts_model.to(device)

        if(config.train):
            model.apply(reset_weights)
        else:
            model.load_state_dict(torch.load(weight_path))
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        # optimizer = torch.optim.Adam(list(model.parameters()) + list(five_parts_model.parameters())+ list(fusionmodel.parameters()), lr=learning_rate)

        # Initialize training dataloader
        # X_train = torch.Tensor(X_train).permute(0, 3, 1, 2)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        four_parts_train =  torch.Tensor(four_parts_train).permute(0, 3, 1, 2)
        # l_eye_train = torch.Tensor(l_eye_train).permute(0, 3, 1, 2)
        # l_lips_train = torch.Tensor(l_lips_train).permute(0, 3, 1, 2)
        # nose_train = torch.Tensor(nose_train).permute(0, 3, 1, 2)
        # r_eye_train = torch.Tensor(r_eye_train).permute(0, 3, 1, 2)
        # r_lips_train = torch.Tensor(r_lips_train).permute(0, 3, 1, 2)
        dataset_train = TensorDataset(four_parts_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=batch_size)

        # print(dataset_train)

        # Initialize testing dataloader
        # X_test = torch.Tensor(X_test).permute(0, 3, 1, 2)
        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(four_parts_test).permute(0, 3, 1, 2)
        # l_eye_test = torch.Tensor(l_eye_test).permute(0, 3, 1, 2)
        # l_lips_test = torch.Tensor(l_lips_test).permute(0, 3, 1, 2)
        # nose_test = torch.Tensor(nose_test).permute(0, 3, 1, 2)
        # r_eye_test = torch.Tensor(r_eye_test).permute(0, 3, 1, 2)
        # r_lips_test = torch.Tensor(r_lips_test).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(four_parts_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size)
        # store best results
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        record_dir = os.path.join('/workspace/HTNet/training_record', n_subName)
        os.makedirs(weight_dir, exist_ok=True)

        train_acc_lst = []
        train_loss_lst = []
        test_acc_lst = []
        test_loss_lst = []

        for epoch in tqdm(range(1, epochs + 1)):
            if (config.train):
                # Training
                model.train()
                # five_parts_model.train()
                # fusionmodel.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    # x1 = batch[2].to(device)
                    # x2 = batch[3].to(device)
                    # x3 = batch[4].to(device)
                    # x4 = batch[5].to(device)
                    # x5 = batch[6].to(device)
                    yhat = model(x)
                    # y_1 = five_parts_model(x1)
                    # y_2 = five_parts_model(x2)
                    # y_3 = five_parts_model(x3)
                    # y_4 = five_parts_model(x4)
                    # y_5 = five_parts_model(x5)
                    # yhat = fusionmodel(y_whole[0], y_1[0], y_2[0], y_3[0], y_4[0], y_5[0])
                    # print(yhat)
                    # print(y)
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.data.item() * x.size(0)
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_dl.dataset)
                train_acc_lst.append(train_acc)
                train_loss_lst.append(train_loss)

            # Testing
            model.eval()
            # five_parts_model.eval()
            # fusionmodel.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            for batch in test_dl:
                x = batch[0].to(device)
                y = batch[1].to(device)
                # x1 = batch[2].to(device)
                # x2 = batch[3].to(device)
                # x3 = batch[4].to(device)
                # x4 = batch[5].to(device)
                # x5 = batch[6].to(device)
                yhat = model(x)
                # y_1 = five_parts_model(x1)
                # y_2 = five_parts_model(x2)
                # y_3 = five_parts_model(x3)
                # y_4 = five_parts_model(x4)
                # y_5 = five_parts_model(x5)
                # yhat = fusionmodel(y_whole[0], y_1[0], y_2[0], y_3[0], y_4[0], y_5[0])
                loss = loss_fn(yhat, y)

                val_loss += loss.data.item() * x.size(0)
                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)
            test_acc_lst.append(val_acc)
            test_loss_lst.append(val_loss)
            #### best result
            temp_best_each_subject_pred = []
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
                best_each_subject_pred = temp_best_each_subject_pred

        print('Epoch %3d/%3d, train loss: %5.4f, train acc: %5.4f, val loss: %5.4f, val acc: %5.4f' % (epoch, epochs, train_loss, train_acc, val_loss, val_acc))
        # print(best_each_subject_pred)
        acc_record_path = os.path.join(record_dir, 'acc.png')
        epoch_num = range(1, len(train_acc_lst) + 1)
        plt.figure(figsize=(8,5))
        plt.plot(epoch_num, train_acc_lst, label='Train Acc', marker='o', markersize=2)
        plt.plot(epoch_num, test_acc_lst, label='Test Acc', marker='s', markersize=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training vs Testing Accuracy')
        plt.legend()
        plt.grid(True)
        os.makedirs(record_dir, exist_ok=True)
        plt.savefig(acc_record_path, dpi=300, bbox_inches='tight')
        plt.close()

        loss_record_path = os.path.join(record_dir, 'loss.png')
        plt.figure(figsize=(8,5))
        plt.plot(epoch_num, train_loss_lst, label='Train loss', marker='o', markersize=2)
        plt.plot(epoch_num, test_loss_lst, label='Test loss', marker='s', markersize=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Testing loss')
        plt.legend()
        plt.grid(True)
        os.makedirs(record_dir, exist_ok=True)
        plt.savefig(loss_record_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save Weights
        if (config.train):
            torch.save(model.state_dict(), weight_path)

        # For UF1 and UAR computation
        print('Predicted    :', torch.max(yhat, 1)[1].tolist())
        print('Best Predicted    :', best_each_subject_pred)
        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        total_pred.extend(torch.max(yhat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)


if __name__ == '__main__':
    # csvに基づくオプティカルフローの作成と保存 
    # create_norm_u_v_os_train_test()
    
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--train', type=strtobool, default=False)  # Train or use pre-trained weight for prediction
    config = parser.parse_args()
    main(config)
