# visualize result of ycb tests. read each test image and result, project object pointcloud according to pose to image

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.io import loadmat
from functools import reduce
import math
from matplotlib import colors
import random

def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < 0.0001:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


result_path = '/media/alienicp/New Volume/DenseFusion/experiments/eval_result/ycb/Densefusion_wo_refine_result/'
image_path_file = '/media/alienicp/New Volume/DenseFusion/datasets/ycb/dataset_config/test_data_list.txt'
image_root_path = '/media/alienicp/New Volume/DenseFusion/datasets/ycb/YCB_Video_Dataset/'
model_path = '/media/alienicp/New Volume/DenseFusion/datasets/ycb/YCB_Video_Dataset/models/'
classes = ['002_master_chef_can',
            '003_cracker_box',
            '004_sugar_box',
            '005_tomato_soup_can',
            '006_mustard_bottle',
            '007_tuna_fish_can',
            '008_pudding_box',
            '009_gelatin_box',
            '010_potted_meat_can',
            '011_banana',
            '019_pitcher_base',
            '021_bleach_cleanser',
            '024_bowl',
            '025_mug',
            '035_power_drill',
            '036_wood_block',
            '037_scissors',
            '040_large_marker',
            '051_large_clamp',
            '052_extra_large_clamp',
            '061_foam_brick']
color_codes = [hex for name, hex in colors.cnames.items()]

object_pcds = []
for i in range(len(classes)):
    points = np.loadtxt(model_path + classes[i] + '/points.xyz')
    object_pcds.append(points) # divided by z, get x/z, y/z, 1

image_file = open(image_path_file)
for i in range(2949):
    print(i)
    img = Image.open(image_root_path + image_file.readline()[:-1] + '-color.png')
    pose_trans_quat = loadmat(result_path + '%04d.mat' % i)['poses']
    meta = loadmat(image_root_path + image_file.readline()[:-1] + '-meta.mat')
    obj_inds = meta['cls_indexes']
    intrinsic = meta['intrinsic_matrix']
    drawer = ImageDraw.Draw(img)
    for j in range(min(len(obj_inds), np.shape(pose_trans_quat)[0])):
        trans = pose_trans_quat[j, 4:]
        rot_mat = quaternion_matrix(pose_trans_quat[j, :4])[0:3, 0:3]
        est_points = np.add(np.dot(object_pcds[obj_inds[j][0]-1], np.transpose(rot_mat)), trans)
        est_points = np.divide(est_points, np.array(est_points[:,-1])[:, None])
        img_coor = np.matmul(est_points, np.transpose(intrinsic))[:, :-1].astype('int')
        tmp1, tmp2 = img_coor[:, 0], img_coor[:, 1]
        print(classes[obj_inds[j][0]-1])
        valid_ind = reduce(np.intersect1d, np.where(np.logical_and(tmp1 >= 1, tmp1 <= 480)), np.where(np.logical_and(tmp2 >= 1, tmp2 <= 640)))
        if valid_ind.size > 0 and j in [0, 2, 3]:
            coors = list(zip(img_coor[valid_ind, 0], img_coor[valid_ind, 1]))
            color = color_codes[obj_inds[j][0]]
            drawer.point(coors, fill=color)
            drawer.text((np.min(img_coor[valid_ind, 0])-15, np.min(img_coor[:, 1])-15), classes[obj_inds[j][0]-1], fill=color, font=ImageFont.truetype("Ubuntu-R.ttf", 15))
            drawer.rectangle((np.min(img_coor[valid_ind, 0]), np.min(img_coor[:, 1]), np.max(img_coor[valid_ind, 0]), np.max(img_coor[:, 1])),outline=color, width=2)
    img.save(result_path + '%04d' % i + '-output.png')