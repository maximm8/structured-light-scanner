

import os
import os.path
import glob
import argparse
import cv2
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import structuredlight as sl
from pathlib import Path


import scipy

import open3d as o3d
import copy
import math

def fitplane(xyz):
    (a, b, c), _, _, _ = np.linalg.lstsq(np.c_[xyz[:,0], xyz[:,1], np.ones(xyz.shape[0])], xyz[:,2], rcond=1.e-10)
    normal = (a, b, -1)
    normal = (a, b, -1) / np.linalg.norm(normal)

    point = np.array([0.0, 0.0, c])
    d = point.dot(normal)

    return np.concatenate( [normal,  np.array([d])] )

def pixel2ray(x, K, kc):
    N = x.shape[1]

    x_undist = cv2.undistortPoints(x,K, kc)
    x_undist = np.concatenate((x_undist.reshape(-1,2), np.ones((N, 1))), axis=1).T
    xnorm=np.linalg.norm(x_undist, axis=0)
    x_undist =  x_undist/xnorm

    return x_undist

def line_plane_intersect(plane_abcd, line_p, line_v):
    w = plane_abcd
    p = line_p
    v = line_v
    wv = np.sum(w[0:3,:]*v, axis=0)
    t = (w[3,:]-np.sum(w[0:3,:]*p, axis=0))/wv;
    pp = p + np.tile(t, [3,1])*v;

    return pp

def calc_shadow_map(black, white, th):
    shadow_map = np.zeros_like(black)
    d = white.astype(np.float32)-black.astype(np.float32)
    ind = np.where(d>th)
    if len(shadow_map.shape) == 2:
        shadow_map[ind[0], ind[1]] = 1
    else:
        shadow_map[ind[0], ind[1], ind[2]] = 1

    shadow_map= shadow_map.astype(np.uint8)
    if len(shadow_map.shape)>2:
        shadow_map[:,:, 0] = shadow_map[:,:,0]|shadow_map[:,:,1]|shadow_map[:,:,2]
        shadow_map = shadow_map[:,:,0]

    return shadow_map

def decode_pattern(params, imgs):

    graycode = sl.Gray()
    patterns_size_vert  = math.ceil(math.log(params['resolution'][1])/math.log(2))
    # patterns_size_hor   = math.ceil(math.log(params['resolution'][0])/math.log(2))

    im_pos, im_neg = imgs[0:patterns_size_vert*2:2], imgs[1:patterns_size_vert*2:2]
    img_index_x = graycode.decode(im_pos, im_neg)

    im_pos, im_neg = imgs[patterns_size_vert*2::2], imgs[patterns_size_vert*2+1::2]
    img_index_y = graycode.decode(im_pos, im_neg)

    return img_index_x, img_index_y

def calc_image_rays(params):
    K   = params['intrinsic']
    kc  = params['distortion']
    res = params['resolution']
    xx, yy = np.meshgrid(np.linspace(0, res[1]-1, res[1]),  np.linspace(0, res[0]-1, res[0]))
    xy = np.concatenate((xx.reshape(1,-1), yy.reshape(1,-1)), axis=0)
    rays  = pixel2ray(xy, K, kc)

    return rays

def calc_plane_params(img_rays, P):
    data_row = []
    proj_plane_row = []
    for r in range(0, img_rays.shape[0], 1):
        # print('row:', r)
        data = img_rays[r,:,:].reshape(-1,3)
        data = np.concatenate([data, P.T], axis=0)
        abcd = fitplane(data)
        proj_plane_row.append(np.copy(abcd))
        data_row.append(np.copy(data))

    return proj_plane_row, data_row

def reconstruction(camera_params, projector_params, dataset):

    # preprocessing
    proj_res    = projector_params['resolution']

    rmat_inv    = projector_params['R']
    tvec_inv    = projector_params['T']
    rmat        = np.linalg.inv(projector_params['R'])
    tvec        = -np.matmul(rmat, projector_params['T'])

    cam_center = np.array([[0,0,0]]).T
    proj_center = np.array([[0,0,0]]).T
    proj_center = np.matmul(rmat, proj_center)+tvec

    img_white = dataset.pop()
    img_black = dataset.pop()

    # decode pattern
    img_index_x, img_index_y = decode_pattern(projector_params, dataset)

    #calc shadow map
    shadow_map = calc_shadow_map(img_black, img_white, 5)

    # estimate rays from camera
    camera_rays = calc_image_rays(camera_params)

    # estimate rays from projector
    projector_rays = calc_image_rays(projector_params)
    projector_rays = np.matmul(rmat, projector_rays) + tvec

    # reshape list of rays to the matrix structure
    projector_rays_img = np.zeros((proj_res[0], proj_res[1], 3), dtype=np.float64)
    projector_rays_img[:,:,0] = projector_rays[0,:].reshape(proj_res[:2])
    projector_rays_img[:,:,1] = projector_rays[1,:].reshape(proj_res[:2])
    projector_rays_img[:,:,2] = projector_rays[2,:].reshape(proj_res[:2])

    # calc projector row and column planes
    proj_plane_row, _ = calc_plane_params(projector_rays_img, proj_center)
    proj_plane_col, _ = calc_plane_params(np.transpose(projector_rays_img, axes=[1,0,2]), proj_center)

    proj_plane_col = np.array(proj_plane_col)
    proj_plane_row = np.array(proj_plane_row)

    # remove invalid estimations
    img_index_x *= shadow_map
    img_index_y *= shadow_map
    img_index_x[img_index_x>proj_res[1]-1] = -1
    img_index_y[img_index_y>proj_res[0]-1] = -1
    img_white_flat   = img_white.reshape(-1,1)
    img_index_x_flat = img_index_x.reshape(-1,1)
    img_index_y_flat = img_index_y.reshape(-1,1)

    ind = np.where((img_index_x_flat>-1) & (img_index_y_flat>-1))[0]

    # calc intersection of camera rays with projector row and column planes
    line_v =  camera_rays[:,ind]
    line_p = np.tile(cam_center, [1, line_v.shape[1]])

    plane_params = np.squeeze(proj_plane_row[img_index_y_flat[ind],:].T)
    depth_rows = line_plane_intersect(plane_params, line_p, line_v).T

    plane_params = np.squeeze(proj_plane_col[img_index_x_flat[ind],:].T)
    depth_cols = line_plane_intersect(plane_params, line_p, line_v).T

    depth_error = np.sqrt(np.sum((depth_cols-depth_rows)**2, axis=1))
    ind_depth = np.where(depth_error<5)[0]
    depth_ = depth_cols[ind_depth]

    gray_ = img_white_flat[ind]
    rgb_ = np.tile(gray_, (1,3))

    depth = np.zeros((img_index_x_flat.shape[0], 3))
    valid_ind = ind[ind_depth]
    depth[valid_ind] = depth_

    rgb = np.tile(img_white_flat, (1,3))

    #transform depth to the projector reference frame
    depth = (np.matmul(rmat_inv, depth.T)+tvec_inv).T

    return depth, rgb, valid_ind, img_index_x, img_index_y
