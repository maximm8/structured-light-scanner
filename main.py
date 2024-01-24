# coding: UTF-8


import os.path
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d

import structuredlight_reconstruction as sl
import utils

def create_pointcloud(depth, rgb, valid_ind):

    pcd = o3d.geometry.PointCloud()
    d_ind = np.where((depth[valid_ind,2]>0)  & (depth[valid_ind,2]<500))[0]
    pcd.points = o3d.utility.Vector3dVector(depth[valid_ind[d_ind],:])
    pcd.colors = o3d.utility.Vector3dVector(rgb[valid_ind[d_ind],:].astype(np.float32)/255)
    # remove outliers
    pcd, ind = pcd.remove_radius_outlier(nb_points=40, radius=2)
    # o3d.visualization.draw_geometries([pcd])

    return pcd

def structured_light_reconstruction(params, dataset_dir, camera_ind):

    camera_params    = utils.load_params(params['camera'])
    projector_params = utils.load_params(params['projector'])
    dataset          = utils.load_dataset(dataset_dir, camera_ind)

    depth, rgb, valid_ind, img_index_x, img_index_y = sl.reconstruction(camera_params, projector_params, dataset)
    pcd = create_pointcloud(depth, rgb, valid_ind)

    return pcd, depth, rgb, img_index_x, img_index_y


if __name__ == '__main__':

    # setup parameters
    projector_ind   = 0 # projector index
    camera_inds     = [0,1] # camera index
    dataset_dir     = 'data/steve/' # dataset location
    dataset_dir     = 'data/yeti/'

    pcds = []
    for camera_ind in camera_inds:
        params = {}
        params['camera']    = f'params/camera{camera_ind:03d}.json'
        params['projector'] = f'params/projector{projector_ind:03d}_camera{camera_ind:03d}.json'
        pcd, _,_,_,_, = structured_light_reconstruction(params, dataset_dir, camera_ind)

        pcds.append(pcd)

    #save point clouds
    for camera_ind, pcd in zip(camera_inds, pcds + [pcds]):
        o3d.io.write_point_cloud(f'{dataset_dir}point_cloud_{camera_ind:03d}.ply', pcd)

    #show combined point cloud
    o3d.visualization.draw_geometries(pcds)