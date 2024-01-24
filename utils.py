import cv2
import json
import numpy as np
import glob

def load_params(json_file):
    camera_params = {}

    with open(json_file, 'r') as f:
        param_data = json.load(f)

        camera_params['intrinsic']      = np.array(param_data['intrinsic']).reshape(3,3)
        camera_params['distortion']     = np.array(param_data['distortion']).reshape(5,1)
        camera_params['resolution']     = np.array(param_data['resolution'])
        camera_params['R']              = np.array(param_data['R']).reshape(3,3)
        camera_params['T']              = np.array(param_data['T']).reshape(3,1)

    return camera_params

def load_dataset(dataset_dir, camera_ind):

    fnames = sorted(glob.glob(f'{dataset_dir}/cam{camera_ind:03d}_*.png'))
    imgs = []
    for fname in fnames:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        imgs.append(img)

    return imgs