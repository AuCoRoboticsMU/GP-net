import time

import numpy as np
from scipy import ndimage
import torch
from PIL import Image

from src.utils import depth_encoding, scale_data
from src.experiments.vis_utils import Grasp
from src.experiments.transform import Transform, Rotation
from src.model import load_network
from skimage.feature import peak_local_max


class GPnet(object):
    def __init__(self, model_path, rviz=False, debug=False, detection_threshold=0.9, centre_representation=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, centre_rep=centre_representation)
        self.centre = centre_representation
        self.detection_threshold = detection_threshold
        self.rviz = rviz
        self.debug = debug

    def __call__(self, state):
        depth_im = state.depth_im
        camera_intr = state.camera_intrinsics
        camera_extr = state.camera_extrinsics

        tic = time.time()
        qual_pred, rot_pred, width_pred = predict(depth_im, self.net, self.device)
        if self.debug:
            image_depth = Image.fromarray(scale_data(depth_im))
            image_depth.save('data/debug/depth_im.png')
            image_quality = Image.fromarray(scale_data(qual_pred, scaling='fixed', sc_min=0.0, sc_max=1.0))
            image_quality.save('data/debug/quality_im.png')

        grasps, scores = select_grasps(qual_pred, rot_pred, width_pred, depth_im,
                                       camera_intr, camera_extr, self.detection_threshold,
                                       centre_rep=self.centre, debug=self.debug)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if len(grasps) > 0:
            p = np.random.permutation(len(grasps))
            grasps = grasps[p]
            scores = scores[p]

        # if self.rviz:
        #     vis.draw_quality(qual_pred, threshold=0.01)

        return grasps, scores, toc


def predict(depth_image, net, device):
    x = depth_encoding(depth_image)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual, rot, width = net(x)

    # move output back to the CPU
    qual = qual.cpu().squeeze().numpy()
    rot = rot.cpu().squeeze().numpy()
    width = width.cpu().squeeze().numpy()
    return qual, rot, width

def select_grasps(pred_qual, pred_quat, pred_width, depth_im, camera_intr,
                  camera_extr, threshold, n=5, centre_rep=False, debug=False):
    indices = peak_local_max(pred_qual, min_distance=4, threshold_abs=threshold, num_peaks=n)
    grasps = []
    qualities = []

    if debug:
        selected = np.zeros(pred_qual.shape)
        selected[indices[:, 0], indices[:, 1]] = 255
        image_quality = Image.fromarray(selected.astype(np.uint8))
        image_quality.save('data/debug/selected_im.png')

    for index in indices:
        quaternion = pred_quat[:, index[0], index[1]]
        quality = pred_qual[index[0], index[1]]

        contact = (index[1], index[0])
        if centre_rep:
            width = pred_width[0, index[0], index[1]]
            z = pred_width[-1, index[0], index[1]]
            grasp = reconstruct_grasp_from_variables(depth_im, contact, quaternion,
                                                     width, camera_intr, camera_extr, z)
        else:
            width = pred_width[index[0], index[1]]
            grasp = reconstruct_grasp_from_variables(depth_im, contact, quaternion, width, camera_intr, camera_extr)

        grasps.append(grasp)
        qualities.append(quality)
    return grasps, qualities


def reconstruct_grasp_from_variables(depth_im, contact, quaternion, width, camera_intr, T_camera_world, z=None):
    # Deproject from depth image into image coordinates
    # Note that homogeneous coordinates have the image coordinate order (x, y), while accessing the depth image
    # works with numpy coordinates (row, column)
    homog = np.array((contact[0], contact[1], 1)).reshape((3, 1))
    point = depth_im[contact[1], contact[0]] * np.linalg.inv(camera_intr.intrinsic.K).dot(homog)
    point = point.squeeze()

    # Transform the quaternion into a rotation matrix
    rot = quaternion_rotation_matrix(quaternion)
    if z is not None:
        # Add grasp centre difference to reconstructed, visible 3D point to get grasp centre
        centre_point = point
        centre_point[2] += z
    else:
        # Move from contact to grasp centre by traversing 0.5*grasp width in grasp axis direction
        centre_point = point + width / 2 * rot.T[0, :]

    # Construct transform Camera --> gripper
    T_camera_grasp = Transform.from_matrix(np.r_[np.c_[rot, centre_point], [[0, 0, 0, 1]]])

    # Express grasp in world coordinates
    T_gripper_world = T_camera_world.inverse() * T_camera_grasp

    return Grasp(T_gripper_world, width)


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix
