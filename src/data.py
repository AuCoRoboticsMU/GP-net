import torch.utils.data
import numpy as np
from utils import depth_encoding, get_tensor_list

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, noise=False, centre_representation=False, metric_threshold=0.5, image_size=None):
        self.root = root
        self.metric_threshold = metric_threshold
        self.centre_representation = centre_representation
        self.noise = noise
        self.image_size = image_size

        self.tensor_list = get_tensor_list(self.root)

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, i):
        x, y, im = self._get_ground_truth_data(i)

        return x, y, im

    def _select_grasps(self, grasp_data):
        """
        Selects the entities of the grasp_data tensor
        :param grasp_data: numpy array containing details about the grasp with g in
        {u, v, z, contact_u, contact_v, width, quaternion, hinge_rot, quality, collision_free}
        :return: pose input used to train the network.
        """
        try:
            centre_u = grasp_data[:, 0]
            centre_v = grasp_data[:, 1]
            centre_depth = grasp_data[:, 2]
            u = grasp_data[:, 3]
            v = grasp_data[:, 4]
            width = grasp_data[:, 5]
            rot = grasp_data[:, 6:10]
            hinge = grasp_data[:, 10]
            quality = grasp_data[:, 11]
            collision_free = grasp_data[:, 12]
        except IndexError:
            # Only one grasp in array - index accordingly
            centre_u = grasp_data[0]
            centre_v = grasp_data[1]
            centre_depth = grasp_data[2]
            u = grasp_data[3]
            v = grasp_data[4]
            width = grasp_data[5]
            rot = grasp_data[6:10]
            hinge = grasp_data[10]
            quality = grasp_data[11]
            collision_free = grasp_data[12]
        quality = quality * collision_free
        hinge = np.nan_to_num(hinge, nan=0.0)
        labels = np.where(quality >= self.metric_threshold, 1, 0)
        if self.centre_representation:
            return centre_u.astype(int), centre_v.astype(int), centre_depth, width, rot, hinge, labels

        return u.astype(int), v.astype(int), None, width, rot, hinge, labels

    @staticmethod
    def get_indexes_for_element(values, element):
        element_indexes = [index for index, value in enumerate(values) if value == element]
        return {element: element_indexes}

    def _get_ground_truth_data(self, image_label):
        """
        Chooses a grasp in the loaded tensor, loads the data for network training and updates the indices for
        future selections.
        :return: image_data: numpy array in shape (1, 300, 300) for the depth image
                 pose_data: numpy array in shape (1,) for the pose input (shape varies for alternative self.pose_input)
                 label_data: [0/1] for ground truth negative/positive grasps
        """

        image = np.load('{}/tensors/depth_im_{:07d}.npz'.format(self.root, image_label))['arr_0'].squeeze()
        if self.centre_representation:
            grasps = np.load('{}/tensors/centre_grasps_{:07d}.npz'.format(self.root, image_label))['arr_0']
        else:
            grasps = np.load('{}/tensors/contact_grasps_{:07d}.npz'.format(self.root, image_label))['arr_0']
        segmask = np.load('{}/tensors/binary_{:07d}.npz'.format(self.root, image_label))['arr_0'].squeeze()

        if self.image_size is not None:
            sz = image.shape
            v0 = int(sz[0] / 2 - self.image_size[0] / 2)
            v1 = int(sz[0] / 2 + self.image_size[0] / 2)
            u0 = int(sz[1] / 2 - self.image_size[1] / 2)
            u1 = int(sz[1] / 2 + self.image_size[1] / 2)
            image = image[v0:v1, u0:u1]
            segmask = segmask[v0:v1, u0:u1]

        colored_im = depth_encoding(image)
        y_true = np.zeros(shape=(image.shape[0], image.shape[1], 8))

        # We can get the non-object pixel from the segmask - all those are negative grasps
        non_object = np.where(segmask == 0)

        # Indicate we have ground truth at the table indices, but leaving the quality as zeros
        # Numpy coordinates, since they come from np.where()!!
        y_true[non_object[0], non_object[1], 0] = 1

        try:
            u_ind, v_ind, centre, width, quaternion, hinge_rotation_bounds, labels = self._select_grasps(grasps)
        except IndexError:
            # Didn't have any grasps for this image - we just return the table grasps
            return colored_im, y_true

        if len(u_ind) == 0:
            # Didn't have any contact grasps for this image - we just return the table grasps
            return colored_im, y_true

        if self.image_size is not None:
            mask = (u_ind > u0) & (u_ind < u1) & (v_ind > v0) & (v_ind < v1)
            u_ind = u_ind[mask] - u0
            v_ind = v_ind[mask] - v0
            labels = labels[mask]
            quaternion = quaternion[mask, :]
            width = width[mask]
            if centre is not None:
                centre = centre[mask]
        if centre is not None:
            centre -= image[v_ind, u_ind]

        # Image coordinates (u, v) to numpy coordinates (row, column)
        y_true[v_ind, u_ind, 0] = 1
        y_true[v_ind, u_ind, 1] = labels
        y_true[v_ind, u_ind, 2] = quaternion[:, 0]
        y_true[v_ind, u_ind, 3] = quaternion[:, 1]
        y_true[v_ind, u_ind, 4] = quaternion[:, 2]
        y_true[v_ind, u_ind, 5] = quaternion[:, 3]
        y_true[v_ind, u_ind, 6] = width
        if centre is not None:
            y_true[v_ind, u_ind, 7] = centre

        return colored_im, y_true, image
