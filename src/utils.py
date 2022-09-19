import numpy as np
import matplotlib.pyplot as plt
import os


def get_tensor_list(root_dir):
    files_in_dir = os.listdir('{}/tensors/'.format(root_dir))
    pointers = [string.split('_')[-1][:-4] for string in files_in_dir if 'labels' in string]
    pointers.sort()
    return np.arange(0, int(pointers[-1]) + 1)


def depth_encoding(image):
    """Applies a jet-colourscale to a depth image

        Parameters
        ----------
        image: :obj:`numpy.ndarray`
            Image to be encoded. Should have size of (X, X, 1)

        Returns
        -------
        :obj:`numpy.ndarray`
            Encoded depth image in colour scale. Size: (X, X, 3)
    """
    norm = plt.Normalize(vmin=0.4, vmax=1.4)
    colored_image = plt.cm.jet(norm(image))[:, :, :-1]
    return np.array((colored_image[:, :, 0], colored_image[:, :, 1], colored_image[:, :, 2]), dtype=np.float32)

def scale_data(data, scaling='max', sc_min=0.6, sc_max=0.8):
    """Scales a numpy array to [0, 255].

        Parameters
        ----------
        data: :obj:`numpy.ndarray`
            Data to be scaled.
        scaling: str
            Scaling method. Can be 'fixed' to scale between fixed values,
            or 'max' to scale between the minimum and maximum of the data.
            Defaults to 'max'.
        sc_min: float
            Lower bound for fixed scaling. Defaults to 0.6.
        sc_max: float
            Upper bound for fixed scaling. Defaults to 0.8

        Returns
        -------
        :obj:`numpy.ndarray`
            Scaled numpy array with the same shape as input array data.
    """
    data_fl = data.flatten()
    if scaling == 'fixed':
        scaled = np.interp(data_fl, (sc_min, sc_max), (0, 255), left=0, right=255)
    elif scaling == 'max':
        scaled = np.interp(data_fl, (min(data_fl), max(data_fl)), (0, 255), left=0, right=255)
    else:
        raise AttributeError
    integ = scaled.astype(np.uint8)
    integ.resize(data.shape)
    return integ

