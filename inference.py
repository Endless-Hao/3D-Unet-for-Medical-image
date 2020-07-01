#Peichenhao 20200701
from load_data import loadDataGeneral

import numpy as np
import pandas as pd
import nibabel as nib
from keras.models import load_model

from scipy.misc import imresize
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure

# def IoU(y_true, y_pred):
#     assert y_true.dtype == bool and y_pred.dtype == bool
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.logical_and(y_true_f, y_pred_f).sum()
#     union = np.logical_or(y_true_f, y_pred_f).sum()
#     return (intersection + 1) * 1. / (union + 1)
#
# def Dice(y_true, y_pred):
#     assert y_true.dtype == bool and y_pred.dtype == bool
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.logical_and(y_true_f, y_pred_f).sum()
#     return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)
#
# def saggital(img):
#     """Extracts midle layer in saggital axis and rotates it appropriately."""
#     return img[:, img.shape[1] / 2, ::-1].T

#img_size = 128

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = './20190603/000test.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = './20190603/'

    df = pd.read_csv(csv_path)

    # Load test data
    append_coords = True
    X, y = loadDataGeneral(df, path, append_coords)

    n_test = X.shape[0]
    inpShape = X.shape[1:]

    # Load model
    model_name = './20190603/model/model.100.hdf5' # Model should be trained with the same `append_coords`
    model = load_model(model_name)

    # Predict on test data
    pred = model.predict(X, batch_size=1)[..., 1]

    # Compute scores and visualize
    # ious = np.zeros(n_test)
    # dices = np.zeros(n_test)
    for i in range(n_test):
        gt = y[i, :, :, :, 1] > 0.5 # ground truth binary mask
        pr = pred[i] > 0.5 # binary prediction
    #     # Save 3D images with binary masks if needed
        if True:
            tImg = nib.load(path + df.ix[i].path)
            nib.save(nib.Nifti1Image(255 * pr.astype('float'), affine=tImg.get_affine()), df.ix[i].path+'1000-pred.nii.gz')
