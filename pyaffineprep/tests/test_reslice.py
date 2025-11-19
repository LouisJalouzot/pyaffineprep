import nibabel
import numpy as np
from nilearn.image import iter_img
from nose.tools import assert_false

from ..affine_transformations import apply_realignment, get_initial_motion_params
from ..reslice import reslice_vols


def test_reslice_vols():
    # create basic L pattern
    n_scans = 3
    film = np.zeros((10, 10, 10, n_scans))
    film[-3:, 5:-1, :, :] = 1
    film[:, 2:5, :, :] = 1
    affine = np.array(
        [
            [-2.99256921e00, -1.12436414e-01, -2.23214120e-01, 1.01544670e02],
            [-5.69147766e-02, 2.87465930e00, -1.07026458e00, -8.77408752e01],
            [-2.03200281e-01, 8.50703299e-01, 3.58708930e00, -7.10269012e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    film = nibabel.Nifti1Image(film, affine)

    # rigidly move other volumes w.r.t. the first
    rp = np.array([get_initial_motion_params() for _ in range(n_scans)])
    for t in range(film.shape[-1]):
        rp[t, ...][:3] += t / n_scans
        rp[t, ...][3:6] += np.pi * t

    film = apply_realignment(film, rp)

    # affines are not the same
    assert_false(np.all(film[1].affine == film[0].affine))

    # reslice vols
    film = list(reslice_vols(film))

    # affines are now the same
    np.testing.assert_array_equal(film[1].affine, film[0].affine)
