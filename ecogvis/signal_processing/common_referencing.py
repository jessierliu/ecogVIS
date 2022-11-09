from __future__ import division

import numpy as np

__all__ = ['subtract_CAR',
           'subtract_CAR_by_device',
           'subtract_common_median_reference']


def subtract_CAR(X, b_size=16):
    """
    Compute and subtract common average reference in 16 channel blocks.
    """

    channels, time_points = X.shape
    s = channels // b_size
    r = channels % b_size

    X_1 = X[:channels - r].copy()

    X_1 = X_1.reshape((s, b_size, time_points))
    X_1 -= np.nanmean(X_1, axis=1, keepdims=True)
    if r > 0:
        X_2 = X[channels - r:].copy()
        X_2 -= np.nanmean(X_2, axis=0, keepdims=True)
        X = np.vstack([X_1.reshape((s * b_size, time_points)), X_2])
        return X
    else:
        return X_1.reshape((s * b_size, time_points))


def subtract_CAR_by_device(X, elec_info=None):
    """
    Compute and subtract common average reference by electrode device as
    defined in the electrode table.
    """
    new_X = np.copy(X)

    for elec_device in elec_info.group_name.unique():

        if elec_device.startswith('null'):
            continue

        elec_idx = elec_info.loc[
            elec_info.group_name == elec_device].index.values

        cur_X = np.copy(X[:, elec_idx])
        cur_X -= np.nanmean(cur_X, axis=1, keepdims=True)
        new_X[:, elec_idx] = cur_X

    return new_X


def subtract_common_median_reference(X, channel_axis=-2):
    """
    Compute and subtract common median reference
    for the entire grid.

    Parameters
    ----------
    X : ndarray (..., n_channels, n_time)
        Data to common median reference.

    Returns
    -------
    Xp : ndarray (..., n_channels, n_time)
        Common median referenced data.
    """

    median = np.nanmedian(X, axis=channel_axis, keepdims=True)
    Xp = X - median

    return Xp
