import itertools
import numpy as np
import scipy as sp
import seaborn as sns

def nansem(a, axis=0, ddof=1, nan_policy='omit'):
    '''
    Returns standard error of the mean, while omitting nan values.
    '''
    return sp.stats.sem(a, axis, ddof, nan_policy)

def mean_ci(data, ci=95, axis=0, bootstrap=True, n_boot=10000):
    '''
    Returns mean and 95% confidence intervals, computed by bootstrapping
    '''
    a = 1.0 * np.array(data)
    m = np.nanmean(a, axis=axis)
    if bootstrap:
        boots = sns.algorithms.bootstrap(a, n_boot=1000, func=np.nanmean, axis=axis)
        ci_lo, ci_hi = sns.utils.ci(boots, ci, axis=axis)
    else:
        se = nansem(a, axis=axis)
        h = se * sp.stats.t.ppf((1 + ci/100) / 2., len(a)-1)
        ci_lo, ci_hi = m-h, m+h
    return m, ci_lo, ci_hi

def flatten_nested_list(list_of_lists):
    '''
    Flattens a list of lists to a list
    '''
    return list(itertools.chain(*list_of_lists))

def uneven_list2d_to_np(v, fillval=np.nan):
    '''
    Given a list of uneven lists, returns a 2-dimensional numpy array in which all lists are padded with fillval
    to the length of the longest list.
    '''
    lens = np.array([len(item) for item in v])
    if len(np.unique(lens)) == 1:
        return np.asarray(v)
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


def generate_standard_timestamp(timestamps, trim=False, min_time=None, max_time=None):
    '''
    Finds mean framerate and generates a single timestamp series starting from 0 evenly spaced to the max timestamp.

    timestamps: 2d numpy array with nan padding for uneven timestamp lengths
    
    If trim=True, finds the largest of the leftmost timestamps and the smallest of the rightmost timestamps.
    If min_time or max_time is defined, that value is used regardless of trim.
    '''
    if not isinstance(timestamps, np.ndarray):
        timestamps = uneven_list2d_to_np(timestamps)
    mean_diff = np.nanmean(np.diff(timestamps))
    if trim:
        min_time = np.nanmax(np.nanmin(timestamps,axis=1)) if min_time is None else min_time
        max_time = np.nanmin(np.nanmax(timestamps,axis=1)) if max_time is None else max_time
    else:
        min_time = np.nanmin(timestamps) if min_time is None else min_time
        max_time = np.nanmax(timestamps) if max_time is None else max_time

    return np.arange(min_time, max_time, mean_diff)

def interpolate_to_new_timestamp(y, t, nt):
    '''
    y: 1d data, length same as t
    t: original timestamp
    nt: new timestamp to interpolate to
    Returns ny, linearly interpolated data at nt
    '''
    not_nan = ~np.isnan(y)
    return np.interp(nt, t[not_nan], y[not_nan], left=np.nan, right=np.nan)


def align_traces_to_standardized_timestamp(ts, xs, ts_standard=None, trim=False, min_time=None, max_time=None):
    '''
    Given ts and xs, 2d numpy arrays representing timestamps and corresponding values, 
    returns xs_standardized, xs values interpolated to a standardized timestamp, ts_standard.
    If ts_standard is not provided, it is computed.
    '''
    if ts_standard is None:
        ts_standard = generate_standard_timestamp(ts, trim=trim, min_time=min_time, max_time=max_time)
    xs_standardized = np.array([interpolate_to_new_timestamp(xs[i], ts[i], ts_standard) for i in range(len(xs))])

    return ts_standard, xs_standardized