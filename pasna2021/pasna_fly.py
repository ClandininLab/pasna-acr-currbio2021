import os
import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from scipy.signal._peak_finding_utils import _select_by_peak_distance
from sklearn.linear_model import LinearRegression

class PasnaFly():
    '''
    Computes and stores PaSNA data for a single fly embryo.
    '''
    def __init__(self, csv_path, remove_artifacts=True, artifact_threshold=300, trim_data=True, trim_zscore=0.35):
        # Store path and infer embryo label/name from path
        self.csv_path = csv_path
        self.name = self.csv_path.split(os.path.sep)[-1][:-4]
        
        # Import raw data in DataFrame. 
        self.df_raw = self.import_data(remove_artifacts=remove_artifacts, artifact_threshold=artifact_threshold)
        self.df_raw['time'] /= 60 #seconds -> minutes
        
        # If trim_data, save df_raw in df_raw_untrimmed and overwrite df_raw with its trimmed version.
        if trim_data:
            self.df_raw_untrimmed = self.df_raw
            self.df_raw, self.trim_idx = self.trim_data(trim_zscore=trim_zscore)
            
        self.time = self.df_raw.time.to_numpy()
        
        # Compute ratiometric GCaMP signal (GCaMP / tdTomato)
        self.ratiom_gcamp = self.compute_ratiom_gcamp()
        
        # Compute deltaF/F
        self.dff = compute_dff(self.ratiom_gcamp.to_numpy())
        
        # Compute Savitzky-Golay-filtered signal and its 1st and 2nd derivatives
        self.savgol = spsig.savgol_filter(self.dff, 21, 4, deriv=0)
        self.savgol1 = spsig.savgol_filter(self.dff, 21, 4, deriv=1)
        self.savgol2 = spsig.savgol_filter(self.dff, 21, 4, deriv=2)
    
    def import_data(self, remove_artifacts=True, artifact_threshold=1500):
        '''
        Imports imaging data from indicated CSV file. 
        The imaging protocol sometimes produced artificial spikes in fluorescence. 
        If desired, these artifacts can be removed with a threshold, and the true value is infered by interpolation.
        '''
        df = pd.read_csv(self.csv_path, encoding='utf-16', header=[1])
        df.columns = ['time', 'gcamp', 'tomato']
        if remove_artifacts:
            arts_gc = np.logical_or(df.gcamp>artifact_threshold, df.gcamp==0) 
            df.gcamp.loc[arts_gc] = np.nan
            df.gcamp.loc[arts_gc]= np.interp(df.time[arts_gc], df.time[~arts_gc], df.gcamp.loc[~arts_gc])
            arts_td = np.logical_or(df.tomato>artifact_threshold, df.tomato==0)
            df.tomato.loc[arts_td] = np.nan
            df.tomato.loc[arts_td]= np.interp(df.time[arts_td], df.time[~arts_td], df.tomato.loc[~arts_td])

        return df
    
    def trim_data(self, trim_zscore=5):
        '''
        Computes the z score for each Savitzky-Golay-filtered sample, and removes the data 5 samples prior to the 
        first sample whose absolute value is greater than the threshold trim_zscore. 
        '''
        tomato_savgol = spsig.savgol_filter(self.df_raw.tomato, 251, 2, deriv=0)
        zscored_tomato = zscore(tomato_savgol)
        zscored_tomato -= compute_baseline(zscored_tomato, window_size=51)
        
        trim_points = np.where(np.abs(zscored_tomato) > trim_zscore)[0]
        trim_idx = trim_points[0]-5 if len(trim_points)>0 else None # Trim 5 timepoints before
        return (self.df_raw.loc[:trim_idx,:], trim_idx)
    
    def compute_ratiom_gcamp(self, remove_artifacts=False):
        '''
        Computes the ratiometric GCaMP signal by dividing the raw GCaMP signal by the tdTomato signal.
        Treats ratiom values of >15 as experimental artifacts and fills in those values by interpolation.
        '''
        ratiom = self.df_raw.gcamp/self.df_raw.tomato
        if remove_artifacts:
            arts = ratiom>15
            ratiom[arts] = np.nan
            ratiom[arts]= np.interp(self.time[arts], self.time[~arts], ratiom[~arts])
        return ratiom            

    def detect_peaks(self, mpd=71, order0_min=0.06, order1_min=0.006, extend_true_filters_by=30):
        '''
        Detects peaks using Savitzky-Golay-filtered signal and its derivatives, computed in __init__.
        Partly relies on spsig.find_peaks called on the signal, with parameters mpd (minimum peak distance)
         and order0_min (minimum peak height).
        order1_min sets the minimum first-derivative value, and the second derivative must be <0. These filters
         are stretched out to the right by extend_true_filters_by samples. 
        '''
        order0_idxes = spsig.find_peaks(self.savgol, height=order0_min, distance=mpd)[0]
        order0_filter = np.zeros(len(self.savgol), dtype=bool)
        order0_filter[order0_idxes] = True
        
        order1_filter = self.savgol1 > order1_min
        order1_filter = extend_true_right(order1_filter, extend_true_filters_by)

        order2_filter = self.savgol2 < 0
        order2_filter = extend_true_right(order2_filter, extend_true_filters_by)
        
        joint_filter = np.all([order0_filter, order1_filter, order2_filter], axis=0)
        peak_idxes = np.where(joint_filter)[0]
        peak_times = self.time[peak_idxes]

        self.peak_idxes = peak_idxes
        self.peak_times = peak_times
        self.peak_intervals = np.diff(peak_times)
        self.peak_amplitudes = self.savgol[peak_idxes]
        
    def compute_peak_bounds(self, rel_height=0.92):
        '''
        Computes the left and right bounds of each PaSNA peak using spsig.peak_widths.
        '''
        peak_widths_idxes, _, peak_left_idxes, peak_rights_idxes = spsig.peak_widths(self.savgol, self.peak_idxes, rel_height)
        peak_left_times = np.interp(peak_left_idxes, np.arange(len(self.time)), self.time)
        peak_right_times = np.interp(peak_rights_idxes, np.arange(len(self.time)), self.time)
        peak_widths_time = peak_right_times - peak_left_times
        peak_bounds_time = np.dstack((peak_left_times,peak_right_times)).squeeze()
        
        self.peak_widths_idxes = peak_widths_idxes
        self.peak_widths_time = peak_widths_time
        self.peak_bounds_exact_idxes = np.dstack((peak_left_idxes, peak_rights_idxes)).squeeze()
        self.peak_bounds_time = peak_bounds_time

    def get_peak_slices_from_bounds(self, left_pad=0, right_pad=0):
        '''
        Returns slices of all detected peaks, with indicated left and right padding (samples) around peak boundaries.
        Assumes that compute_peak_bounds has been called.
        '''
        peak_left_idxes_rnd = np.round(self.peak_bounds_exact_idxes[:,0]).astype(int)
        peak_right_idxes_rnd = np.round(self.peak_bounds_exact_idxes[:,1]).astype(int)
        peak_edges_idxes_rnd = np.dstack((peak_left_idxes_rnd-left_pad, peak_right_idxes_rnd+right_pad)).squeeze()
        peak_slices = [self.savgol[x[0]:x[1]] for x in peak_edges_idxes_rnd]
        time_slices = [self.time[x[0]:x[1]] for x in peak_edges_idxes_rnd]
        return list(zip(peak_slices, time_slices))
    
    def get_peak_slices_from_peaks(self, left_pad=3, right_pad=10):
        '''
        Returns slices of all detected peaks, with indicated left and right padding (samples) around the peak.
        Assumes that detect_peaks has been called.
        '''
        peak_edges_idxes_rnd = np.dstack((self.peak_idxes-left_pad, self.peak_idxes+right_pad)).squeeze().reshape(-1,2)
        peak_slices = [self.savgol[x[0]:x[1]] for x in peak_edges_idxes_rnd]
        time_slices = [self.time[x[0]:x[1]] for x in peak_edges_idxes_rnd]
        return list(zip(peak_slices, time_slices))
    
    def compute_peak_aucs_from_bounds(self, left_pad=0, right_pad=0):
        '''
        Returns AUCs (areas under the curve) for all detected peaks, with indicated left and right 
        padding (samples) around peak boundaries.
        Assumes that compute_peak_bounds has been called.
        '''
        peak_time_slices = self.get_peak_slices_from_bounds(left_pad=left_pad, right_pad=right_pad)
        peak_aucs = np.asarray([np.trapz(pslice*100,tslice) for pslice,tslice in peak_time_slices]) # %*min
        
        self.peak_aucs = peak_aucs

    def compute_peak_aucs_from_peaks(self, left_pad=3, right_pad=10):        
        '''
        Returns AUCs (areas under the curve) for all detected peaks, with indicated left and right 
        padding (samples) around peaks.
        Assumes that detect_peaks has been called.
        '''
        peak_time_slices = self.get_peak_slices_from_peaks(left_pad=left_pad, right_pad=right_pad)
        peak_aucs = np.asarray([np.trapz(pslice*100,tslice) for pslice,tslice in peak_time_slices]) # %*min
        
        self.peak_aucs = peak_aucs

    def get_pre_pasna_baseline(self, idx_bounds_from_peak0=(85,65)):
        '''
        Computes pre-PaSNA baseline from Savitzky-Golay-filtered ratiometric GCaMP signal. 
        Takes the mean of the window indicated by the index (sample) bounds. The window is
        from idx_bounds_from_peak0[0] to idx_bounds_from_peak0[1] left of the first detected peak.
        Assumes detect_peaks has been called.
        '''
        assert (self.peak_idxes is not None)
        ratiom_savgol = spsig.savgol_filter(self.ratiom_gcamp, 21, 4, deriv=0)
        first_peak_idx = self.peak_idxes[0]
        singleton_input = False
        if type(idx_bounds_from_peak0) is not list:
            idx_bounds_from_peak0 = [idx_bounds_from_peak0]
            singleton_input = True
        output = []
        for idx_bounds in idx_bounds_from_peak0:
            assert (idx_bounds[0] >= idx_bounds[1])
            if (first_peak_idx < idx_bounds[0]):
                output.append(np.nan)
                print("Not enough time before first peak: " + self.name)
                continue
            start_idx = first_peak_idx - idx_bounds[0]
            end_idx   = first_peak_idx - idx_bounds[1]
            output.append(np.nanmean(ratiom_savgol[start_idx:end_idx]))
        if singleton_input:
            output = output[0]
        return output
    
    def plot(self, raw=True, figsize=None):
        '''
        Plots raw signals and Savitzky-Golay-filtered signals, with indication of where data was trimmed (if applicable).
        If detect_peaks has been called, the detected peaks are marked.
        '''
        if raw:
            fig,ax = plt.subplots(3,1, figsize=figsize, sharex=True)
            if hasattr(self, 'df_raw_untrimmed'):
                ax[0].plot(self.df_raw_untrimmed.time, self.df_raw_untrimmed.gcamp, color='green', label='GCaMP')
                ax[1].plot(self.df_raw_untrimmed.time, self.df_raw_untrimmed.tomato, color='#b00000', label='tdTomato')
                if self.trim_idx is not None:
                    ax[0].axvline(self.df_raw_untrimmed.time[self.trim_idx])
                    ax[1].axvline(self.df_raw_untrimmed.time[self.trim_idx])
            else:
                ax[0].plot(self.df_raw.time, self.df_raw.gcamp, color='green', label='GCaMP')
                ax[1].plot(self.df_raw.time, self.df_raw.tomato, color='#b00000', label='tdTomato')
            ax[0].legend()
            ax[1].legend()
            ax_processed = ax[2]
            ax[0].set_title(self.name)
        else:
            fig,ax_processed = plt.subplots(1,1, figsize=figsize)
            ax_processed.set_title(self.name)
        
        ax_processed.plot(self.time, self.dff, linewidth=0.5, label='dF/F')
        ax_processed.plot(self.time, self.savgol, label='savgol')
        ax_processed.plot(self.time, self.savgol1, label='savgol1')
        ax_processed.plot(self.time, self.savgol2, label='savgol2')
        ax_processed.set_xlabel('Time [min]')
        ax_processed.legend()

        if hasattr(self, 'peak_idxes'):
            for x in self.peak_idxes:
                ax_processed.scatter(self.time[x],self.savgol[x])

        fig.show()
        return fig
    


def dff(signal, baseline):
    '''
    Helper function to compute deltaF/F given signal and baseline
    '''
    return (signal-baseline)/baseline
        
def get_start_end_idxes(mask):
    '''
    Given a boolean array mask, finds the first and last indices of each stretch of True values.
    '''
    mask_diffs = np.insert(np.diff(mask.astype(int)), 0, 0)
    starts = np.where(mask_diffs == +1)[0]
    ends   = np.where(mask_diffs == -1)[0]
    if starts[0] > ends[0]:
        ends = ends[1:]
    if len(starts) != len(ends):
        starts = starts[:-1]
    assert len(starts) == len(ends)
    return starts, ends

def compute_baseline(signal, window_size=140, n_bins=20):
    '''
    Compute baseline for each sliding window by dividing up the signal into n_bins amplitude bins and
    taking the mean of the bin with the most samples. This assumes that PaSNA peaks are sparse.
    For the first bit of signal in which the window falls off the left edge, we fit a linear regression
    to infer the baseline value. This assumes that PaSNA peaks do not occur during this period.
    
    window_size: number of frames
    n_bins: number of bins for binning into histogram
    '''
    n_windows = len(signal) - window_size + 1

    baseline = np.zeros_like(signal)
    first_center = window_size // 2
    for i in range(n_windows):
        window = signal[i:i+window_size]
        counts, bins = np.histogram(window, bins=n_bins)
        mode_bin_idx = np.argmax(counts)
        mode_bin_mask = np.logical_and(window > bins[mode_bin_idx], window <= bins[mode_bin_idx+1])
        window_baseline = np.mean(window[mode_bin_mask])
        baseline[first_center+i] = window_baseline
        
    # Linear regression up to first_center
    beginning_x = np.arange(0,first_center)
    model = LinearRegression()
    model.fit(beginning_x.reshape(-1,1), signal[0:first_center].reshape(-1,1))
    # fitted values
    trend = model.predict(beginning_x.reshape(-1,1))
    baseline[0:first_center] = trend.reshape(-1)
    
    # for the last first_center values, just take the last calculated baseline
    baseline[first_center+n_windows:] = baseline[first_center+n_windows-1]
    return baseline

def compute_dff(signal):
    '''
    Compute deltaF/F for signal by first computing the baseline using compute_baseline. 
    '''
    baseline = compute_baseline(signal)
    return dff(signal, baseline)

def extend_true_right(bool_array, n_right):
    '''
    Helper function that takes in a boolean array and extends each stretch of True values by n_right indices.
    
    Example:
    >> extend_true_right([False, True, True, False, False, True, False], 1)
    returns:             [False, True, True, True,  False, True, True]
    '''
    extended = np.zeros_like(bool_array, dtype=bool)
    for i in range(len(bool_array)):
        if bool_array[i] == True:
            rb = i+n_right
            rb = min(rb, len(bool_array))
            extended[i:rb] = True
    return extended