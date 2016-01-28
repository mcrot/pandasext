#
# Extensions to pandas.TimeSeries
#

import pandas as pd
import scipy.integrate
import numpy as np

import logging
_log = logging.getLogger(__name__)


def _get_segment_indices( times_nanosec, max_nanoseconds):
    """Find contigous segments in a sequence of points in time.

    Parameters
    ---------- 

    times_nanosec -- array of times in nanoseconds
       
    max_nanoseconds -- float or int
       maximum distance of two times in nanoseconds.
       Where distance is larger, the
       time series is split into segments.

    Returns
    -------
    
    Returns a list of 2-tuples with

      (begin index, end index)

    which are indices of 'times', e.g.

      [ (0,10),(11,15) ]

    means there are 16 points in time which are split
    in two segments.

    If times is empty, return an empty list.

    Examples
    --------

    >>> onedaysecs = 24*60*60
    >>> t = [datetime.datetime(2009, 7, 21, 0, 0),
    ...      datetime.datetime(2009, 7, 22, 0, 0),
    ...      datetime.datetime(2009, 7, 23, 0, 0),
    ...      datetime.datetime(2009, 7, 25, 0, 0),
    ...      datetime.datetime(2009, 7, 31, 0, 0),
    ...      datetime.datetime(2009, 7, 31, 12, 0)]
    >>> t = datetime2epoch(t)*10**9
    >>> seg_idxs = _get_segment_indices(t,onedaysecs*10**9)
    >>> seg_idxs == [ (0,2),(3,3),(4,5) ]
    True

    >>> t = [datetime.datetime(2009, 7, 25, 0, 0),
    ...      datetime.datetime(2009, 7, 31, 0, 0),
    ...      datetime.datetime(2009, 8, 1, 12, 0)]
    >>> t = datetime2epoch(t)*10**9
    >>> seg_idxs = _get_segment_indices(t,onedaysecs*10**9)
    >>> seg_idxs == [ (0,0),(1,1),(2,2) ]
    True

    >>> t = [datetime.datetime(2009, 7, 25, 0, 0),
    ...      datetime.datetime(2009, 7, 30, 0, 0),
    ...      datetime.datetime(2009, 7, 30, 12, 0),
    ...      datetime.datetime(2009, 7, 31, 0, 0),
    ...      datetime.datetime(2009, 7, 31, 12, 0),
    ...      datetime.datetime(2009, 8, 1, 12, 0),
    ...      datetime.datetime(2009, 8, 6, 12, 0)]
    >>> t = datetime2epoch(t)*10**9
    >>> seg_idxs = _get_segment_indices(t,onedaysecs*10**9)
    >>> seg_idxs == [ (0,0),(1,5),(6,6) ]
    True

    Test with emtpy list of times:

    >>> t = []
    >>> seg_idxs = _get_segment_indices(t,onedaysecs*10**9)
    >>> seg_idxs == []
    True
    
    """
    _log.debug("Searching segment indices in %d times, max. nanoseconds: %s..",
               len(times_nanosec), max_nanoseconds)

    num_times = len(times_nanosec)

    if num_times==0:
        return []

    deltas = times_nanosec[1:]-times_nanosec[:-1]

    assert len(deltas)==len(times_nanosec)-1

    #
    # Search segments of subsequent points not having
    # a distance of more than max_timedelta
    #
    # in other words: split in time intervals,
    # where distance between intervals is at least
    # max_timedelta.
    #
    _log.debug("Searching too large deltas..")
    a = deltas>max_nanoseconds
    b = np.hstack((True,a))

    # now in array b every "True" marks a new segment

    # we just have to build the segments indices
    _log.debug("Building array of indices marking the begin of segments..")
    begin_indices = b.nonzero()[0]
    end_indices = np.hstack((begin_indices[1:]-1, num_times-1))

    #_log.info("t: %s, delta/max: %s, a: %s, b: %s, begin: %s, end: %s",
    #          times_sec, deltas/max_seconds, a, b, begin_indices, end_indices)
        
    return zip(begin_indices,end_indices)



def segments(ts, max_timedelta=None):
    """Divide TimeSeries into segments.

    Parameters
    ----------

    ts: pandas.TimeSeries
        Time series to be divided into segments.

    max_timedelta: string or datetime.timedelta or pandas.Timedelta
        maximum time gap allowed in one segment

    Returns
    -------

    GroupBy object
    """

    # TODO handle exclusion
    
    times_nanosec = ts.index.astype(np.int64)

    # TODO find another handling of max_timedelta

    if max_timedelta is None:
        seg_bound_idxs = [(0,len(ts)-1)]
    else:
        max_timedelta = pd.Timedelta(max_timedelta)
        max_nanoseconds = 10**9*max_timedelta.total_seconds()
        seg_bound_idxs = _get_segment_indices( times_nanosec,
                                               max_nanoseconds=max_nanoseconds)

    # print("Segment indices:", list(seg_bound_idxs))

    seg_df = pd.DataFrame(ts.values, index=ts.index)

    seg_df['segment period'] = None

    for seg_no, (start_idx, end_idx) in enumerate(seg_bound_idxs):
        seg_begin = ts.index[start_idx]
        seg_end = ts.index[end_idx]

        # print("#{}: {}/{} -----> {}/{}".format(seg_no, start_idx, seg_begin, end_idx, seg_end))        
        seg_df.loc[start_idx:end_idx+1, 'segment start'] = seg_begin
        seg_df.loc[start_idx:end_idx+1, 'segment end'] = seg_end

    return ts.groupby((seg_df['segment start'], seg_df['segment end']))

def integral(ts):
    """Return time series integral over seconds.

    Parameters
    ----------
    
    ts: pandas.TimeSeries

        The time series to integrate over.

    Returns
    -------

    Integral value with the folling units:
    If the incoming values have units [X] 
    then the result value has the units [Xs].
    Example: [W] -> [Ws] (watt-second)
    """
    return scipy.integrate.trapz(ts.values,
                                 x=ts.index.astype(np.int64)/10**9)


def interpolate_at(ts, times, max_timedelta=None, keep=False):
    """Interpolate a time series at given times.

    Parameters
    ----------
    ts : pandas.TimeSeries
       Time series which should be interpolated at
       other time stamps
    times: pandas.DatetimeIndex or sequence of datetimes/strings
       Times at which the resulting time series should have.
       Times outside the time span of the given series
       will result in NaN (we don't want to extrapolate).
       You can specify everything pandas.DatetimeIndex
       accepts as argument in order to create an index.
    max_timedelta: datetime.timedelta
       Maximum allowed time gap for choosing an
       interpolation value. If a gap is larger than this,
       NaN will be inserted instead.
       If given None, the size of the time gaps is not
       taken into account.
    keep: boolean
       If true, preserve times from original time series.
       
    """
    its_index = pd.DatetimeIndex(times)

    # use pandas interpolation not taking into account
    # too large time gaps or outer regions    
    its = ts.reindex(ts.index|its_index).interpolate(method='time').loc[its_index]

    #
    # remove extrapolated values
    #
    # outside the interpolation should be NaN
    #
    its.loc[its.index>ts.index.max()] = np.nan
    its.loc[its.index<ts.index.min()] = np.nan

    if max_timedelta is not None:
        #
        # set values to NaN if time gaps are too large
        #
        ts_nanosec = ts.index.astype(np.int64)
        its_nanosec = its_index.astype(np.int64)

        #
        # append inf at both sides
        #
        gaps_nanosec = np.concatenate(((np.inf,), np.diff(ts_nanosec), (np.inf,)))
        # gaps_nanosec = np.diff(ts_nanosec)

        # gaps have indices 0,..,n-1
        mtd_nanosec = max_timedelta.total_seconds()*10**9


        # too_large_cond = gaps_nanosec > mtd_nanosec

        #
        # find indices of gaps where the given times
        # belong to
        #
        n = len(ts)
        gaps_idxs = np.searchsorted(ts_nanosec, its_nanosec, side='left')
        gaps_idxs = gaps_idxs.clip(0, n-1)
        
        # gaps_idxs are almost indices of the intervals of original time series.
        # What is still to be done:
        # Correct interval indices by +1 where target time is equal
        # to leftmost interval point (because of side='left' in searchsorted)
        # but do not increase from n to n+1!
        
        known_times_cond = (its_nanosec==ts_nanosec[gaps_idxs])
        
        gaps_idxs = np.where(known_times_cond & (gaps_idxs+1<n),
                             gaps_idxs+1, gaps_idxs)

        
        # build a condition, which target times should be excluded from the result
        # 
        # include those:
        #
        #  - where time gap is finite and small enough
        #  - or we know already the value (no interpolation needed)
        #
        gaps_at_idxs = gaps_nanosec[gaps_idxs]
        gaps_ok_cond = (np.isfinite(gaps_at_idxs) & (gaps_at_idxs<=mtd_nanosec)) | known_times_cond


        ## print("ts times: {}".format(ts.index.tolist()))
        ## print("its times: {}".format(its.index.tolist()))
        ## print("known times: {}".format(known_times_cond))
        ## print("gaps ok: {}, gap indices: {}, gaps at indices (days): {}".format(
        ##     gaps_ok_cond, gaps_idxs, gaps_at_idxs/10**9/24/3600))


        its.loc[~gaps_ok_cond] = np.nan


    if keep:
        its = its.append(ts).sort_index()
    
    # print(its)
    
    return its

