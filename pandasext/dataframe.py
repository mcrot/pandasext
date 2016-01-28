#
# Extensions to pandas.DataFrame
#

import pandas as pd

def interpolate_at(df, times, max_timedelta=None, keep=False):
    """Interpolate a data frame at given times.

    Parameters
    ----------
    df : pandas.DataFrame
       Dataframe with DatetimeIndex which should be interpolated at
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
    idf_index = pd.DatetimeIndex(times)

    #
    # Apply interpolation on every column when seen as series
    #

    idf = pd.DataFrame(columns=df.columns)
    for c in df.columns:
        try:
            idf[c] = df[c].interpolate_at(idf_index, max_timedelta=max_timedelta)
        except Exception as exc:
            idf[c] = None

    if keep:
        idf = idf.append(df).sort_index()

    return idf

