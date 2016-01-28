"""
Extension for Pandas.

When loaded, pandas is automatically extended
by certain functionalities.
"""

import pandas as pd
from .timeseries import segments, integral, interpolate_at
from .dataframe import interpolate_at as dataframe_interpolate_at

### strange why we cannot import like that?
# from . import timeseries

#
# Extend exisiting pandas classed by some methods
#
# WARNING: monkey patching, can fail if pandas changes
#
pd.TimeSeries.segments = segments
pd.TimeSeries.integral = integral
pd.TimeSeries.interpolate_at = interpolate_at
pd.core.groupby.SeriesGroupBy.integral = lambda gb: gb.apply(integral)

pd.DataFrame.interpolate_at = dataframe_interpolate_at
