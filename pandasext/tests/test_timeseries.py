"""
Tests for sam.pandasext.timeseries
"""

import sam.pandasext
import pandas as pd
import numpy as np

import datetime

from numpy.testing import assert_array_almost_equal


import pytest

@pytest.fixture(scope='module')
def ts():
     t = [datetime.datetime(2009, 7, 21, 0, 0),
          datetime.datetime(2009, 7, 23, 0, 0),
          datetime.datetime(2009, 7, 25, 0, 0),
          datetime.datetime(2009, 7, 27, 0, 0)]

     v = [ 1,2,3,4 ]

     return pd.TimeSeries(data=v, index=t)
    

def test_interpolate_at_01(ts):

     new_times = [datetime.datetime(2009, 7, 22, 0, 0),
                  datetime.datetime(2009, 7, 26, 0, 0)]
     
     ts2 = ts.interpolate_at(new_times,
                      max_timedelta=datetime.timedelta(2))

     assert ts2.index.equals(pd.Index(new_times))
     assert np.allclose(ts2.values, [1.5, 3.5])

def test_interpolate_at_01_using_timestrings(ts):

     new_times = ['2009-07-22 00:00', '2009-07-26 00:00']
     
     ts2 = ts.interpolate_at(new_times,
                             max_timedelta=datetime.timedelta(2))

     assert ts2.index.equals(pd.Index(new_times))
     assert np.allclose(ts2.values, [1.5, 3.5])

     

def test_interpolate_at_01_keep(ts):

     new_times = [datetime.datetime(2009, 7, 22, 0, 0),
                  datetime.datetime(2009, 7, 26, 0, 0)]
     
     ts2 = ts.interpolate_at(new_times,
                             max_timedelta=datetime.timedelta(2),
                             keep=True)

     print(ts2)
     
     assert ts2.index.equals(ts.index | pd.Index(new_times))
     assert np.allclose(ts2.values, [1, 1.5, 2, 3, 3.5, 4])


def test_interpolate_at_01_giving_index_keep(ts):

     new_times = [datetime.datetime(2009, 7, 22, 0, 0),
                  datetime.datetime(2009, 7, 26, 0, 0)]

     # just putting these times in a series, values not important here
     ts2 = pd.TimeSeries(index=new_times, data=[99,98]) 
     
     ts3 = ts.interpolate_at(ts2.index,
                             max_timedelta=datetime.timedelta(2),
                             keep=True)

     assert ts3.index.equals(ts.index | pd.Index(new_times))
     assert np.allclose(ts3.values, [1, 1.5, 2, 3, 3.5, 4])

     
def test_interpolate_at_02(ts):
    
     #
     # Try to interpolate at points which already exist
     #

     
     its = ts.interpolate_at([datetime.datetime(2009, 7, 21, 0, 0),
                              datetime.datetime(2009, 7, 25, 0, 0)],
                              max_timedelta=datetime.timedelta(2))
     
     assert np.allclose(its.values,[ 1, 3])

def test_interpolate_at_03(ts):
     
     its = ts.interpolate_at([datetime.datetime(2009, 7, 21, 0, 0),
                              datetime.datetime(2009, 7, 27, 0, 0)],
                              max_timedelta=datetime.timedelta(2))
     assert np.allclose(its.values,[ 1, 4])


def test_interpolate_at_04(ts):
     
     # Try to interpolate at a point which already exists with time
     # segments which have only one point (max timedelta: 1 day):


     its = ts.interpolate_at([datetime.datetime(2009, 7, 21, 0, 0),
                              datetime.datetime(2009, 7, 27, 0, 0)],
                              max_timedelta=datetime.timedelta(1))
     
     assert np.allclose(its.values,[ 1, 4])


def test_interpolate_at_05(ts):
     
     # Test, whether NaNs are returned:
        
     its = ts.interpolate_at([datetime.datetime(2009, 7, 20, 0, 0),
                              datetime.datetime(2009, 7, 28, 0, 0)],
                              max_timedelta=datetime.timedelta(1))
     assert np.all(np.isnan(its.values))


def test_interpolate_at_06():
     
     # Interpolation values should be NaN, when time gap is too large:

     t = [datetime.datetime(2009, 7, 21, 0, 0),
          datetime.datetime(2009, 7, 23, 0, 0),
          datetime.datetime(2009, 7, 27, 0, 0),
          datetime.datetime(2009, 7, 27, 0, 1),]
     
     v = [ 1,2,4,5 ]
     ts = pd.TimeSeries(index=t,data=v)

     itimes = [ datetime.datetime(2009, 7, 22, 0, 0),
                datetime.datetime(2009, 7, 26, 0, 0)]
     
     its = ts.interpolate_at(itimes,
                             max_timedelta=datetime.timedelta(2))
     
     assert its.values[0] == 1.5
     assert np.isnan(its.values[1])


     its = ts.interpolate_at(itimes,
                             max_timedelta=datetime.timedelta(2),
                             keep=True)
     print(its)
          
     assert its.index.equals(ts.index | pd.Index(itimes))
     
     # Expected:
     #
     # 21   22   23   26   27a   27b
     #  1  1.5    2   NaN    4     5
     #
     assert_array_almost_equal(its.values,[ 1, 1.5, 2, np.nan, 4, 5])



############################### TEST OF INTEGRAL ################
     

## Initial imports

## >>> import datetime
## >>> import pandas as pd
## >>> def timeseries(times, values):
## ...     return pd.TimeSeries(values, times)
## >>> import numpy as np
## >>> import pdint
## >>> pd.TimeSeries.integral = pdint.integrate_method

## Returns integral over time series in given time interval.

##         begin -- datetime object, begin of time interval
##         end   -- datetime object, end of time interval

##         interpol_limits -- if True, search interpolation values for 'begin'
##         and 'end' and add this values to value list for numeric
##         interpolation; default: True

##         Warning: When there's no splitting into
##         time segments, e.g. when max_timedelta is None,
##         you maybe integrate over interpolation errors,
##         which can result in large errors

##         fallback_limit_value -- if given, this value is taken for
##         function values at interval limits,
##         if interpol_limits=True and
##         interpolation fails. If you have
##         passed a fill_func (see below), please
##         consider if this may be is sufficient.
##         The full_func is also used for limit values
##         if the interpolation of a limit fails.                                 
        

##         max_timedelta -- only perform integration over subintervals,
##         where maximum delta between times is not larger
##         than this delta (timedelta object);
##         Inside the large gaps, no integration can be done,
##         so an exception is raised unless a fill function
##         via the 'fill_func' argument is specified.
        
##         If given as None, the timeseries is integrated
##         as one segment. Be careful with this,
##         e.g. powercuts can result in wrong energies
##         when integrating over power.

##         fill_func -- function to fill time series with for too large gaps,
##         with signature

##         fill_func(t)

##         where t is a datetime object. It should return a float value.
##         Can be used to insert a model for
##         where no data is available. Simplest one is

##         fill_func = lambda t: 0

##         always returning zero (useful for powercuts or in
##         the night).
##         Default is None resulting in an exception when
##         a gap is too large.

##         The fill_func is also used to calculate a fallback
##         value for integration limits if the interpolation
##         of a limits value fails and if no explicit
##         fallback_limit_value is given.


##         min_coverage -- if given a number between 0 and 1,
##         perform a check, whether the segments
##         into which the integration interval is split
##         cover the full interval to a certain degree;
##         if the coverage is smaller, an exception is thrown;
##         if given None, no check is performed;
##         periods explicitly excluded are not taken into
##         account;
##         in other words: The coverage is the ratio between
##         the time we have data and the complete integration
##         interval. This ratio is checked for having at least
##         the given value.

##         exclude_periods -- if given a sequence of periods with 2-tuples

##         (<datetime.datetime>, <datetime.datetime>)

##         exclude these periods from integration.
##         This can be used e.g. to exclude nights.

##         The units of the result are (units of data)*seconds.

##         Examples:




def __test_integral_01():

    # Integration over constant:

    t = [datetime.datetime(2009, 7, 21, 0, 0),
         datetime.datetime(2009, 7, 23, 0, 0),
         datetime.datetime(2009, 7, 25, 0, 0),
         datetime.datetime(2009, 7, 27, 0, 0)]
    
    v = [ 3,3,3,3]
    ts = pd.TimeSeries(index=t,data=v)
    I = ts.integral(datetime.datetime(2009, 7, 23, 0, 0),
                    datetime.datetime(2009, 7, 25, 0, 0))
    assert I == 2*24*3600*3 # integration without extrapolation


##         Integration over constant with value change at limit:
##         >>> v = [ 0,0,1,1]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 25, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(3))
##         >>> I == 0.5*2*24*3600
##         True
        
##         Integration over linear function.

##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 23, 0, 0),
##         ...      datetime.datetime(2009, 7, 25, 0, 0),
##         ...      datetime.datetime(2009, 7, 27, 0, 0)]
##         >>> v = [ 1,2,3,4]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 23, 0, 0),
##         ...                 datetime.datetime(2009, 7, 25, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(3))
##         >>> I == 2*24*3600*(2+0.5*1) # integration without extrapolation
##         True
##         >>> I = ts.integral(datetime.datetime(2009, 7, 22, 0, 0),
##         ...                 datetime.datetime(2009, 7, 26, 0, 0),
##         ...                 interpol_limits=True,
##         ...                 max_timedelta=datetime.timedelta(3))
##         >>> I == 4*24*3600*(1.5+0.5*2)
##         True

##         Same test, but values are given in a matrix with one column:
        
##         >>> v = np.array([[ 1],[2],[3],[4]])
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 23, 0, 0),
##         ...                 datetime.datetime(2009, 7, 25, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(3))
##         >>> I == 2*24*3600*(2+0.5*1) # integration without extrapolation
##         True


##         Now, the limits are part of the values, test with and without interpolation.
        
##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 27, 0, 0),
##         ...                 interpol_limits=False,
##         ...                 max_timedelta=datetime.timedelta(3))
##         >>> I == (1+0.5*(4-1))*(27-21)*24*3600
##         True
##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 27, 0, 0),
##         ...                 interpol_limits=True,
##         ...                 max_timedelta=datetime.timedelta(3))
##         >>> I == (1+0.5*(4-1))*(27-21)*24*3600
##         True
##         >>> ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...             datetime.datetime(2009, 7, 21, 0, 0),
##         ...             interpol_limits=True,
##         ...             max_timedelta=datetime.timedelta(3))
##         0.0

##         When integration is done with only one value, it is
##         assumed constant about the whole interval:
##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0)]
##         >>> v = [4]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 20, 0, 0),
##         ...                 datetime.datetime(2009, 7, 22, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(3))
##         >>> I == 4*2*24*3600
##         True
        
##         This should not work, when max_timedelta ist too small:
        
##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0)]
##         >>> v = [4]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 20, 0, 0),
##         ...                 datetime.datetime(2009, 7, 22, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(0,10000))
##         Traceback (most recent call last):
##         ...
##         OverlargeTimeGapIntegrationError: Given lower limit is too small, there's no data! (times>='2009-07-21 00:00:00')

##         When integration is done over an interval with gaps>max_timedelta
##         and no fill_func is given and max_timedelta is not None,
##         there should be an exception:

##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 22, 0, 0),
##         ...      datetime.datetime(2009, 7, 26, 0, 0),
##         ...      datetime.datetime(2009, 7, 27, 0, 0)]
##         >>> v = [ 1,2,6,7]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 27, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(2))
##         Traceback (most recent call last):
##         ...
##         OverlargeTimeGapIntegrationError: Tried to integrate over gap [2009-07-22 00:00:00, 2009-07-26 00:00:00], but no fill function given!

##         With max_timedelta=None, there should be no exception:

##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 22, 0, 0),
##         ...      datetime.datetime(2009, 7, 26, 0, 0),
##         ...      datetime.datetime(2009, 7, 27, 0, 0)]
##         >>> v = [ 2,2,2,2]
##         >>> ts = timeseries(t,v)

##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 27, 0, 0),
##         ...                 max_timedelta=None)
##         >>> I == 6*2*24*3600
##         True

##         When integration is done over an interval with gaps>max_timedelta
##         the interval should be split and integrated separately.
##         We use a fill_func always resulting in 0 for all times:

##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 22, 0, 0),
##         ...      datetime.datetime(2009, 7, 26, 0, 0),
##         ...      datetime.datetime(2009, 7, 27, 0, 0)]
##         >>> v = [ 1,2,6,7]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 27, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(2),
##         ...                 fill_func=lambda t: 0)
##         >>> I == 24*3600*(1*1.5+1*6.5) 
##         True

##         The same should work, if there are more points around
##         (again with fill function returning zero):
        
##         >>> t = [datetime.datetime(2009, 7, 17, 0, 0),
##         ...      datetime.datetime(2009, 7, 18, 0, 0),
##         ...      datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 22, 0, 0),
##         ...      datetime.datetime(2009, 7, 26, 0, 0),
##         ...      datetime.datetime(2009, 7, 27, 0, 0),
##         ...      datetime.datetime(2009, 7, 30, 0, 0),
##         ...      datetime.datetime(2009, 7, 31, 0, 0)]
##         >>> v = [ 99, 99, 1, 2, 6, 7, 99, 99]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 27, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(2),
##         ...                 fill_func=lambda t: 0)
##         >>> I == 24*3600*(1*1.5+1*6.5) 
##         True

##         Example for giving a fill function not zero everywhere:

##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 25, 0, 0)]
##         >>> v = [ 1,5]
##         >>> ts = timeseries(t,v)
##         >>> a = datetime.datetime(2009, 7, 21, 0, 0)
##         >>> b = datetime.datetime(2009, 7, 25, 0, 0)
##         >>> I = ts.integral(a, b,
##         ...                 max_timedelta=datetime.timedelta(1),
##         ...                 fill_func=lambda t: timedelta2seconds(t-a)/(24*3600))
##         >>> np.allclose(I, 24*3600*0.5*4*4)
##         True

##         When integration is done over an interval 
##         and no fill_func is given and max_timedelta is too small
##         because there are not enough values
##         there should be an exception, also when a fallback value
##         for a limit is given:

##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 22, 0, 0),
##         ...      datetime.datetime(2009, 7, 23, 0, 0),
##         ...      datetime.datetime(2009, 7, 24, 0, 0)]
##         >>> v = [ 1,2,6,7]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 26, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(1),
##         ...                 fallback_limit_value=0)
##         Traceback (most recent call last):
##         ...
##         OverlargeTimeGapIntegrationError: Given upper limit is too large, there's no data! (times<='2009-07-24 00:00:00')

##         Similar exception should be thrown for limits too small:
##         >>> t = [datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 22, 0, 0),
##         ...      datetime.datetime(2009, 7, 23, 0, 0),
##         ...      datetime.datetime(2009, 7, 24, 0, 0)]
##         >>> v = [ 1,2,6,7]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 19, 0, 0),
##         ...                 datetime.datetime(2009, 7, 24, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(1),
##         ...                 fallback_limit_value=0)
##         Traceback (most recent call last):
##         ...
##         OverlargeTimeGapIntegrationError: Given lower limit is too small, there's no data! (times>='2009-07-21 00:00:00')

##         Raise exception, when values in integration interval are NaN:

##         >>> t = [datetime.datetime(2009, 7, 17, 0, 0),
##         ...      datetime.datetime(2009, 7, 18, 0, 0),
##         ...      datetime.datetime(2009, 7, 21, 0, 0),
##         ...      datetime.datetime(2009, 7, 22, 0, 0),
##         ...      datetime.datetime(2009, 7, 26, 0, 0),
##         ...      datetime.datetime(2009, 7, 27, 0, 0),
##         ...      datetime.datetime(2009, 7, 30, 0, 0),
##         ...      datetime.datetime(2009, 7, 31, 0, 0)]
##         >>> v = [ 99, 99, 1, 2, np.nan, 7, 99, 99]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 21, 0, 0),
##         ...                 datetime.datetime(2009, 7, 27, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(2),
##         ...                 fill_func=lambda t: 0)
##         Traceback (most recent call last):
##         ...
##         IntegrationError: Non-finite result when integrating time series from 2009-07-26 00:00:00 to 2009-07-27 00:00:00.


##         When a fill function is given and we integrate over an interval
##         in which is no data, use numerical integration over the fill function:
        
##         >>> I = ts.integral(datetime.datetime(2009, 7, 23, 0, 0),
##         ...                 datetime.datetime(2009, 7, 25, 0, 0),
##         ...                 max_timedelta=datetime.timedelta(2),
##         ...                 fill_func=lambda t: 9)
##         >>> np.allclose(I, 9*2*24*3600)
##         True

##         >>> logging.getLogger('cx.piecewise').setLevel(logging.WARN)

##         If there is no interpolation of limits but a fill function
##         given, the fill function can be used to calculate integrals
##         at the sides even if there is no data:

##         >>> t = [datetime.datetime(2009, 7, 16, 0, 0),
##         ...      datetime.datetime(2009, 7, 17, 0, 0),
##         ...      datetime.datetime(2009, 7, 18, 0, 0),
##         ...      datetime.datetime(2009, 7, 19, 0, 0),
##         ...      datetime.datetime(2009, 7, 20, 0, 0),]
##         >>> v = [ 0, 10, 10, 10, 0]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 16, 12, 0),
##         ...                 datetime.datetime(2009, 7, 19, 12, 0),
##         ...                 max_timedelta=datetime.timedelta(1.25),
##         ...                 fill_func=lambda t: 3)
##         >>> np.allclose(I, (1+0.5)*5*12*3600 + 2*10*24*3600 + (1+0.5)*5*12*3600)
##         True

##         This should also work, if the max_timedelta is large enough
##         to reach inner points, but too small to reach the integration limits:

##         >>> t = [datetime.datetime(2009, 7, 17, 10, 0),
##         ...      datetime.datetime(2009, 7, 17, 11, 0),
##         ...      datetime.datetime(2009, 7, 17, 12, 0),
##         ...      datetime.datetime(2009, 7, 17, 13, 0),
##         ...      datetime.datetime(2009, 7, 17, 14, 0),]
##         >>> v = [ 0, 10, 10, 10, 0]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 17, 8, 0),
##         ...                 datetime.datetime(2009, 7, 17, 16, 0),
##         ...                 max_timedelta=datetime.timedelta(0,90*60),
##         ...                 fill_func=lambda t: 3)
##         >>> np.allclose(I, (2*3 + 1*10./2 + 2*10 + 1*10./2 + 2*3)*3600)
##         True

##         Check whether the minimum coverage check can be switched off:

##         >>> t = [datetime.datetime(2009, 7, 15, 0, 0),
##         ...      datetime.datetime(2009, 7, 16, 0, 0),
##         ...      datetime.datetime(2009, 7, 17, 0, 0),
##         ...      datetime.datetime(2009, 7, 19, 0, 0),
##         ...      datetime.datetime(2009, 7, 20, 0, 0),]
##         >>> v = [ 0, 10, 10, 10, 0]
##         >>> ts = timeseries(t,v)
##         >>> I = ts.integral(datetime.datetime(2009, 7, 15, 0),
##         ...                 datetime.datetime(2009, 7, 20, 0),
##         ...                 max_timedelta=datetime.timedelta(1.25),
##         ...                 fill_func=lambda t: 10, min_coverage=None)
##         >>> np.allclose(I, (0.5*10+3*10+0.5*10)*24*3600)
##         True

##         A too small coverage is not ok and should raise an exception:
        
##         >>> I = ts.integral(datetime.datetime(2009, 7, 15, 0),
##         ...                 datetime.datetime(2009, 7, 20, 0),
##         ...                 max_timedelta=datetime.timedelta(1.25),
##         ...                 fill_func=lambda t: 10, min_coverage=0.8)
##         Traceback (most recent call last):
##         ...        
##         TooSmallCoverageIntegrationError: Requested coverage was 0.8, but we only have 0.6.


##         The time period from the integration limits to
##         the first/last real data points should not be taken into account
##         for the coverage if the gap is larger than max_timedelta:

##         >>> I = ts.integral(datetime.datetime(2009, 7, 13, 0),
##         ...                 datetime.datetime(2009, 7, 22, 0),
##         ...                 max_timedelta=datetime.timedelta(1.25),
##         ...                 fill_func=lambda t: 10, min_coverage=0.8)
##         Traceback (most recent call last):
##         ...        
##         TooSmallCoverageIntegrationError: Requested coverage was 0.8, but we only have 0.333333333333.


##         We can exclude periods from the integration. Here we integrate
##         over five days, excluding one day and filling the rest
##         with a constant function f(t)=10.
        
##         >>> ep = [[datetime.datetime(2009, 7, 18, 0),datetime.datetime(2009, 7, 19, 0)]]
##         >>> I = ts.integral(datetime.datetime(2009, 7, 15, 0),
##         ...                 datetime.datetime(2009, 7, 20, 0),
##         ...                 max_timedelta=datetime.timedelta(1.25),
##         ...                 fill_func=lambda t: 10, min_coverage=None,
##         ...                 exclude_periods=ep)
##         >>> np.allclose(I, (0.5*10+2*10+0.5*10)*24*3600)
##         True


##         Even than the coverage can be too small, here
##         we integrate over five days, exclude one day,
##         so can have a maximum of 4 days to integrate over.
##         We want 80%, but the remaining 3 days are only 75 %!

##         >>> ep = [[datetime.datetime(2009, 7, 18, 0),datetime.datetime(2009, 7, 19, 0)]]
##         >>> I = ts.integral(datetime.datetime(2009, 7, 15, 0),
##         ...                 datetime.datetime(2009, 7, 20, 0),
##         ...                 max_timedelta=datetime.timedelta(1.25),
##         ...                 fill_func=lambda t: 10, min_coverage=0.8,
##         ...                 exclude_periods=ep)
##         Traceback (most recent call last):
##         ...        
##         TooSmallCoverageIntegrationError: Requested coverage was 0.8, but we only have 0.75.


############################### TEST OF SEGMENTS ################

@pytest.fixture(scope='module')
def ts_seg():
    t = [ datetime.datetime(2009, 7, 21, 12, 0),
          datetime.datetime(2009, 7, 21, 12, 1),
          datetime.datetime(2009, 7, 21, 12, 2),
          datetime.datetime(2009, 7, 21, 12, 10),
          datetime.datetime(2009, 7, 21, 12, 12)]
    
    v = [ 1,2,3,4,5 ]

    return pd.TimeSeries(index=t, data=v)
    

def test_segments_01(ts_seg):

    segs = ts_seg.segments(None)

    assert len(segs)==1

    d = dict(list(segs))

    assert len(d)==1

    ts1, = [d[k] for k in sorted(d.keys())]

    assert ts1.index.equals(ts_seg.index)
    assert_array_almost_equal(ts1.values, ts_seg.values)
    

## Use 2 minutes as maximum timedelta:
def test_segments_02(ts_seg):
            
    segs = ts_seg.segments(datetime.timedelta(0,2*60))

    d = dict(list(segs))
        
    assert len(d)==2
    
    ts1, ts2 = [d[k] for k in sorted(d.keys())]

    assert ts1.index.equals(pd.DatetimeIndex([datetime.datetime(2009, 7, 21, 12, 0),
                                              datetime.datetime(2009, 7, 21, 12, 1),
                                              datetime.datetime(2009, 7, 21, 12, 2)]))
    assert ts2.index.equals(pd.DatetimeIndex([datetime.datetime(2009, 7, 21, 12, 10),
                                              datetime.datetime(2009, 7, 21, 12, 12)]))
                                              
    
    
        ## True
        ## >>> np.all(ts1.values == np.array([1,2,3]))
        ## True
        ## >>> np.all(ts2.values == np.array([4,5]))
        ## True

        ## The column names are conserved:
        
        ## >>> ts.colnames == ts1.colnames
        ## True
        ## >>> ts1.colnames == ts2.colnames
        ## True


        ## All deltas are already too large, so each point
        ## is a segment. To test this max_timedelta=30 seconds is
        ## chosen.

        ## >>> segs = ts.segments(datetime.timedelta(0,30))
        ## >>> len(segs)
        ## 5
        ## >>> ts1, ts2, ts3, ts4, ts5 = segs
        ## >>> ts1.times == [datetime.datetime(2009, 7, 21, 12, 0)]
        ## True
        ## >>> ts2.times == [datetime.datetime(2009, 7, 21, 12, 1)]
        ## True
        ## >>> ts3.times == [datetime.datetime(2009, 7, 21, 12, 2)]
        ## True
        ## >>> ts4.times == [datetime.datetime(2009, 7, 21, 12, 10)]
        ## True
        ## >>> ts5.times == [datetime.datetime(2009, 7, 21, 12, 12)]
        ## True
        ## >>> np.all(ts1.values == np.array([1]))
        ## True
        ## >>> np.all(ts2.values == np.array([2]))
        ## True
        ## >>> np.all(ts3.values == np.array([3]))
        ## True
        ## >>> np.all(ts4.values == np.array([4]))
        ## True
        ## >>> np.all(ts5.values == np.array([5]))
        ## True

        ## Test empty time series:

        ## >>> ts = timeseries([],[])
        ## >>> len(ts)
        ## 0
        ## >>> ts.segments(datetime.timedelta(0,1))
        ## []


        ## Another example:
        
        ## >>> t = [datetime.datetime(2009, 7, 17, 0, 0),
        ## ...      datetime.datetime(2009, 7, 18, 0, 0),
        ## ...      datetime.datetime(2009, 7, 19, 0, 0)]
        ## >>> v = [ 10, 10, 10]
        ## >>> ts = timeseries(t,v)
        ## >>> segs = ts.segments(datetime.timedelta(1.25))
        ## >>> len(segs)
        ## 1
        ## >>> segs[0].times == t
        ## True
        ## >>> np.all(segs[0].values == v)
        ## True

     
