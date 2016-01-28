"""
Tests for sam.pandasext.dataframe
"""

import sam.pandasext
import pandas as pd
import numpy as np

import datetime

from numpy.testing import assert_array_almost_equal


import pytest

@pytest.fixture(scope='module')
def df():
     t = [datetime.datetime(2009, 7, 21, 0, 0),
          datetime.datetime(2009, 7, 23, 0, 0),
          datetime.datetime(2009, 7, 25, 0, 0),
          datetime.datetime(2009, 7, 27, 0, 0)]

     a = [ 1,2,3,4 ]
     b = [ 10,20,30,40 ] 

     return pd.DataFrame(data=dict(a=a,b=b), index=t)
    

def test_interpolate_at_01(df):

     new_times = [datetime.datetime(2009, 7, 22, 0, 0),
                  datetime.datetime(2009, 7, 26, 0, 0)]
     
     df2 = df.interpolate_at(new_times,
                             max_timedelta=datetime.timedelta(2))

     assert df2.index.equals(pd.Index(new_times))
     
     assert np.allclose(df2.loc[:,'a'].values, [1.5, 3.5])
     assert np.allclose(df2.loc[:,'b'].values, [15, 35])

def test_interpolate_at_02_using_timestrings(df):

     new_times = ['2009-07-22 00:00', '2009-07-26 00:00']
     
     df2 = df.interpolate_at(new_times,
                             max_timedelta=datetime.timedelta(2))

     assert df2.index.equals(pd.Index(new_times))
     assert np.allclose(df2.loc[:,'a'].values, [1.5, 3.5])
     assert np.allclose(df2.loc[:,'b'].values, [15, 35])

def test_interpolate_at_01_giving_index_keep(df):

     new_times = [datetime.datetime(2009, 7, 22, 0, 0),
                  datetime.datetime(2009, 7, 26, 0, 0)]

     # just putting these times in a series, values not important here
     df2 = pd.DataFrame(index=new_times, data=[(99,98),(100,101)]) 
     
     df3 = df.interpolate_at(df2.index,
                             max_timedelta=datetime.timedelta(2),
                             keep=True)

     assert df3.index.equals(df.index | pd.Index(new_times))

     assert np.allclose(df3.loc[:,'a'].values, [1, 1.5, 2, 3, 3.5, 4])
     assert np.allclose(df3.loc[:,'b'].values, [10, 15, 20, 30, 35, 40])


     
