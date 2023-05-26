#This code fills in values by linear interpolation in time, 
#And reproduces the code https://github.com/r-forge/greenbrown/blob/master/pkg/greenbrown/R/TSGFlinear.R 
# Because I am too lazy to import and convert np to ts 
import numpy as np
from scipy.interpolate import Akima1DInterpolator
def interpolate(ts):
    #Takes in a uni-variate Time Series
    #The frequency is monthly, so 12, for test it is almost 5 years, as its 57 time steps, 
    # for train it is 108 time steps so 12*9 so 9 years
    min = np.nanmin(ts,axis = 1)
    max = np.nanmax(ts,axis = 1)
    #I didn't understand, forkel used na.approx, which in turn used splines, how is that linear?
    #I'll probably use Akima1DInterpolator in scipy instead of na.approx 
    # It takes time and values, so lets see and is cubic spline
    # The interpolation method by Akima uses a continuously differentiable sub-spline built from piecewise cubic polynomials. 
    # The resultant curve passes through the given data points and will appear smooth and natural.
    # But we need a rapidly changing second derivative for a good use of Akima
    # Did anyone say, the fancier the better?, if not I'll say it

