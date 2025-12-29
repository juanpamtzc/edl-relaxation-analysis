from scipy.interpolate import UnivariateSpline
import numpy as np
from typing import Optional

def find_critical_points_via_spline_fitting(x: np.array, y: np.array, smoothing_factor: Optional[float] = 1.0, spline_order: Optional[int] = 3) -> tuple:
    """
    Fits smoothing spline to the data given.

    Args:
        x:                  numpy array with x values
        y:                  numpy array with y values (same shape as x)
        smoothing_factor:   float representing the smoothing factor (no smoothing at s=0, smoother as s increases)
        spline_order:       int representing spline order (default is cubic, k=3)

    Returns:
        x_minima:          list of x values where local minima occur
        y_minima:          list of y values where local minima occur
        x_maxima:          list of x values where local maxima occur
        y_maxima:          list of y values where local maxima occur
        x_saddle_points:   list of x values where saddle points occur
        y_saddle_points:   list of y values where saddle points occur
        spline:            the fitted UnivariateSpline object
    """

    # fit spline
    spline = UnivariateSpline(x,y,k=spline_order,s=smoothing_factor)

    # take the first and second derivatives of the spline
    dspline = spline.derivative()
    d2spline = spline.derivative(2)

    # find the critical points of the spline by finding the roots of the derivative
    critical_points = dspline.roots()
    x_minima = [critical_point for critical_point in critical_points if d2spline(critical_point)>0]
    x_maxima = [critical_point for critical_point in critical_points if d2spline(critical_point)<0]
    x_saddle_points = [critical_point for critical_point in critical_points if d2spline(critical_point)==0]

    y_minima = spline(x_minima)
    y_maxima = spline(x_maxima)
    y_saddle_points = spline(x_saddle_points)

    return x_minima, y_minima, x_maxima, y_maxima, x_saddle_points, y_saddle_points, spline
