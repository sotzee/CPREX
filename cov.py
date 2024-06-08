#from filterpy.stats package

import scipy.linalg as linalg
import math
import numpy as np

def cov_ellipse(cov, deviations=1):
    U, s, _ = linalg.svd(cov)
    orientation = np.arctan2(U[1, 0], U[0, 0])
    width = deviations * np.sqrt(s[0])
    height = deviations * np.sqrt(s[1])
    return width,height,orientation

def cov_ellipse_xy(mean,cov, theta=np.linspace(0,2*np.pi,100), deviations=1):
    """
    Returns a array defining the ellipse representing the 2 dimensional
    covariance matrix cov and mean.

    Parameters
    ----------

    cov : nd.array shape (2,2)
       covariance matrix

    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.

    Returns x,y
    """

    width,height,orientation=cov_ellipse(cov,deviations=deviations)

    x_=width *np.cos(theta)
    y_=height*np.sin(theta)
    r =np.sqrt(x_**2+y_**2)
    theta_=np.arctan2(y_,x_)
    x =r*np.cos(orientation+theta_)+mean[0]
    y =r*np.sin(orientation+theta_)+mean[1]

    return np.array([x,y])
