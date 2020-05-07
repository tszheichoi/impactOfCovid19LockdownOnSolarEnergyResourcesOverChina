import scipy as sp
import numpy as np
import scipy.interpolate

def log_interp1d(xx, yy, kind = 'linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    # lin_interp = sp.interpolate.interp1d(logx, logy, kind = kind, fill_value = "extrapolate")
    lin_interp = sp.interpolate.UnivariateSpline(logx, logy, k = 1, s = 0, ext = 0, check_finite = True)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp
