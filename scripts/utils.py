import scipy
import numpy as np
from scipy import constants
from scipy import stats
import astropy.constants

def calc_Teff(luminosity, radius):
    return np.array(((luminosity.values*astropy.constants.L_sun) / (4*np.pi*constants.sigma*((radius.values*astropy.constants.R_sun)**2)))**0.25)

def calc_L(Teff, radius):
    return 4*np.pi*((radius.values*astropy.constants.R_sun)**2)*constants.sigma*(Teff**4)

def rescale_preds(preds, df, column):
    if 'star_feh' in column:
        return (preds[column+"_std"]*df[column].std())+df[column].mean()
    else:
        return 10**((preds["log_"+column+"_std"]*df["log_"+column].std())+df["log_"+column].mean())