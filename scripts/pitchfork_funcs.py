import pickle
import numpy as np
import tensorflow as tf
import os
import gc

import ultranest

import logging
logging.getLogger('ultranest').setLevel(logging.WARNING)

from ultranest import ReactiveNestedSampler

from ultranest.popstepsampler import PopulationSliceSampler, generate_cube_oriented_direction

import scipy
from scipy import constants
from scipy import stats
import astropy.constants


class InversePCA(tf.keras.layers.Layer):
    """
    Inverse PCA layer for tensorflow neural network
    
    Usage:
        - Define dictionary of custom objects containing Inverse PCA
        - Use arguments of PCA mean and components from PCA of output parameters for inverse PCA (found in JSON dict)
        
    Example:

    > f = open("pcann_info.json")
    >
    > data = json.load(f)
    >
    > pca_comps = np.array(data["pca_comps"])
    > pca_mean = np.array(data["pca_mean"])
    > 
    > custom_objects = {"InversePCA": InversePCA(pca_comps, pca_mean)}
    > pcann_model = tf.keras.models.load_model("pcann_name.h5", custom_objects=custom_objects)
    
    """
    
    def __init__(self, pca_comps, pca_mean, **kwargs):
        super(InversePCA, self).__init__()
        self.pca_comps = pca_comps
        self.pca_mean = pca_mean
        
    def call(self, x):
        y = tf.tensordot(x, np.float32(self.pca_comps),1) + np.float32(self.pca_mean)
        return y
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pca_comps': self.pca_comps,
            'pca_mean': self.pca_mean
        })
        return config

class WMSE(tf.keras.losses.Loss):
    """
    Weighted Mean Squared Error Loss Function for tensorflow neural network
    
    Usage:
        - Define list of weights with len = labels
        - Use weights as arguments - no need to square, this is handled in-function
        - Typical usage - defining target precision on outputs for the network to achieve, weights parameters in loss calculation to force network to focus on parameters with unc >> weight.
    
    """
    
    def __init__(self, weights, name = "WMSE",**kwargs):
        super(WMSE, self).__init__()
        self.weights = np.float32(weights)
        
    def call(self, y_true, y_pred):
        loss = ((y_true - y_pred)/(self.weights))**2
        return tf.math.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'weights': self.weights
        })
        return config

def WMSE_metric(y_true, y_pred):
    metric = ((y_true - y_pred)/(weights))**2
    return tf.reduce_mean(metric)


class emulator:
    def __init__(self, emulator_name, file_path='pitchfork/'):
        self.emulator_name = emulator_name
        self.file_path = file_path + self.emulator_name
        
        with open(self.file_path+".pkl", 'rb') as fp:
             self.emulator_dict = pickle.load(fp)

        self.log_inputs_mean = np.array(self.emulator_dict["data_scaling"]["inp_mean"][0])
        
        self.log_inputs_std = np.array(self.emulator_dict["data_scaling"]["inp_std"][0])

        self.log_outputs_mean = np.array(self.emulator_dict["data_scaling"]["classical_out_mean"][0] + self.emulator_dict["data_scaling"]["astero_out_mean"][0])
        
        self.log_outputs_std = np.array(self.emulator_dict["data_scaling"]["classical_out_std"][0] + self.emulator_dict["data_scaling"]["astero_out_std"][0])
            
        self.custom_objects = {"InversePCA": InversePCA(self.emulator_dict['custom_objects']['inverse_pca']['pca_comps'], self.emulator_dict['custom_objects']['inverse_pca']['pca_mean']),"WMSE": WMSE(self.emulator_dict['custom_objects']['WMSE']['weights'])}

        self.model = tf.keras.models.load_model(self.file_path+".h5", custom_objects=self.custom_objects)

        [print(str(key).replace("log_","") + " range: " + "[min = " + str(self.emulator_dict['parameter_ranges'][key]["min"]) + ", max = " + str(self.emulator_dict['parameter_ranges'][key]["max"]) + "]") for key in self.emulator_dict['parameter_ranges'].keys()];

    def predict(self, input_data, n_min=6, n_max=40, verbose=False):
        
        log_inputs = np.log10(input_data)
        
        standardised_log_inputs = (log_inputs - self.log_inputs_mean)/self.log_inputs_std

        standardised_log_outputs = self.model(standardised_log_inputs)

        standardised_log_outputs = np.concatenate((np.array(standardised_log_outputs[0]),np.array(standardised_log_outputs[1])), axis=1)

        log_outputs = (standardised_log_outputs*self.log_outputs_std) + self.log_outputs_mean

        outputs = 10**log_outputs

        outputs[:,2] = log_outputs[:,2] ##we want star_feh in dex

        teff = np.array(((outputs[:,1]*astropy.constants.L_sun) / (4*np.pi*constants.sigma*((outputs[:,0]*astropy.constants.R_sun)**2)))**0.25)
        
        outputs[:,0] = teff
        
        outputs = np.concatenate((np.array(outputs[:,:3]), np.array(outputs[:,n_min-3:n_max-2])), axis=1)

        return outputs

class nested_sampler():
    def __init__(self, priors, observed_vals, pitchfork, log_sigma_det, sigma_inv, nu_max, n_min=6, n_max=40, logl_scale = 1):
        self.priors = priors
        self.obs_val = observed_vals
        self.ndim = len(priors)
        self.pitchfork = pitchfork
        self.logl_scale = logl_scale
        self.logl_factor = -(len(observed_vals)*0.5*np.log(2*np.pi))-(0.5*log_sigma_det)
        self.sigma_inv = sigma_inv
        self.n_min = n_min
        self.n_max = n_max
        self.nu_max = nu_max
        self.matmul_path = np.einsum_path('ij, jk, ik->i', np.array(observed_vals).reshape(-1,1), sigma_inv, np.array(observed_vals).reshape(-1,1), optimize='optimal')[0]
    
    def ptform(self, u):

        theta = np.array([self.priors[i].ppf(u[:,i]) for i in range(self.ndim)]).T
        return theta

    def surf_corr(self, freqs, a, b):
        return freqs + a*((freqs/self.nu_max)**b)      
    
    def logl(self, theta):

        m = self.pitchfork.predict(theta[:,:-2], n_min=self.n_min, n_max=self.n_max)

        a_arr = np.expand_dims(theta[:,-2],1)

        b_arr = np.expand_dims(theta[:,-1],1)

        m_corrected = m.copy()
        
        m_corrected[:,3:] = self.surf_corr(m[:,3:],a_arr, b_arr)
        
        
        residual_matrix = np.array(m_corrected-self.obs_val)

        ll = self.logl_factor-0.5*np.einsum('ij, jk, ik->i', residual_matrix, self.sigma_inv, residual_matrix, optimize=self.matmul_path)

        return self.logl_scale * ll

    def __call__(self, ndraw_min, ndraw_max, draw_multiple=True):

        if hasattr(self, 'sampler') and self.sampler is not None:
            del self.sampler
            gc.collect()

        self.sampler = ultranest.ReactiveNestedSampler(['initial_mass', 'initial_Zinit', 'initial_Yinit', 'initial_MLT', 'star_age','a','b'], self.logl, transform = self.ptform, vectorized=True, ndraw_min=ndraw_min, ndraw_max=ndraw_max, draw_multiple=draw_multiple)
        
        return self.sampler

    def cleanup(self):
        if hasattr(self, 'sampler') and self.sampler is not None:
            del self.sampler
        if hasattr(self, 'pitchfork'):
            del self.pitchfork

        tf.keras.backend.clear_session()
        gc.collect()
