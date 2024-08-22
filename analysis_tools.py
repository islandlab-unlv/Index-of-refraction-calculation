"""A collection of functions useful for analysis"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import os
# from lmfit import Parameters, minimize

mpl.rcParams.update({'font.size': 20,
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'figure.subplot.top': 0.95,
                     'figure.subplot.bottom': 0.13,
                     'figure.subplot.left': 0.165,
                     'figure.subplot.right': 0.94,
                     'figure.figsize': (9.6, 9.0)})

perme0 = 1.25663706212e-6 ## N/A**2 (vacuum permeability)
permi0 = 8.8541878128e-12 ## F/m (vacuum permittivity)
permiBN =      3.4*permi0 ## F/m [A Pierret et al 2022 Mater. Res. Express 9 065901]
charge_e = 1.60217663e-19 ## C
mass_e = 9.1093837015e-31 ## kg

def load_data(file, data_shape=None, skiprows=3, usecols=None, unpack=True, **kwargs):
    """
    Loads data file. Designed for .dat files, with a header, and 
    where each column of data corresponds to a different parameter.
    
    Arguments:
        file <str or file handle>:
            Location of the data file.
            
        data_shape <tuple of ints or None>:
            Shape for data to assume after loading. Data is typically created
            as some n-dimensional array of shape (a_1, a_2,...,a_n) then 
            flattened to fit in a text file. This parameter will reshape each 
            column to their original data shape. None will load the data 
            in one-dimension.
            
        skiprows <int or None>:
            Skip the first skiprows rows of the file, typically the header.
            Our convention uses 3 lines for the header.
            
        usecols <tuple of ints or None>:
            Simplifies loaded data to just the columns you want. "None" returns
            all columns.
            
        **kwargs: See documentation for numpy.loadtxt ("dtype=np.cdouble" can 
                  be useful for importing complex-valued data).
    
    Return:
        data <list of ndarrays>:
            List of arrays with the data of corresponding columns. Length will 
            correspond to length of usecols, or total columns if usecols was None.
    """
    data = np.loadtxt(file, delimiter='\t', skiprows=skiprows, 
                      usecols=usecols, unpack=unpack, **kwargs)
    if len(data.shape) == 1:
        data = [data]
    elif len(data.shape) == 2:
        data = list(data)
    if data_shape != None:
        for ii in range(len(data)):
            data[ii] = data[ii].reshape(data_shape)
    return data

def header_read(file, lines=3):
    with open(file, 'r') as hfile:
        header = [hfile.readline().strip() for _ in range(lines)]
        return header

def ez_plot(x_data, y_data, x_label, y_label, fig=None, ax=None, data_label=None):
    if fig == None or ax == None:
        fig, ax = plt.subplots()
    ax.plot(x_data, y_data, label=data_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    return fig, ax

def extenter(x, y):
    """Returns a tuple to work with plt.imshow's 'extent' argument. 'x' and 'y'
    are the values for the x- and y-axis of the colorplot."""
    dx = np.abs(np.mean(np.diff(x))/2)
    dy = np.abs(np.mean(np.diff(y))/2)
    if x.shape[0] == 1:
        dx = 0.5
    if y.shape[0] == 1:
        dy = 0.5
    return (np.min(x)-dx, np.max(x)+dx, np.min(y)-dy, np.max(y)+dy)

def imdata(data, extent=None, ax=None, cmap='seismic', aspect='auto', origin='lower', **kwargs):
    """Shorthand for using imshow to display data as a heatmaps"""
    if ax == None:
        return plt.imshow(data, extent=extent, cmap=cmap, aspect=aspect, origin=origin, **kwargs)
    else:
        return ax.imshow(data, extent=extent, cmap=cmap, aspect=aspect, origin=origin, **kwargs)

def ez_imdata(data, x_label, y_label, cb_label, fig=None, ax=None, figsize=mpl.rcParams['figure.figsize'], extent=None, cmap='seismic', interpolation='none', cb_aspect=20, **kwargs):
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    img = imdata(data, extent, ax=ax, cmap=cmap, interpolation=interpolation, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cb = plt.colorbar(img, ax=ax, label=cb_label, aspect=cb_aspect)
    plt.tight_layout()
    return fig, ax, cb

def nearest_index(value: float, array: np.ndarray):
    """Find index of element in array closest to value"""
    if len(array.shape) == 1:
        return (np.abs(array-value) == np.min(np.abs(array-value))).nonzero()[0][0]
    elif len(array.shape) == 2:
        return (np.abs(array-value) == np.min(np.abs(array-value))).nonzero()

def save_feature(folder, data, name):
    if not os.path.exists(folder+"\\Features"):
        os.makedirs(folder+"\\Features")
    np.savetxt(folder+'\\Features\\'+name+".txt", data.flatten())

def load_feature(file, data_shape=None):
    try:
        feat = np.loadtxt(file)
        if data_shape:
            return feat.reshape(data_shape)
        else:
            return feat
    except:
        raise Exception("Feature does not exist.")

def feature_extract(data, indices):
    return data[[ii for ii in range(data.shape[0])],[*indices]]

def linear(x, a, b):
    """Linear function for fitting slopes"""
    return a + b*x

def quadratic(x, a, b, c):
    """Quadratic function for fitting peaks"""
    return a + b*x + c*x**2

def quad_fit(x, y):
    """Fits the data then extracts peak height and location"""
    popt, pcov = curve_fit(quadratic, x, y)
    y_peak = popt[0] - 0.25*popt[1]**2/popt[2] ## y_peak = a - b^2/(4c)
    x_peak = -0.5*popt[1]/popt[2] ## x_peak = -b/(2c)
    y_error = np.sqrt(pcov[0,0]**2 + popt[1]**2/(8*popt[2]**2)*pcov[1,1]**2 + popt[1]**4/(16*popt[2]**4)*pcov[2,2]**2)
    x_error = np.abs(0.5*popt[1]/popt[2])*np.sqrt((pcov[1,1]**2)/(popt[1]**2) + (pcov[2,2]**2)/(popt[2]**2))
    return popt, pcov, y_peak, x_peak, y_error, x_error

def gaussfunc1D(x, amp, cen, sig):
    return amp/(sig*(2*np.pi)**0.5)*np.exp(-0.5*((x-cen)/sig)**2)

def gaussblur(x, data, sigma):
    """
    Blurs a 1D or 2D data set with a gaussian distribution.
    
    For 1D data, x should be a 1D array, and sigma a single float.
    
    For 2D data, x should be a tuple of two 1D arrays (x_axis, y_axis), and sigma can be a tuple of two floats (x_sigma, y_sigma)
    Each axis is blurred individually, line by line. Remember data.shape = (y, x)
    (Functionality for a true 2D blur will be done later)
    
    Arguments:
        - x (1D array or tuple of 1D arrays):
            The independent axis (axes) for the data
        - data (1D or 2D array):
            Data to be blurred
        - sigma (float or tuple of floats):
            Standard deviation of the gaussian distribution used to blur. 
            A value of 0 will skip blurring in that direction.
    """
    if len(data.shape) == 1: ## Handles 1D data
        if sigma == 0:
            return data
        ii_window = abs(round(4*sigma/np.mean(np.diff(x))))
        try:
            blur = np.zeros(data.shape)
            for ii, xx in enumerate(x):
                iia, iib = (ii-ii_window, ii+ii_window)
                if iia < 0:
                    iia = 0
                kern = gaussfunc1D(x[iia:iib], 1, xx, sigma)
                kern_norm = kern/np.sum(kern)
                blur[ii] = np.sum(kern_norm*data[iia:iib])
            return blur
        except TypeError:
            raise Exception(f"1D data must be given a single sigma. Given sigma was {sigma}.")
    elif len(data.shape) == 2: ## Handles 2D data
        ## Blur y axis
        if sigma[1] == 0:
            blury = data.copy()
        else:
            kerny = np.zeros([data.shape[0]]*2) ## (y,y')
            for jj, yy in enumerate(x[1]):
                kern = gaussfunc1D(x[1], 1, yy, sigma[1]) ## Each gaussian distribution in the y direction
                kerny[jj] = kern/np.sum(kern) ## Normalized weights, saved to y' axis
            kerny = np.tile(kerny.reshape(*kerny.shape,1), data.shape[1]) ## (y,y',x)
            blury = np.sum(kerny*data, axis=1) ## (y,x)
        ## Blur x axis
        if sigma[0] == 0:
            bluryx = blury.copy()
        else:
            kernx = np.zeros([data.shape[1]]*2) ## (x,x')
            for ii, xx in enumerate(x[0]):
                kern = gaussfunc1D(x[0], 1, xx, sigma[0]) ## Each gaussian distribution in the x direction
                kernx[ii] = kern/np.sum(kern) ## Normalized weights, saved to x' axis
            kernx = np.tile(kernx.reshape(kernx.shape[0],1,kernx.shape[1]), (data.shape[0],1)) ## (x,y,x')
            bluryx = np.sum(kernx*blury, axis=2).T ## (y,x)
        return bluryx
        
