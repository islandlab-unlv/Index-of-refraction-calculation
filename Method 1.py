import numpy as np
from reflection4media import reflection4media
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal
from lmfit import Parameters, minimize
try:
    import sys
    sys.path.append("Z:/Randy/Measurements/Analysis/")
    from analysis_tools import gaussblur, nearest_index
    blurr = True
except:
    print("analysis_tools does not exist. Will not be able to apply Gaussian blur.")
    blurr = False

    #### STEP 1: Load and crop data ####

data_file = 'Muscovite 2 Contrast.tsv'
flake_file = 'Muscovite 2 Thickness.txt'

# Load the file containing the data. It should be a .txt file with columns.
# The first column should correspond to the wavelength (in nm) and the rest of the
# columns should correspond to the experimental optical contrast. Each
# column for each Flake thickness studied
data = np.loadtxt(data_file)

# Load the file containing the SiO2 thickness (in nm) studied. The file
# should be in the same order as the columns in the data_file
dsio2 = 285

# Number of layers of the dataset
# Thickness in 'nm' of the samples studied
thexp =  np.loadtxt(flake_file)
# Target range of study
thickness_min = 0
thickness_max = 200

# Use flake if index i corresponds with use_flakes[i] == 1
dont_use_flakes = np.ones(len(thexp))
bad_indices = np.logical_and(thexp >= thickness_min, thexp <= thickness_max)
for i, is_good in enumerate(bad_indices):
    if not is_good:  # Check if the current element is False
        # Set the corresponding element in another_array to 0
        dont_use_flakes[i] = 0
dont_use_flakes[(),] = 0 ## don't use certain flakes
dont_use_flakes_inds = dont_use_flakes.nonzero()[0]
flake_count = np.sum(dont_use_flakes, dtype=int)
thexp = thexp[dont_use_flakes_inds]

# This crops the experimental dataset to the region of interest (in this
# example from 500 nm to 700 nm) Crops to rows in txt file.
ind_min = nearest_index(500, data[:,0]) 
ind_max = nearest_index(700, data[:,0])
downsample = 10 
lambdaexp = data[ind_min:ind_max:downsample,0]
Cexp = np.zeros((flake_count, len(lambdaexp)))

# Assign values from the second to the last column of data to Cexp
if blurr:
    for ii, jj in enumerate(dont_use_flakes_inds):
        Cexp[ii, :] = gaussblur(lambdaexp, data[ind_min:ind_max:downsample, jj+1], 5)
else:
    for ii, jj in enumerate(dont_use_flakes_inds):
        Cexp[ii, :] = data[ind_min:ind_max:downsample, jj+1]

    #### STEP 2: Establish Fresnel equation parameters ####

# Refractive index of air
n0 = 1
k0 = 0

# K of material set to 0
k1 = 0

# Refractive index of SiO2
def n_sio2(lambda_w):
    # https:../../refractiveindex.info../?shelf=main&book=SiO2&page=Malitson
    x = lambda_w
    p1 = -3.68e-08
    p2 = 8.219e-05
    p3 = 1.38
    p4 = 8.513
    p5 = 0.2227
    q1 = -11.52
    q2 = -0.4289
    n_sio2 = (p1*x**4 + p2*x**3 + p3*x**2 + p4*x + p5)/(x**2 + q1*x + q2)
    return n_sio2
n2 = n_sio2(lambdaexp)
k2 = 0

# Refractive index of Si
def nsi(lambda_w):
    # C. Schinke, P. C. Peest, J. Schmidt, R. Brendel, K. Bothe, M. R. Vogt, I. Kröger, S. Winter, A. Schirmacher, S. Lim, H. T. Nguyen, D. MacDonald. Uncertainty analysis for the coefficient of band-to-band absorption of crystalline silicon. AIP Advances 5, 67168 (2015)
    # Valid in the range 390 nm - 1100 nm
    x = lambda_w
    a0 = 553.179941792293
    a1 = -5.80676536899888
    a2 = 0.026936774694618
    a3 = -7.17691954078284e-05
    a4 = 1.20884605067916e-07
    a5 = -1.33448648761134e-10
    a6 = 9.65714830292706e-14
    a7 = -4.41950518031896e-17
    a8 = 1.16130143243018e-20
    a9 = -1.33582671927738e-24
    n_si = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5 + a6 * x ** 6 + a7 * x ** 7 + a8 * x ** 8 + a9 * x ** 9
    a0 = 1
    a1 = 1
    a2 = -17085125.1211601
    a3 = 9015717.57547127
    a4 = -10514.2162300923
    a5 = 16.4780721917572
    a6 = -0.0126338854309534
    b0 = 1
    b1 = 1
    b2 = 1
    b3 = 8253747.97018674
    b4 = -234136.932420437
    b5 = -494.349230637116
    b6 = 2.8888768704717
    k_si = (a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5 + a6 * x ** 6) / (b0 + b1 * x + b2 * x ** 2 + b3 * x ** 3 + b4 * x ** 4 + b5 * x ** 5 + b6 * x ** 6)
    return n_si, k_si
n3, k3 = nsi(lambdaexp)

# Thickness of the 2D material and the SiO2 layer
# d1 = thexp
# Set dsio2 to a constant
d2 = 285

    #### STEP 3: Fit contrast data vs thickness ####

def contrast_calc(wl, n0, k0, n1, k1, n2, k2, n3, k3, d1, d2):
    R = reflection4media(wl, n0, k0, n1, k1, d1, n2, k2, d2, n3, k3) ## flake
    R0 = reflection4media(wl, n0, k0, n0, k0, d1, n2, k2, d2, n3, k3) ## substrate 
    return (R - R0) / (R + R0)

# Calculate global residuals over the whole contrast spectra
def global_residuals(params, data, wl, n0, k0, n2, k2, n3, k3, d1, d2):
    n1 = params['n1']
    residuals = np.zeros(len (wl))
    for ii, it in enumerate(zip(wl, n2, n3, k3)):
        (wwl, nn2, nn3, kk3) = it ## (wavelength, SiO2 real refrac, Si real refrac, Si imag refrac)
        contrast = contrast_calc(wwl, n0, k0, n1, k1, nn2, k2, nn3, kk3, d1, d2)
        residuals[ii] = data[ii] - contrast
    return residuals.flatten()

# Establish containers
nn = np.zeros(flake_count)
Cfit = np.zeros((flake_count, len(lambdaexp)))
param_fits = np.empty((flake_count, len(lambdaexp)), dtype=object)
stderrors = np.empty((flake_count, len(lambdaexp), 2))
nofit_ind = []

# Calculate n for each flake, k = 0
for ii, d1 in enumerate(thexp):
    params = Parameters()
    params.add('n1', value=1.55, min=0.0, max=2.5,brute_step=0)
    wl = lambdaexp
    args = (Cexp[ii,:], wl, n0, k0, n2, k2, n3, k3, d1, d2)
    # Perform the global fit for the current flake
    result = minimize(global_residuals, params, args=args, method='least_squares')

    # Extract the best-fit value of n1
    best_n1 = result.params['n1'].value

    # Store the best-fit n1 for the current flake
    nn[ii] = best_n1
    print(f"{ii}: {nn[ii]} + j*0")
    Cfit[ii,:] = contrast_calc(wl, n0, k0, nn[ii], k1, n2, k2, n3, k3, d1, d2)

# Residuals
Cres = np.abs(Cfit - Cexp)
Cres_avg = np.mean(Cres, axis=1)
# n mean and std dev
n1 = np.mean(nn)
n1sd = np.std(nn)

# Print the results
print(f"Average of nn: {n1}")
print(f"Standard deviation of nn: {n1sd}")


    #### STEP 4: Plot selected flakes ####


# Load the file containing the data. It should be a .txt file with columns.
# The first column should correspond to the wavelength (in nm) and the rest of the
# columns should correspond to the experimental optical contrast. Each
# column for each Flake thickness studied
data = np.loadtxt(data_file)


thickness_min = 130  # Example value, replace with your desired minimum thickness
thickness_max = 120

thexp2 =  np.loadtxt(flake_file)


# Use flake if index i corresponds with use_flakes[i] == 1
use_flakes = np.zeros(len(thexp2))

good_indices = np.logical_and(thexp2 >= thickness_min, thexp2 <= thickness_max)


# Flakes used for plots in Fig4:
# Biotite 1 (47, 45, 43, 42, 38, 12, 32)
# Biotite 2 (40, 16, 17, 4, 13, 3, 6)
# Phlogopite (49, 25, 13, 20, 16, 36, 40)
# Muscovite 1 (36, 26, 35, 19, 0 , 11, 14)
# Muscovite 2 (2, 20, 40, 6, 19 , 21, 8)

use_flakes[(2, 20, 40, 6, 19 , 21, 8),] = 1 ## use certain flakes

for i, is_good in enumerate(good_indices):
    if is_good:  # Check if the current element is False
        # Set the corresponding element in another_array to 0
        use_flakes[i] = 1


use_flakes_inds = (use_flakes==1).nonzero()[0]
flake_count = np.sum(use_flakes, dtype=int)
thexp2 = thexp2[(use_flakes==1).nonzero()[0]]
Cres_avg = Cres_avg[(use_flakes==1).nonzero()[0]]


# This crops the experimental dataset to the region of interest (in this
# example from 450 nm to 800 nm) Crops to rows in txt file.
ind_min = nearest_index(500, data[:,0])
ind_max = nearest_index(700, data[:,0])
lambdaexp2 = data[ind_min:ind_max,0]
Cexp2 = np.zeros((flake_count, ind_max-ind_min))
n3, k3 = nsi(lambdaexp2)
n2 = n_sio2(lambdaexp2)
# Assign values from the second to the last column of data to Cexp
if blurr:
    for ii, jj in enumerate((use_flakes==1).nonzero()[0]):
        Cexp2[ii, :] = gaussblur(lambdaexp2, data[ind_min:ind_max, jj+1], 5)
else:
    for ii, jj in enumerate((use_flakes==1).nonzero()[0]):
        Cexp2[ii, :] = data[ind_min:ind_max, jj+1]

def contrast_calc2(wl, n0, k0, n1, k1, n2, k2, n3, k3, d1, d2):
    R = reflection4media(wl, n0, k0, n1, k1, d1, n2, k2, d2, n3, k3)
    R0 = reflection4media(wl, n0, k0, n0, k0, d1, n2, k2, d2, n3, k3)
    return (R - R0) / (R + R0)

sorted_indices = np.argsort(thexp2)  # Get indices for sorting in descending order
sorted_thexp2 = thexp2[sorted_indices]
sorted_Cexp = Cexp2[sorted_indices]

fig, ax1 = plt.subplots(figsize=(5, 5))  # Create a single subplot

# Create a gradient from dark blue to light blue to light purple to dark purple
gradient_colors = ['#0504aa', '#072CFA', '#0d75f8', '#7bc8f6', '#8e8ce7', '#510ac9', '#490DAD', '#4B0082']
Cfit2 = np.zeros((flake_count, len(lambdaexp2)))
Cfit_upper = np.zeros((flake_count, len(lambdaexp2)))
Cfit_lower = np.zeros((flake_count, len(lambdaexp2)))

# Define xytext values for manual adjustment
# xytext_values = [(-10, 4), (-18, 4), (-18, 4), (-18, 6), (-18, 6), (-26, 6), (-26, 6)]

contrast_diff = []

for ii, d1 in enumerate(thexp2):
    Cfit2[ii,:] = contrast_calc2(lambdaexp2, n0, k0, n1, k1, n2, k2, n3, k3, d1, d2)
    Cfit_upper[ii,:] = contrast_calc2(lambdaexp2, n0, k0, n1 + n1sd + Cres_avg[ii], k1, n2, k2, n3, k3, d1, d2)
    #print(n1sd + Cres_avg[ii])
    Cfit_lower[ii,:] = contrast_calc2(lambdaexp2, n0, k0, n1 - (n1sd + Cres_avg[ii]), k1, n2, k2, n3, k3, d1, d2)
for i in range(flake_count):
    offset = i * 0.5
    sorted_Cexp[i] = sorted_Cexp[i] + offset
    Cfit_offset = Cfit2[sorted_indices[i]] + offset
    Cfit_upper_offset = Cfit_upper[sorted_indices[i]] + offset
    Cfit_lower_offset = Cfit_lower[sorted_indices[i]] + offset

    color = gradient_colors[i]
    ax1.plot(lambdaexp2, sorted_Cexp[i], color=color, alpha=1, linestyle='', marker='x', markevery=100)
    
    # Plot the fitted contrast with confidence intervals
    ax1.plot(lambdaexp2, Cfit_offset, linestyle='-', color=color, alpha=0.7)
    ax1.fill_between(lambdaexp2, Cfit_lower_offset, Cfit_upper_offset, color=color, alpha=0.4)
    # Annotate line labels next to the lines
    label_x = lambdaexp2[-1]  # Adjust the label position as needed
    label_y = sorted_Cexp[i][-1]  # Label position at the end of the line
    rounded_thickness = int(round(sorted_thexp2[i]))
    # ax1.annotate(f"{rounded_thickness}", (label_x, label_y), textcoords="offset points", xytext=xytext_values[i],
    #              ha='left', fontsize=14, color=color)
    ax1.annotate(f"{rounded_thickness} nm", (label_x, label_y), textcoords="offset points", xytext=(10, 0),  # Adjust xytext for side position
             ha='left', fontsize=14, color=color)
    # Calculate contrast difference
    contrast_diff.append(np.max(sorted_Cexp[i]) - np.min(sorted_Cexp[i]))

# Plot contrast difference
ax1.set_xlabel('Wavelength (nm)', fontsize=18)
ax1.set_ylabel('Contrast', fontsize=18)

# Set the font size for both x and y axes on ax1
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_xticks([500, 550, 600, 650, 700])
ax1.set_xticklabels([500, 550, 600, 650, 700])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Set the plot limits to accommodate labels outside the graph
ax1.set_xlim(lambdaexp.min(), 700)
plt.tight_layout()

plt.show()
# Save data
#result_data = np.column_stack((thexp, nn))
#np.savetxt('Muscovite_2_thickness_n.txt', result_data, header='Thickness (nm)  n', delimiter='\t',fmt='%.2f\t%.5f')