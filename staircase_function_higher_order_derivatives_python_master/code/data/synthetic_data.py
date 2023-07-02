"""
Synthetic data test 

Python script to generate the synthetic results. The script loads the total-field anomaly of a synthetic model from the file "synthetic_data.dat"
and computes the S-function of the regularized higher-order derivatives, regularization parameters, and non-regularized and regularized higher-order 
derivatives using the functions in "filtering.py". The figures are generated using the function "plot_figure.py".

This code is released from the Master's thesis: "Regularization parameters in aeromagnetic data processing using differential operators".

The program is under the conditions terms in the file README.txt.

authors:Janaína A. Melo (IAG-USP)
email: janaina.melo@usp.br (J.A. Melo)
"""

"""
Input:

- input/synthetic_data.dat: 2D-array with "n" rows by 4 columns, where "n" rows correspond to the size of the data.
            x-coordinate, y-coordinate, z-coordinate, total-field anomaly.

- input/batholith_vertices: 2D-array with "n" rows by 2 columns.
		    x and y coordinates of the batholith vertices to plot on synthetic data figures.
  
- input/dike_vertices: 2D-array with "n" rows by 2 columns.
		    x and y coordinates of the dike vertices to plot on synthetic data figures.

- input/sill_vertices: 2D-array with "n" rows by 2 columns.
		    x and y coordinates of the sill vertices to plot on synthetic data figures.

Parameters:

- Variation range of the trial regularization parameters: 
            alpha_test - 1D-array 

- Linear variation interval limits of the S-function:
            inferior_limit - float
            upper_limit - float
"""


import numpy as np
from filtering import *
from plot_figure import *




# Input data
data = np.loadtxt("input/synthetic_data.dat")

x = data[:,0]                         # x coordinates (m)
y = data[:,1]                         # y coordinates (m)
z = data[:,2]                         # z coordinates (m)
tfa = data[:,3]                       # total-field anomaly (nT)

inc, dec = -15, 30                    # geomagnetic inclination and declination (degrees)

area = (5000, 25000, 5000, 25000)     # (x1, x2, y1, y2) - mesh boundaries
nx, ny = 100, 100                     # number of points in the x and y axis
shape = (nx, ny)


# Polygon vertices (m)
dike = np.loadtxt("input/dike_vertices.dat")
sill = np.loadtxt("input/sill_vertices.dat")
batholith = np.loadtxt("input/batholith_vertices.dat")

model = [dike, sill, batholith]




'''
STAIRCASE FUNCTION OF THE DIRECTIONAL SECOND-ORDER DERIVATIVES
'''

# The user establishes the interval of the trial regularization parameters
l = np.arange(-2,15.5,0.5)
alpha_test = 10**(l[:])


# Calculates the Euclidean norm of the regularized directional second-order derivatives to different regularization parameters
norm_sol_dxx, norm_sol_dyy, norm_sol_dzz = s_function_derivative(x, y, tfa, shape, alpha_test, order=2)

'''The user establishes the interval limits in which the S-function presents a linear variation to determine the regularization parameter 
associated with S=0.50'''
value_norm = 0.50
upper_limit = 0.80
inferior_limit = 0.45

# Determines the regularization parameter associated with S=0.50
alpha_x = regularization_parameter(norm_sol_dxx, alpha_test, upper_limit, inferior_limit, value_norm)

# Prints the exponent of the regularization parameter
print(np.round(alpha_x, 1))

# Values of the S-function to configured regularization parameters by user

value_norm1 = value_sfunction(alpha_test, norm_sol_dxx, value_parameter=10**(8))
value_norm2 = value_sfunction(alpha_test, norm_sol_dxx, value_parameter=10**(9))

print(value_norm1,value_norm2)




'''
SECOND-ORDER DERIVATIVES
'''

# Non-regularized directional second-order derivatives (nT/m²) of the total-field anomaly 
dxx_tfa, dyy_tfa, dzx_tfa = nonregularized_derivative(x, y, tfa, shape, order=2)

# Regularized directional second-order derivatives (nT/m²) of the total-field anomaly
reg_dxx_tfa, reg_dyy_tfa, reg_dzz_tfa = regularized_derivative(x, y, tfa, shape, order=2, alpha=10**(alpha_x))
reg1_dxx_tfa, reg1_dyy_tfa, reg1_dzz_tfa = regularized_derivative(x, y, tfa, shape, order=2, alpha=10**(8))
reg2_dxx_tfa, reg2_dyy_tfa, reg2_dzz_tfa = regularized_derivative(x, y, tfa, shape, order=2, alpha=10**(9))




'''
PLOT THE FIGURES
'''

# Plot the total-field anomaly and the non-regularized and regularized second-order x-derivatives - Figure 1
plot_figure1(x, y, tfa, reg_dxx_tfa, reg1_dxx_tfa, reg2_dxx_tfa, model)

# Plot the S-function of the second-order x-derivative - Figure 2
plot_figure2(alpha_test, norm_sol_dxx)


