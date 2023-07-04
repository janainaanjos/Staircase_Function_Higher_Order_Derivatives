"""
A Python program computes the non-regularized and regularized directional higher-order derivatives, the S-function of the regularized directional 
derivatives, and the regularization parameters on gridded data.

This code is released from the Master's thesis: "Regularization parameters in aeromagnetic data processing using differential operators".

The program is under the conditions terms in the file README.txt.

author:Janaína A. Melo (IAG-USP)
email: janaina.melo@usp.br (J.A. Melo)
"""


import numpy as np
from sklearn.linear_model import LinearRegression




def pad_data(data, shape):

    """
    Padded data until reaches the length of the next higher power of two, and the pad values are the edge values. 

    Based on Fatiando a Terra package (https://www.fatiando.org/).
    
    Parameters:
        
    * data: 1D-array
        input data set 
    * shape: tuple = (nx, ny)
        data points number in each direction 
        
    Returns:
        
    * padded_data: 2D-array
        data set padded
    * padx: float
        x-direction padded
    * pady: float
        y-direction padded
    """

    nx, ny = shape
    n_points=int(2**(np.ceil(np.log(np.max(shape))/np.log(2))))

    padx = (n_points - nx) // 2
    pady = (n_points - ny) // 2

    # Pads the matrix edges
    padded_data = np.pad(data.reshape(shape), ((padx, padx), (pady, pady)), mode='edge')

    return padded_data, padx, pady




def fft_wavenumbers(x, y, shape, padshape):

    """
    Computes the wavenumbers 2D-arrays.

    Based on Fatiando a Terra package (https://www.fatiando.org/).
    
    Parameters:

    * x, y: 2D-array
        coordinates mesh in x- and y-directions
    * shape: tuple = (nx, ny)
        data points number in each direction before padding
    * padshape: tuple = (nx, ny)
        data points number in each direction after padding             
    
    Returns:

    * kx, ky: 2D-array
        wavenumbers in x- and y-directions
    """

    nx, ny = shape

    # Discretization range
    dx = (x.max() - x.min()) / (nx - 1)
    dy = (y.max() - y.min()) / (ny - 1)

    # Wavenumbers in the x- and y-directions
    kx = 2 * np.pi * np.fft.fftfreq(padshape[0], dx)
    ky = 2 * np.pi * np.fft.fftfreq(padshape[1], dy)

    return np.meshgrid(ky, kx)[::-1]




def nonregularized_derivative(x, y, data, shape, order):

    """
    Computes the non-regularized directional derivatives in the Fourier domain in the x-, y-, and z-directions using equations 3 and 4 of the thesis.

    Based on Fatiando a Terra package (https://www.fatiando.org/).

    Parameters:

    * x, y: 1D-array
        coordinates mesh in x- and y-directions
    * data: 1D-array
        input data set
    * shape: tuple = (nx, ny)
        data points number in each direction 
    * order: integer
        derivative order

    Returns:

    * dx, dy, dz: 1D-array
        derivatives in x-, y- and z-directions
    """

    nx, ny = shape

    # Fills the matriz edges 
    padded, padx, pady = pad_data(data, shape)

    # Wavenumbers in x- and y-directions
    kx, ky = fft_wavenumbers(x, y, shape, padded.shape)  

    # Calculates the derivatives in the Fourier domain
    derivx_fft = np.fft.fft2(padded) * ((kx * 1j) ** order)
    derivy_fft = np.fft.fft2(padded) * ((ky * 1j) ** order)
    derivz_fft = np.fft.fft2(padded) * (np.sqrt(kx ** 2 + ky ** 2) ** order)

    # np.real: returns the real part of the complex argument
    # np.fft.ifft2: calculates the two-dimensional inverse discrete Fourier transform
    derivx_pad = np.real(np.fft.ifft2(derivx_fft))
    derivy_pad = np.real(np.fft.ifft2(derivy_fft))
    derivz_pad = np.real(np.fft.ifft2(derivz_fft))

    # Removes the padding in derivative
    derivx = derivx_pad[padx: padx + nx, pady: pady + ny]
    derivy = derivy_pad[padx: padx + nx, pady: pady + ny]
    derivz = derivz_pad[padx: padx + nx, pady: pady + ny]

    # Converts a matrix to a 1D vector
    dx = np.ravel(derivx)
    dy = np.ravel(derivy)
    dz = np.ravel(derivz)

    return dx, dy, dz




def regularized_derivative(x, y, data, shape, order, alpha):

    """
    Computes the regularized directional derivatives in the Fourier domain in the x-, y-, and z-directions using equations 31, 32 and 35 of the thesis.

    Parameters:

    * x, y: 1D-array
        coordinates mesh in x- and y-directions
    * data: 1D-array
        input data set
    * shape: tuple = (nx, ny)
        data points number in each direction 
    * order: integer
        derivative order
    * alpha: float
        regularization parameter

    Returns:

    * dx, dy, dz: 1D-array
        derivatives in x-, y- and z-directions
    """
    
    nx, ny = shape

    # Fills the matriz edges 
    padded, padx, pady = pad_data(data, shape)
    
    # Wavenumbers in x-, y- and z-directions
    kx, ky = fft_wavenumbers(x, y, shape, padded.shape)  
    kz = np.sqrt(kx ** 2 + ky ** 2)

    # Derivative operator
    gamma_y = ((1j*ky)**order)/(1-alpha*((1j*ky)**(order+1)))
    gamma_x = ((1j*kx)**order)/(1-alpha*((1j*kx)**(order+1)))
    gamma_z = (kz**order) / (1 + alpha * (kz **(2*order)))

    # Calculates the derivatives in the Fourier domain
    derivx_fft = np.fft.fft2(padded) * gamma_x
    derivy_fft = np.fft.fft2(padded) * gamma_y 
    derivz_fft = np.fft.fft2(padded) * gamma_z 

    # np.real: returns the real part of the complex argument
    # np.fft.ifft2: calculates the two-dimensional inverse discrete Fourier transform
    derivx_pad = np.real(np.fft.ifft2(derivx_fft))
    derivy_pad = np.real(np.fft.ifft2(derivy_fft))
    derivz_pad = np.real(np.fft.ifft2(derivz_fft))

    # Removes the padding in derivative
    derivx = derivx_pad[padx: padx + nx, pady: pady + ny]
    derivy = derivy_pad[padx: padx + nx, pady: pady + ny]
    derivz = derivz_pad[padx: padx + nx, pady: pady + ny]

    # Converts a matrix to a 1D vector
    dy = np.ravel(derivy)
    dx = np.ravel(derivx)
    dz = np.ravel(derivz)

    return dx, dy, dz




def s_function_derivative(x, y, data, shape, alpha, order):

    """
    Computes the normalized Euclidean norm of the directional derivatives to different regularization parameter values using equations 
    36 and 37 of the thesis.

    Parameters:

    * x, y: 1D-array
        coordinates mesh in x- and y-directions
    * data: 1D-array
        input data set
    * shape: tuple = (nx, ny)
        data points number in each direction 
    * alpha: 1D-array
        trial regularization parameters
    * order: integer
        derivative order

    Returns:

    * norm_sol_dx, norm_sol_dy, norm_sol_dz: 1D-array
        normalized Euclidean norm of the x-, y- and z-derivatives to different regularization parameter values
    """

    norm_sol_dx = []
    norm_sol_dy = []
    norm_sol_dz = []

    for i in range(len(alpha)):

        dx, dy, dz = regularized_derivative(x, y, data, shape, order, alpha[i])

        soma_x = 0
        soma_y = 0
        soma_z = 0

        for i in range(len(dx)):

            elem_dx = dx[i] ** 2
            elem_dy = dy[i] ** 2
            elem_dz = dz[i] ** 2

            soma_x = soma_x + elem_dx
            soma_y = soma_y + elem_dy
            soma_z = soma_z + elem_dz

        norm_dx = np.sqrt(soma_x)
        norm_dy = np.sqrt(soma_y)
        norm_dz = np.sqrt(soma_z)

        norm_sol_dx.append(norm_dx)
        norm_sol_dy.append(norm_dy)
        norm_sol_dz.append(norm_dz)

    norm_sol_dx = np.ravel(norm_sol_dx)
    norm_sol_dy = np.ravel(norm_sol_dy)
    norm_sol_dz = np.ravel(norm_sol_dz)

    norm_sol_dx = norm_sol_dx/max(norm_sol_dx)
    norm_sol_dy = norm_sol_dy/max(norm_sol_dy)
    norm_sol_dz = norm_sol_dz/max(norm_sol_dz)

    return norm_sol_dx, norm_sol_dy, norm_sol_dz




def regularization_parameter(norm_sol, alpha_test, upper_limit, inferior_limit, value_norm):

    """
    Determines the regularization parameter associated with a Euclidean norm-specific value located in the S-function sloped portion. 
    As the S-function is evaluated for regularization parameter discrete values, it is fitted a linear approximation in the S-function 
    sloped portion considering a specific interval [inferior limit, upper limit].

    Parameters:

    * norm_sol: 1D-array
        Euclidean norm of the regularized derivatives
    * alpha_test: 1D-array
        trial regularization parameters
    * upper_limit: float
        upper limit of the S-function's linear variation interval 
    * inferior_limit: float
        inferior limit of the S-function's linear variation interval 
    * value_norm: float
        Euclidean norm-specific value

    Returns:

    * alpha_value: float
        regularization parameter associate with Euclidean norm-specific value
    """

    norm = []
    alpha = []

    # Defines the linear variation vector of the S-function and the vector of associated regularization parameters
    for i in range (len(norm_sol)):

        if  (inferior_limit <= np.round(norm_sol[i],1) <= upper_limit):
            norm.append(norm_sol[i])
            alpha.append(alpha_test[i])

    alpha = np.array(alpha).reshape(-1,1)

    # Fits a linear regression model
    linreg = LinearRegression().fit(alpha, norm)

    # calculates the angular and linear coefficients for the linear regression
    a = linreg.coef_
    b = linreg.intercept_

    # Calculates the regularization parameter associated with a particular Euclidean norm value
    alpha_value = np.log10((value_norm - b)/a)

    return alpha_value




def value_sfunction(alpha, norm_sol, value_parameter):

    """
    Returns the Euclidean norm value associated with a specific regularization parameter.

    Parameters:

    * alpha: 1D-array
        trial regularization parameters
    * norm_sol: 1D-array
        Euclidean norm of the regularized derivatives
    * value_parameter: float
        regularization parameter specific value

    Returns:

    * value_norm: float
        Euclidean norm value associated with the configured regularization parameter 
    """

    alpha = alpha.tolist()

    index = alpha.index(value_parameter)

    value_norm = norm_sol[index]    

    return value_norm