# Regularization parameter in the aeromagnetic data processing using differential operators

by Janaína A. Melo (IAG-USP)

Email: janaina.melo@usp.br (J.A. Melo)

This repository contains several Python codes and a synthetic data example for a Master's thesis about the open-source package staircase function. The synthetic example data reproduce the figures shown in the thesis. The codes 'filtering.py', 'synthetic_data.py',  and 'plot_figure.py' are compatible with both Python 2.7 and Python 3.7 programming languages. To run the main program "synthetic_data.py", the numpy library is required. In addition, the numpy and sklearn.linear_model libraries are necessary to run the complementary code "filtering.py" and the matplotlib and numpy libraries are required to run the script "plot_figure.py".


1 - Abstract

2 - Content

3 - Prerequisites

4 - Parameterization

5 - Running the files

6 - License

===============================================================================


1 - Abstract
----------------------
The evaluation of numerical derivatives is an essential tool in magnetic data processing, aiming to map structural lineaments and estimate the depth of the respective anomalous sources in the subsurface. Generally, directional derivatives are obtained through numerical methods based on the calculation of the Fourier transform, susceptible to noise amplification due to numerical instability. One way to improve stability in differentiation is to apply Tikhonov regularization to balance the oscillatory characteristics of the derivatives with the smoothing degree associated with the particular choice of regularization parameter. In this work, a graphical procedure is proposed to estimate regularization parameters for different potential field transformations that require first or second-order derivatives. This tool is based on the normalization of the L2-norm of the respective transformed fields to a sequence of trial regularization parameters, resulting in a characteristic function with a staircase format. This function has smooth and monotonic behavior, decreasing from 1 to 0 for increasing regularization values, in which the upper step (1) of the function is associated with non-regularized and sub-regularized transformations and the lower step (0) corresponds to over-regularized transformations. Synthetic tests simulating models with different noise levels or anomaly complexities illustrated that the well-suited selection of the regularization parameter in a characteristic interval can be associated with the properties of the processed transformed fields (presence or absence of distortions in the data) or with erroneous depth inferences and overrated. The applicability of the regularization procedure is evaluated on mesh aeromagnetic data covering two study areas in the Tocantins Province, central Brazil. In I Area, covering the Anápolis-Itauçu Complex, transformations using first-order derivatives regularized under the intermediate ramp criterion were efficient in mapping the continuity of magnetic lineaments with different directions and intersections, associated with shear zones, geological faults, and intrusive bodies. Applications in II Area covering the Transbrasiliano tectonic corridor revealed the need for a regularization low dose to obtain depth estimates consistent with the depths of the underlying basement of the Bananal Basin, according to available information from seismic lines and gravimetric models. Regularization tuned to the intermediate ramp criterion was sufficient for transformations with first-order derivatives to map the complex pattern of multiple linear structures. The results in II Area showed that transformations based on second-order derivatives require a high degree of regularization, detecting the individual contribution of subtle structural features.


2 - Content
----------------------

Python codes:

	- filtering.py:
		General Python module containing the functions to compute the non-regularized and 
        	regularized directional higher-order derivatives, S-function of the regularized 
		directional higher-order derivatives, and regularization parameters.
	
	- synthetic_data.py:
		Python script to generate the synthetic results. The script loads the total-field 
		anomaly of a synthetic model from the file "synthetic_data.dat" and computes the 
		S-function of the regularized higher-order derivatives, regularization parameters, 
		and non-regularized and regularized higher-order derivatives using the functions 
		in "filtering.py". The figures are generated using the function "plot_figure.py".

	- plot_figure.py:
		Python script to generate the figures of the synthetic data.
	
Outputs (folders): 
 
	- figures - figures of the synthetic data will be saved in this folder. The figures 
		    generated may differ from the Master's thesis and can be adapted in the script 
		    'plot_figure.py'.
						
Test data:

	- input/synthetic_data.dat
		    Synthetic total-field anomaly data are available by Uieda et al. (2014) at 
		    http://github.com/pinga-lab/paper-tle-euler-tutorial.	

Complementary files:

	- input/batholith_vertices: 
		    x and y coordinates of the batholith vertices to plot on synthetic data 
		    figures.
  
	- input/dike_vertices: 
		    x and y coordinates of the dike vertices to plot on synthetic data 
		    figures.

	- input/sill_vertices: 
		    x and y coordinates of the sill vertices to plot on synthetic data 
		    figures.


3 - Prerequisites
----------------------
The Python program "filtering.py" requires the Python packages "numpy" and "sklearn.linear_model", and 
the script "synthetic_data.py" requires the Python package "numpy", and the script "plot_figure.py" 
requires the Python packages "numpy" and "matplotlib". 
The easier way to get Python and all libraries installed is through the Anaconda Python 
distribution (https://www.anaconda.com/distribution/). After installed Anaconda, install the libraries 
by running the following command in your terminal:

	conda install numpy matplotlib sklearn.linear_model

 
4 - Parameterization
----------------------
To construct the S-function, the user must configure the parameter 'alpha_test' that corresponds 
to the variation range of the trial regularization parameters to calculate the Euclidean norm of 
the regularized directional derivatives.

To determine the regularization parameter associated with the Euclidean norm specific-value in 
the slope of the S-function, the user must configure two parameters: 'inferior_limit' and 
'upper_limit'. These parameters represent the interval limits in which the S-function presents a 
linear variation.


5 - Running the files
----------------------
After Anaconda and the libraries are installed, you need to run 
the code 'synthetic_data.py' to generate the synthetic data figures.
There are two easy ways to run:

1. Via command line: 

	1.1 On linux based systems open the command prompt windows (terminal), and run the command:
	
	On Python 2.7: python synthetic_data.py 

	On Python 3.7: python3 synthetic_data.py 

	1.2 On windows based systems open the Iphyton console (Python 2.7) or 
	Anaconda Navigator and then Jupyter QtConsole (Python 3.7), and run the command:

	run synthetic_data.py

2. Open the PyCharm program that comes with the Ananconda installation. In the PyCharm, open
the example script "synthetic_data.py" and click in "Run" inside the PyCharm to execute it.

If everything runs ok, the figures will be saved in the folder '/figures'.


6 - License
----------------------
The following legal note is restricted solely to the content of the named files. It cannot
overrule licenses from the Python standard distribution modules, which are imported and
used therein.

The 'filtering.py', 'synthetic_data.py', and 'plot_figure.py' files are distributed under 
the following license agreement:

MIT license

Copyright (c) 2023 Janaína A. Melo.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of any contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
