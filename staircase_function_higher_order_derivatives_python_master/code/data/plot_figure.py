"""
Plot functions 

A Python program to plot the total-field anomaly, non-regularized and regularized directional higher-order derivatives, and the S-function 
of the regularized directional derivatives.

This code plot the figures 1 and 2 in the folder 'figures'.

This code is released from the Master's thesis: "Regularization parameters in aeromagnetic data processing using differential operators".

The program is under the conditions terms in the file README.txt.

authors:Janaína A. Melo (IAG-USP)
email: janaina.melo@usp.br (J.A. Melo)
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

#Locale settings
import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")
plt.rcdefaults()
# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True




def plot_figure1(x, y, tfa, reg_dxx_tfa, reg1_dxx_tfa, reg2_dxx_tfa, vertices):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))

    v1 = np.linspace(min(tfa), max(tfa), 20, endpoint=True)
    v1_ = np.linspace(min(tfa), max(tfa), 5, endpoint=True)
    v2 = np.linspace(min(reg2_dxx_tfa * (10**6)), max(reg2_dxx_tfa * (10**6)), 20, endpoint=True)
    v2_ = np.linspace(min(reg2_dxx_tfa * (10**6)), max(reg2_dxx_tfa * (10**6)), 5, endpoint=True)

    tmp1 = ax[0][0].tricontourf(y/1000, x/1000, tfa, 30, cmap='gist_ncar', levels=v1)
    tmp2 = ax[1][0].tricontourf(y/1000, x/1000, reg_dxx_tfa * (10**6), 30, cmap='gist_ncar', levels=v2)
    tmp3 = ax[0][1].tricontourf(y/1000, x/1000, reg1_dxx_tfa * (10**6), 30, cmap='gist_ncar', levels=v2)
    tmp4 = ax[1][1].tricontourf(y/1000, x/1000, reg2_dxx_tfa * (10**6), 30, cmap='gist_ncar', levels=v2)

    plt.colorbar(tmp1, ax=ax[0][0], fraction=0.030, aspect=20, spacing='uniform', format='%.f', orientation='vertical',
                      ticks=v1_).set_label('(nT)', fontsize=10, labelpad=-20, y=-0.10, rotation=0)

    plt.colorbar(tmp2, ax=ax[1][0], fraction=0.030, aspect=20, spacing='uniform', format='%.f', orientation='vertical',
                      ticks=v2_).set_label('(nT/km²)', fontsize=10, labelpad=-15, y=-0.10, rotation=0)

    plt.colorbar(tmp3, ax=ax[0][1], fraction=0.030, aspect=20, spacing='uniform', format='%.f', orientation='vertical',
                      ticks=v2_).set_label('(nT/km²)', fontsize=10, labelpad=-15, y=-0.10, rotation=0)
    
    plt.colorbar(tmp4, ax=ax[1][1], fraction=0.030, aspect=20, spacing='uniform', format='%.f', orientation='vertical',
                      ticks=v2_).set_label('(nT/km²)', fontsize=10, labelpad=-15, y=-0.10, rotation=0)


    # Draw the polygons
    for b in vertices:
        path1 = Path(b)
        path2 = Path(b)
        path3 = Path(b)
        path4 = Path(b)

        pathpatch1 = PathPatch(path1, facecolor='none', edgecolor='black', linewidth=1.5)
        pathpatch2 = PathPatch(path2, facecolor='none', edgecolor='black', linewidth=1.5)
        pathpatch3 = PathPatch(path3, facecolor='none', edgecolor='black', linewidth=1.5)
        pathpatch4 = PathPatch(path4, facecolor='none', edgecolor='black', linewidth=1.5)

        ax[0][0].add_patch(pathpatch1)
        ax[0][1].add_patch(pathpatch2)
        ax[1][0].add_patch(pathpatch3)
        ax[1][1].add_patch(pathpatch4)
    

    ax[0][0].set_xlabel('y (km)', fontsize=12)
    ax[0][1].set_xlabel('y (km)', fontsize=12)
    ax[1][0].set_xlabel('y (km)', fontsize=12)
    ax[1][1].set_xlabel('y (km)', fontsize=12)
    ax[0][0].set_ylabel('x (km)', fontsize=12)
    ax[0][1].set_ylabel('x (km)', fontsize=12)
    ax[1][0].set_ylabel('x (km)', fontsize=12)
    ax[1][1].set_ylabel('x (km)', fontsize=12)

    ax[0][0].text(1.3, 27, 'a)', fontsize=15, horizontalalignment='center', verticalalignment='center')
    ax[0][1].text(1.3, 27, 'c)', fontsize=15, horizontalalignment='center', verticalalignment='center')
    ax[1][0].text(1.3, 27, 'b)', fontsize=15, horizontalalignment='center', verticalalignment='center')
    ax[1][1].text(1.3, 27, 'd)', fontsize=15, horizontalalignment='center', verticalalignment='center')

    ax[0][0].text(12, 21, 'A', fontsize=13, horizontalalignment='center', verticalalignment='center', weight='bold')
    ax[0][0].text(12, 13, 'B', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[0][0].text(20, 6, 'C', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[0][1].text(12, 21, 'A', fontsize=13, horizontalalignment='center', verticalalignment='center', weight='bold')
    ax[0][1].text(12, 13, 'B', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[0][1].text(20, 6, 'C', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[1][0].text(12, 21, 'A', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[1][0].text(12, 13, 'B', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[1][0].text(20, 6, 'C', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[1][1].text(12, 21, 'A', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[1][1].text(12, 13, 'B', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')
    ax[1][1].text(20, 6, 'C', fontsize=13, horizontalalignment='center', verticalalignment='center',  weight='bold')

    ax[0][0].yaxis.set_ticks(np.arange(5, 30, step=5))
    ax[0][1].yaxis.set_ticks(np.arange(5, 30, step=5))
    ax[1][0].yaxis.set_ticks(np.arange(5, 30, step=5))
    ax[1][1].yaxis.set_ticks(np.arange(5, 30, step=5))
    ax[0][0].xaxis.set_ticks(np.arange(5, 30, step=5))
    ax[0][0].xaxis.set_ticks(np.arange(5, 30, step=5))
    ax[1][0].xaxis.set_ticks(np.arange(5, 30, step=5))
    ax[1][1].xaxis.set_ticks(np.arange(5, 30, step=5))

    ax[0][0].tick_params(axis='both', which='major', labelsize=11)
    ax[0][1].tick_params(axis='both', which='major', labelsize=11)
    ax[1][0].tick_params(axis='both', which='major', labelsize=11)
    ax[1][1].tick_params(axis='both', which='major', labelsize=11)

    plt.subplots_adjust(wspace=0.6, hspace=0.4)

    plt.savefig('figures/FIG1.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    return




def plot_figure2(alpha, norm_sol_dx):

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(np.log10(alpha), norm_sol_dx, '-*', color='black', linewidth=1.5)

    ax.set_ylabel('S$_{xx}(\u03B1)$', color='black', fontsize=7)
    ax.set_xlabel('log$_{10}$($\u03B1$)', fontsize=7)
    ax.xaxis.set_ticks(np.arange(min(np.log10(alpha)), max(np.log10(alpha)), step=2))
    ax.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=6)

    plt.savefig('figures/FIG2.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    return

