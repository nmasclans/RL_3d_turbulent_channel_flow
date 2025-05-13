import copy
import os
import matplotlib
import configparser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns

from sklearn.neighbors import KernelDensity
from matplotlib.ticker import LogFormatter
from PIL import Image
from scipy import stats


# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 12, frameon = False)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')
#plt.rc( 'savefig',    format = "jpg", dpi = 600)

um_target = 0.0075

# Define the Tableau 10 colors
tab_colors = [
    colors.TABLEAU_COLORS['tab:blue'],
    colors.TABLEAU_COLORS['tab:green'],
    colors.TABLEAU_COLORS['tab:orange'],
    colors.TABLEAU_COLORS['tab:red'],
    colors.TABLEAU_COLORS['tab:purple'],
    colors.TABLEAU_COLORS['tab:brown'],
    colors.TABLEAU_COLORS['tab:pink'],
    colors.TABLEAU_COLORS['tab:gray'],
    colors.TABLEAU_COLORS['tab:olive'],
    colors.TABLEAU_COLORS['tab:cyan']
]

class ChannelVisualizer():

    def __init__(self, postRlzDir, figs_format='svg'):

        self.postRlzDir = postRlzDir
        self.format = figs_format

        # --- Location of Barycentric map corners ---
        self.x1c = np.array( [ 1.0 , 0.0 ] )
        self.x2c = np.array( [ 0.0 , 0.0 ] )
        self.x3c = np.array( [ 0.5 , np.sqrt(3.0)/2.0 ] )

    #--------------------------------------------------------------------------------------------
    #   Methods:                            ODT vs. DNS
    #--------------------------------------------------------------------------------------------

    def build_u_mean_profile(self, yplus_odt, yplus_ref, u_odt_post, u_odt_rt, u_ref, reference_data_name):
        
        reference_data_name_ = reference_data_name.replace(" ","_") 
        # yplus coordinates should be [0, 1, ....] according to ODT code
        # To build semilogx plot (logarithmic yplus-coordinates) we should:
        # -> initial checks on yplus coordinates
        assert yplus_odt[0] == 0,  "[ChannelVisualizer/build_u_mean_profile] ODT 1st y+ coordinate should be == 0."
        assert yplus_ref[0] == 0, f"[ChannelVisualizer/build_u_mean_profile] {reference_data_name} 1st y+ coordinate should be == 0."
        # -> round 2nd coordinate from 1+-eps to 1, if possible
        if np.abs(yplus_odt[1]-1.0) < 1e-3:
            yplus_odt[1] = 1.0
        if np.abs(yplus_ref[1]-1.0) < 1e-3:
            yplus_ref[1] = 1.0
        
        # Take only data with yplus >= 1
        idxRef = np.where(yplus_ref>=0.85)[0]
        idxOdt = np.where(yplus_odt>=1)[0]

        ### Build plots
        # plot v1:
        filename = os.path.join(self.postRlzDir, f"u_mean_vs_{reference_data_name_}_v1")
        print(f"\nMAKING PLOT OF MEAN U PROFILE: ODT vs {reference_data_name} in {filename}" )
        fig, ax = plt.subplots()
        ax.semilogx(yplus_ref[idxRef], u_ref[idxRef],    '-',  color='tab:blue',   lw=2, label=r"$\overline{u}^{+}$, " + reference_data_name)
        ax.semilogx(yplus_odt[idxOdt], u_odt_rt[idxOdt], '-.', color='tab:orange', lw=2, label=r"$\overline{u}^{+}$, ODT")
        ax.set_xlabel(r'$y^{+}$')
        ax.set_ylabel(r'$\overline{u}^{+}$')
        ax.legend(loc='upper left')
        ax.set_xlim([0.8, None])
        ax.set_ylim([0,   None])
        #ax.set_xlim([1, 1000])
        plt.tight_layout()
        plt.savefig(filename)
        # plot v2:
        filename = os.path.join(self.postRlzDir, f"u_mean_vs_{reference_data_name_}_v2")
        print(f"\nMAKING PLOT OF MEAN U PROFILE: ODT vs {reference_data_name} in {filename}" )
        fig, ax = plt.subplots()
        ax.semilogx(yplus_ref[idxRef], u_ref[idxRef],    '<-',  color='tab:blue',   lw=2, label=r"$\overline{u}^{+}$, " + reference_data_name)
        ax.semilogx(yplus_odt[idxOdt], u_odt_rt[idxOdt], 's-.', color='tab:orange', lw=2, label=r"$\overline{u}^{+}$, ODT")
        ax.set_xlabel(r'$y^{+}$')
        ax.set_ylabel(r'$\overline{u}^{+}$')
        ax.legend(loc='upper left')
        ax.set_xlim([0.8, None])
        ax.set_ylim([0,   None])
        #ax.set_xlim([1, 1000])
        plt.tight_layout()
        plt.savefig(filename)


    def build_u_rmsf_profile(self, yplus_odt, yplus_ref, urmsf_odt, vrmsf_odt, wrmsf_odt, urmsf_ref, vrmsf_ref, wrmsf_ref, reference_data_name):

        ### non-logarithmic plot

        reference_data_name_ = reference_data_name.replace(" ","_") 
        filename = os.path.join(self.postRlzDir, f"u_rmsf_vs_{reference_data_name_}_v1")
        print(f"\nMAKING PLOT OF RMS VEL PROFILES: ODT vs {reference_data_name} in {filename}" )

        fig, ax = plt.subplots()
        ax.plot(yplus_odt,  urmsf_odt, '-',  color='tab:blue',  lw=2, label=r"$u^{+}_{\textrm{rms}}$")
        ax.plot(yplus_odt,  vrmsf_odt, '--', color='tab:red',   lw=2, label=r"$v^{+}_{\textrm{rms}}$")
        ax.plot(yplus_odt,  wrmsf_odt, ':',  color='tab:green', lw=2, label=r"$w^{+}_{\textrm{rms}}$")
        ax.plot(-yplus_ref, urmsf_ref, '-',  color='tab:blue',  lw=2, label='')
        ax.plot(-yplus_ref, vrmsf_ref, '--', color='tab:red',   lw=2, label='')
        ax.plot(-yplus_ref, wrmsf_ref, ':',  color='tab:green', lw=2, label='')

        ax.plot([0,-0.1], [0,3.1], '-', linewidth=1, color='gray')
        ax.arrow( 30, 0.1,  50, 0, head_width=0.05, head_length=10, color='gray')
        ax.arrow(-30, 0.1, -50, 0, head_width=0.05, head_length=10, color='gray')
        ax.text(  30, 0.2, "ODT",               fontsize=14, color='gray')
        ax.text( -90, 0.2, reference_data_name, fontsize=14, color='gray')

        ax.set_xlabel(r'$y^{+}$')
        ax.set_ylabel(r"$u^{+}_{i, \textrm{rms}}$")
        ax.legend(loc='upper right')
        #ax.set_xlim([-300, 300])
        ax.set_ylim([-0.1, 3.1])
        plt.tight_layout()
        plt.savefig(filename)

        ### logarithmic plot

        reference_data_name_ = reference_data_name.replace(" ","_") 
        filename = os.path.join(self.postRlzDir, f"u_rmsf_vs_{reference_data_name_}_v2")
        print(f"\nMAKING PLOT OF RMS VEL PROFILES: ODT vs {reference_data_name} in {filename}" )
        
        # Take only data with yplus >= 1
        idxRef = np.where(yplus_ref>=0.85)[0]
        idxOdt = np.where(yplus_odt>=1)[0]
        
        fig, ax = plt.subplots()
        ax.semilogx(yplus_ref[idxRef], urmsf_ref[idxRef], '>--', color='tab:blue',  label=r"$u^{+}_{\textrm{rms}}$, " + reference_data_name)
        ax.semilogx(yplus_ref[idxRef], vrmsf_ref[idxRef], '^--', color='tab:red',   label=r"$v^{+}_{\textrm{rms}}$, " + reference_data_name)
        ax.semilogx(yplus_ref[idxRef], wrmsf_ref[idxRef], 'v--', color='tab:green', label=r"$w^{+}_{\textrm{rms}}$, " + reference_data_name)
        ax.semilogx(yplus_odt[idxOdt], urmsf_odt[idxOdt], 's:',  color='tab:blue',  label=r"$u^{+}_{\textrm{rms}}$, ODT")
        ax.semilogx(yplus_odt[idxOdt], vrmsf_odt[idxOdt], 'D:',  color='tab:red',   label=r"$v^{+}_{\textrm{rms}}$, ODT")
        ax.semilogx(yplus_odt[idxOdt], wrmsf_odt[idxOdt], 'o:',  ms=4, color='tab:green', label=r"$w^{+}_{\textrm{rms}}$, ODT")
        ax.set_xlim([0.8, None])
        ax.set_xlabel(r'$y^{+}$')
        ax.set_ylabel(r"$u^{+}_{i, \textrm{rms}}$")
        ax.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename)


    def build_runtime_vs_post_statistics(self, yplus_odt, um_odt_post, vm_odt_post, wm_odt_post, urmsf_odt_post, vrmsf_odt_post, wrmsf_odt_post, um_odt_rt, vm_odt_rt, wm_odt_rt, urmsf_odt_rt, vrmsf_odt_rt, wrmsf_odt_rt):

        filename = os.path.join(self.postRlzDir, "runtime_vs_postprocessed_velocity_stats")
        print(f"\nMAKING PLOT OF AVERAGED AND RMSF VEL PROFILES calculated at RUNTIME vs. POST-PROCESSED (ODT) in {filename}")

        fig, ax = plt.subplots(3, figsize=(8,10))

        # post-processed calculations as lines, runtime calculations as markers
        ms = 5; s = 10
        ax[0].plot(yplus_odt,      um_odt_post,    'k-',  label=r'$\overline{u}^{+}$ (post)')
        ax[0].plot(yplus_odt[::s], um_odt_rt[::s], 'ko',  label=r'$\overline{u}^{+}$ (runtime)', markersize=ms)
        ax[0].set_xlabel(r'$y^{+}$')
        ax[0].set_ylabel(r'$\overline{u}^{+}$')

        ax[1].plot(yplus_odt,      vm_odt_post,    'b--', label=r'$\overline{v}^{+}$ (post)')
        ax[1].plot(yplus_odt,      wm_odt_post,    'r:',  label=r'$\overline{w}^{+}$ (post)')
        ax[1].plot(yplus_odt[::s], vm_odt_rt[::s], 'b^',  label=r'$\overline{v}^{+}$ (runtime)', markersize=ms*2)
        ax[1].plot(yplus_odt[::s], wm_odt_rt[::s], 'ro',  label=r'$\overline{w}^{+}$ (runtime)', markersize=ms)
        ax[1].set_xlabel(r'$y^{+}$')
        ax[1].set_ylabel(r'$\overline{v}^{+}$, $\overline{w}^{+}$')

        ax[2].plot(yplus_odt,      urmsf_odt_post,    'k-',  label=r'$u^{+}_{\textrm{rms}}$ (post)')
        ax[2].plot(yplus_odt,      vrmsf_odt_post,    'b--', label=r'$v^{+}_{\textrm{rms}}$ (post)')
        ax[2].plot(yplus_odt,      wrmsf_odt_post,    'r:',  label=r'$w^{+}_{\textrm{rms}}$ (post)')
        ax[2].plot(yplus_odt[::s], urmsf_odt_rt[::s], 'kv',  label=r'$u^{+}_{\textrm{rms}}$ (runtime)', markersize=ms)
        ax[2].plot(yplus_odt[::s], vrmsf_odt_rt[::s], 'b^',  label=r'$v^{+}_{\textrm{rms}}$ (runtime)', markersize=ms*2)
        ax[2].plot(yplus_odt[::s], wrmsf_odt_rt[::s], 'ro',  label=r'$w^{+}_{\textrm{rms}}$ (runtime)', markersize=ms)
        ax[2].set_ylabel(r'$u^{+}_{\textrm{rms}}$, $v^{+}_{\textrm{rms}}$, $w^{+}_{\textrm{rms}}$')

        for axis in range(3):
            ax[axis].legend(loc="upper right", fontsize="small")

        plt.tight_layout()
        plt.savefig(filename)


    def build_reynolds_stress_diagonal_profile(self, y_odt, y_ref, Rxx_odt, Ryy_odt, Rzz_odt, Rxx_ref, Ryy_ref, Rzz_ref, reference_data_name):
        
        reference_data_name_ = reference_data_name.replace(" ","_") 
        filename = os.path.join(self.postRlzDir, f"reynolds_stress_diagonal_vs_{reference_data_name_}")
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES (DIAGONAL): ODT vs {reference_data_name} in {filename}" )

        fig, ax = plt.subplots()

        ax.plot(y_odt,  Rxx_odt, 'k-',  label=r"$\overline{u^{+}_{\textrm{rms}}\,u^{+}_{\textrm{rms}}}$")
        ax.plot(y_odt,  Ryy_odt, 'b--', label=r"$\overline{v^{+}_{\textrm{rms}}\,v^{+}_{\textrm{rms}}}$")
        ax.plot(y_odt,  Rzz_odt, 'r:',  label=r"$\overline{w^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$")

        ax.plot(-y_ref, Rxx_ref, 'k-',  label='')
        ax.plot(-y_ref, Ryy_ref, 'b--', label='')
        ax.plot(-y_ref, Rzz_ref, 'r:',  label='')

        ax.plot([0,0], [-1,8], '-', linewidth=1, color='gray')
        ax.arrow( 30, -0.5,  50, 0, head_width=0.05, head_length=10, color='gray')
        ax.arrow(-30, -0.5, -50, 0, head_width=0.05, head_length=10, color='gray')
        ax.text(  30, -0.4, "ODT",               fontsize=14, color='gray')
        ax.text( -80, -0.4, reference_data_name, fontsize=14, color='gray')

        ax.set_xlabel(r'$y^{+}$')
        ax.set_ylabel(r"$\overline{u^{+}_{i, \textrm{rms}}\,u^{+}_{i, \textrm{rms}}}$")
        ax.legend(loc='upper right', frameon=False)
        #ax.set_xlim([-300, 300])
        #ax.set_ylim([-1, 8])

        plt.tight_layout()
        plt.savefig(filename)


    def build_reynolds_stress_not_diagonal_profile(self, y_odt, y_ref, Rxy_odt, Rxz_odt, Ryz_odt, Rxy_ref, Rxz_ref, Ryz_ref, reference_data_name):

        reference_data_name_ = reference_data_name.replace(" ","_") 
        filename = os.path.join(self.postRlzDir, f"reynolds_stress_not_diagonal_vs_{reference_data_name_}")
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES (NOT-DIAGONAL): ODT vs DNS in {filename}" )

        fig, ax = plt.subplots()

        ax.plot(y_odt,  Rxy_odt, 'k-',  label=r"$\overline{u^{+}_{\textrm{rms}}\,v^{+}_{\textrm{rms}}}$")
        ax.plot(y_odt,  Rxz_odt, 'b--', label=r"$\overline{u^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$")
        ax.plot(y_odt,  Ryz_odt, 'r:',  label=r"$\overline{v^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$")

        ax.plot(-y_ref, Rxy_ref, 'k-',  label='')
        ax.plot(-y_ref, Rxz_ref, 'b--', label='')
        ax.plot(-y_ref, Ryz_ref, 'r:',  label='')

        ax.plot([0,0], [-1,3], '-', linewidth=1, color='gray')
        ax.arrow( 30, 0.2,  50, 0, head_width=0.05, head_length=10, color='gray')
        ax.arrow(-30, 0.2, -50, 0, head_width=0.05, head_length=10, color='gray')
        ax.text(  30, 0.3, "ODT",               fontsize=14, color='gray')
        ax.text( -80, 0.3, reference_data_name, fontsize=14, color='gray')

        ax.set_xlabel(r'$y^{+}$')
        ax.set_ylabel(r"$<u_i'u_j'>/u_\tau^2$")
        ax.legend(loc='upper right')
        #ax.set_xlim([-300, 300])
        #ax.set_ylim([-1, 3])

        plt.tight_layout()
        plt.savefig(filename)

    
    def build_runtime_vs_post_reynolds_stress(self, yplus_odt, ufufm_odt_post, vfvfm_odt_post, wfwfm_odt_post, ufvfm_odt_post, ufwfm_odt_post, vfwfm_odt_post, ufufm_odt_rt, vfvfm_odt_rt, wfwfm_odt_rt, ufvfm_odt_rt, ufwfm_odt_rt, vfwfm_odt_rt):

        filename = os.path.join(self.postRlzDir, "runtime_vs_postprocessed_reynolds_stress")
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES calculated at RUNTIME vs. POST-PROCESSED (ODT) in {filename}" )

        fig, ax = plt.subplots(3,2,figsize=(10,10))

        ms = 3; s = 10
        ax[0,0].plot(yplus_odt,      ufufm_odt_post,    'k-',  label="post")
        ax[0,0].plot(yplus_odt[::s], ufufm_odt_rt[::s], 'b--', label="runtime")
        ax[0,0].set_ylabel(r"$\overline{u^{+}_{\textrm{rms}}\,u^{+}_{\textrm{rms}}}$")

        ax[1,0].plot(yplus_odt,      vfvfm_odt_post,    'k-', label="post")
        ax[1,0].plot(yplus_odt[::s], vfvfm_odt_rt[::s], 'b--', label="runtime")
        ax[1,0].set_ylabel(r"$\overline{v^{+}_{\textrm{rms}}\,v^{+}_{\textrm{rms}}}$")

        ax[2,0].plot(yplus_odt,      wfwfm_odt_post,    'k-', label="post")
        ax[2,0].plot(yplus_odt[::s], wfwfm_odt_rt[::s], 'b--', label="runtime")
        ax[2,0].set_ylabel(r"$\overline{w^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$")

        ax[0,1].plot(yplus_odt,      ufvfm_odt_post,    'k-', label="post")
        ax[0,1].plot(yplus_odt[::s], ufvfm_odt_rt[::s], 'b--', label="runtime")
        ax[0,1].set_ylabel(r"$\overline{u^{+}_{\textrm{rms}}\,v^{+}_{\textrm{rms}}}$")

        ax[1,1].plot(yplus_odt,      ufwfm_odt_post,    'k-', label="post")
        ax[1,1].plot(yplus_odt[::s], ufwfm_odt_rt[::s], 'b--', label="runtime")
        ax[1,1].set_ylabel(r"$\overline{u^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$")

        ax[2,1].plot(yplus_odt,      vfwfm_odt_post,    'k-', label="post")
        ax[2,1].plot(yplus_odt[::s], vfwfm_odt_rt[::s], 'b--', label="runtime")
        ax[2,1].set_ylabel(r"$\overline{v^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$")

        for axis1 in range(3):
            for axis2 in range(2):
                ax[axis1,axis2].set_xlabel(r"$y^{+}$")
                ax[axis1,axis2].legend(loc="upper right", fontsize="small")

        plt.tight_layout()
        plt.savefig(filename)


    #--------------------------------------------------------------------------------------------
    #   Methods:       ODT profiles convergence for increasing averaging time
    #--------------------------------------------------------------------------------------------

    def build_u_mean_profile_odt_convergence(self, yplus_odt, yplus_ref, um_odt, um_ref, averaging_times, reference_data_name):
        """
        Builds a plot of the u_mean profile of ODT data at several averaging times  
        Mean stream-wise direction (u_mean) is already normalized by u_tau.

        Parameters:
            # TODO: add other input params   
            averaging_times (np.array): averaging times at which u_mean is obtained
                                        Shape: (num_averaging_times), column vector
        """
        filename = os.path.join(self.postRlzDir, "u_mean_convergence")
        print(f"\nMAKING PLOT OF MEAN U PROFILE CONVERGENCE of ODT vs {reference_data_name} in {filename}" )
        
        assert um_odt.shape[1]   == len(averaging_times)

        # --- yplus coordinates at the wall ---
        # yplus coordinates should be [0, 1, ....] according to ODT code
        # To build semilogx plot (logarithmic yplus-coordinates) we should:
        # -> initial checks on yplus coordinates
        assert yplus_odt[0] == 0,  "[ChannelVisualizer/build_u_mean_profile] ODT 1st y+ coordinate should be == 0."
        assert yplus_ref[0] == 0, f"[ChannelVisualizer/build_u_mean_profile] {reference_data_name} 1st y+ coordinate should be == 0."
        # -> round 2nd coordinate from 1+-eps to 1, if possible
        if np.abs(yplus_odt[1]-1.0) < 1e-3:
            yplus_odt[1] = 1.0
        if np.abs(yplus_ref[1]-1.0) < 1e-3:
            yplus_ref[1] = 1.0
        # -> remove 1st coord point (log(0)=-inf)
        yplus_odt = yplus_odt[1:]
        yplus_ref = yplus_ref[1:]
        um_odt    = um_odt[1:]
        um_ref    = um_ref[1:]

        # --- build plot odt vs. reference ---
        fig, ax = plt.subplots(figsize=(8,6))
        ax.semilogx(yplus_odt, um_odt, label = [r"$T_{{avg}}={}$".format(t) for t in averaging_times])
        ax.semilogx(yplus_ref, um_ref, 'k--', label=reference_data_name)
        ax.set_xlabel(r'$y^{+}$')
        ax.set_ylabel(r'$\overline{u}^{+}$')
        if len(averaging_times) <= 10:
            ax.legend(loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.35), fontsize=12)
            fig.subplots_adjust(top=0.75, bottom=0.15)  # Leave space for the legend above the first subplot
        else:
            plt.tight_layout()
        plt.savefig(filename)

        ########################################################################

        filename = os.path.join(self.postRlzDir, "u_mean_error_convergence")
        print(f"\nMAKING PLOT OF MEAN U NRMSE CONVERGENCE of ODT vs {reference_data_name} in {filename}" )
        nt = len(averaging_times)
        NRMSE_um = np.zeros(nt)
        for it in range(nt):
            NRMSE_um[it] = np.linalg.norm(um_odt[:,it]-um_ref, 2) / np.linalg.norm(um_ref, 2)
            
        # --- build plot odt vs. reference ---
        fig, ax = plt.subplots(figsize=(8,6))
        if NRMSE_um[-1] < 1e-16: # reference error, last iteration will have err = 0 -> log(0) = -inf
            ax.semilogy(averaging_times[:-1], NRMSE_um[:-1])
        else:
            ax.semilogy(averaging_times, NRMSE_um)
        ax.set_xlabel(r'$t^{+}$')
        ax.set_ylabel(r'$\textrm{NRMSE}(\overline{u}^{+})$')
        plt.tight_layout()
        plt.savefig(filename)


    def build_u_rmsf_profile_odt_convergence(self, yplus_odt, yplus_ref, urmsf_odt, vrmsf_odt, wrmsf_odt, urmsf_ref, vrmsf_ref, wrmsf_ref, averaging_times, reference_data_name):
        
        # --- odt vs. reference ---

        filename = os.path.join(self.postRlzDir, "u_rmsf_convergence")
        print(f"\nMAKING PLOT OF RMS VEL PROFILES: ODT vs {reference_data_name} in {filename}" )
        fig, ax = plt.subplots(3, figsize=(9,9))
        ax[0].plot(yplus_odt,  urmsf_odt)
        ax[1].plot(yplus_odt,  vrmsf_odt)
        ax[2].plot(yplus_odt,  wrmsf_odt)
        ax[0].plot(yplus_ref, urmsf_ref, 'k--')
        ax[1].plot(yplus_ref, vrmsf_ref, 'k--')
        ax[2].plot(yplus_ref, wrmsf_ref, 'k--')
        # Axis: labels and limits
        ylabel_str = [r'$u^{+}_{\textrm{rms}}$', r'$v^{+}_{\textrm{rms}}$', r'$w^{+}_{\textrm{rms}}$']
        for axis in range(3):
            ax[axis].set_xlabel(r'$y^{+}$')
            ax[axis].set_ylabel(ylabel_str[axis])
            ax[axis].set_xlim([0, np.max(np.concatenate([yplus_odt, yplus_ref]))])
            ax[axis].set_ylim([0, int(np.max([urmsf_ref, vrmsf_ref, wrmsf_ref])+1)])
        # Legend
        # Specify the legend of only for first subplot, idem for other
        if len(averaging_times) <= 10:
            labels_averaging_times = [rf"$t^+ = {t}$" for t in averaging_times]
            labels_str = labels_averaging_times + [reference_data_name,]
            ax[0].legend(labels_str, loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.6), fontsize=12)
            fig.subplots_adjust(top=0.85)  # Leave space for the legend above the first subplot
        else:
            plt.tight_layout()
        plt.savefig(filename)

    
    def build_reynolds_stress_diagonal_profile_odt_convergence(self, y_odt, y_ref, Rxx_odt, Ryy_odt, Rzz_odt, Rxx_ref, Ryy_ref, Rzz_ref, averaging_times, reference_data_name):

        filename = os.path.join(self.postRlzDir, "reynolds_stress_diagonal_odt_convergence")
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES (DIAGONAL): ODT vs {reference_data_name} in {filename}" )

        fig, ax = plt.subplots(3, figsize=(9,9))

        ax[0].plot(y_odt, Rxx_odt)
        ax[1].plot(y_odt, Ryy_odt)
        ax[2].plot(y_odt, Rzz_odt)

        ax[0].plot(y_ref, Rxx_ref, 'k--')
        ax[1].plot(y_ref, Ryy_ref, 'k--')
        ax[2].plot(y_ref, Rzz_ref, 'k--')

        # Axis: labels and limits
        ylabel_str = [r"$\overline{u^{+}_{\textrm{rms}}\,u^{+}_{\textrm{rms}}}$", r"$\overline{v^{+}_{\textrm{rms}}\,v^{+}_{\textrm{rms}}}$", r"$\overline{w^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$"]
        for axis in range(3):
            ax[axis].set_xlabel(r'$y^{+}$')
            ax[axis].set_ylabel(ylabel_str[axis])
            ax[axis].set_xlim([0, np.max(np.concatenate([y_odt,y_ref]))])
            ax[axis].set_ylim([int(np.min([Rxx_ref, Ryy_ref, Rzz_ref]))-1, int(np.max([Rxx_ref, Ryy_ref, Rzz_ref]))+1])

        # Legend
        # Specify the legend of only for first subplot, idem for other
        labels_averaging_times = [rf"$t^+ = {t}$" for t in averaging_times]
        labels_str = labels_averaging_times + [reference_data_name,]
        if Rxx_odt.shape[1] <= 15:
            ax[0].legend(labels_str, loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.6), fontsize=12)
        fig.subplots_adjust(top=0.85)  # Leave space for the legend above the first subplot

        plt.savefig(filename)


    def build_reynolds_stress_not_diagonal_profile_odt_convergence(self, y_odt, y_ref, Rxy_odt, Rxz_odt, Ryz_odt, Rxy_ref, Rxz_ref, Ryz_ref, averaging_times, reference_data_name):

        filename = os.path.join(self.postRlzDir, "reynolds_stress_not_diagonal_odt_convergence")
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES (NOT-DIAGONAL): ODT vs {reference_data_name} in {filename}" )

        fig, ax = plt.subplots(3, figsize=(9,9))

        ax[0].plot(y_odt, Rxy_odt)
        ax[1].plot(y_odt, Rxz_odt)
        ax[2].plot(y_odt, Ryz_odt)

        ax[0].plot(y_ref, Rxy_ref, 'k--')
        ax[1].plot(y_ref, Rxz_ref, 'k--')
        ax[2].plot(y_ref, Ryz_ref, 'k--')

        # Axis: labels and limits
        ylabel_str = [r"$\overline{u^{+}_{\textrm{rms}}\,v^{+}_{\textrm{rms}}}$", r"$\overline{u^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$", r"$\overline{v^{+}_{\textrm{rms}}\,w^{+}_{\textrm{rms}}}$"]
        for axis in range(3):
            ax[axis].set_xlabel(r'$y^{+}$')
            ax[axis].set_ylabel(ylabel_str[axis])
            ax[axis].set_xlim([0, np.max(np.concatenate([y_odt, y_ref]))])
            ax[axis].set_ylim([int(np.min([Rxy_ref, Rxz_ref, Ryz_ref]))-1, int(np.max([Rxy_ref, Rxz_ref, Ryz_ref]))+1])

        # Legend
        # Specify the legend of only for first subplot, idem for other
        labels_averaging_times = [rf"$t^+ = {t}$" for t in averaging_times]
        labels_str = labels_averaging_times + [reference_data_name,]
        ax[0].legend(labels_str, loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.6), fontsize=12)
        fig.subplots_adjust(top=0.85)  # Leave space for the legend above the first subplot

        plt.savefig(filename)


    def build_stress_decomposition(self, ydelta_odt, ydelta_ref, \
                                   tau_viscous_odt, tau_reynolds_odt, tau_total_odt, \
                                   tau_viscous_ref, tau_reynolds_ref, tau_total_ref,
                                   reference_data_name):
        
        reference_data_name_ = reference_data_name.replace(" ","_") 
        filename = os.path.join(self.postRlzDir, f"stress_decomposition_vs_{reference_data_name_}")
        print(f"\nMAKING PLOT OF STRESS DECOMPOSITION ODT vs {reference_data_name} in {filename}")

        fig, ax = plt.subplots(2,figsize=(9,9))
        ax[0].set_title("ODT")
        ax[0].plot(ydelta_odt[:-1], tau_viscous_odt[:-1],  'k-',  label=r"$\tau_{viscous}=\rho\nu\,d<U>/dy$"  )
        ax[0].plot(ydelta_odt[:-1], tau_reynolds_odt[:-1], 'b--', label=r"$\tau_{reynolds,uv}=-\rho<u'v'>$"  )
        ax[0].plot(ydelta_odt[:-1], tau_total_odt[:-1],    'r:',  label=r"$\tau_{total}$"  )
        ax[1].set_title(reference_data_name)
        ax[1].plot(ydelta_ref[:-1], tau_viscous_ref[:-1],  'k-',  label=r"$\tau_{viscous}=\rho\nu\,d<U>/dy$"  )
        ax[1].plot(ydelta_ref[:-1], tau_reynolds_ref[:-1], 'b--', label=r"$\tau_{reynolds,uv}=-\rho<u'v'>$"  )
        ax[1].plot(ydelta_ref[:-1], tau_total_ref[:-1],    'r:',  label=r"$\tau_{total}$"  )
        for i in range(2):
            ax[i].set_xlabel(r"$y/\delta$")
            ax[i].set_ylabel(r"$\tau(y)$")
            ax[i].legend(loc='upper right', ncol = 1)
        fig.tight_layout()
        plt.savefig(filename)


    def build_TKE_budgets(self, yplus_odt, yplus_ref, vt_u_plus_odt, d_u_plus_odt, vt_u_plus_ref, p_u_plus_ref, reference_data_name):

        reference_data_name_ = reference_data_name.replace(" ","_") 
        filename = os.path.join(self.postRlzDir, f"TKE_budgets_vs_{reference_data_name_}")
        print(f"\nMAKING PLOT OF TKE BUDGETS ODT vs {reference_data_name} in {filename}")
       
        fig, ax = plt.subplots()

        # ODT
        ax.plot(yplus_odt[1:-1], vt_u_plus_odt[1:-1],  '-', color = '#ff7575', label=r'$vt_{u}^{+}$')
        ax.plot(yplus_odt[1:-1], d_u_plus_odt[1:-1],   '-', color = '#22c7c7', label=r'$-d_{u}^{+}$')
        # DNS
        ax.plot(-yplus_ref[1:-1], vt_u_plus_ref[1:-1], '-', color = '#ff7575', label='')
        ax.plot(-yplus_ref[1:-1], p_u_plus_ref[1:-1],  '-', color = '#0505ff', label='')

        arrOffset  = -500 # arrows offset
        textOffset = 100
        ax.plot([0,0], [0,3], '-', linewidth=1, color='gray')
        ax.arrow( 30, arrOffset,  20, 0, head_width=50, head_length=5, color='gray')
        ax.arrow(-30, arrOffset, -20, 0, head_width=50, head_length=5, color='gray')
        ax.text(  30, arrOffset + textOffset, "ODT",               fontsize=14, color='gray')
        ax.text( -45, arrOffset + textOffset, reference_data_name, fontsize=14, color='gray')
        ax.set_xlabel(r'$y^{+}$')
        ax.set_ylabel("TKE budgets")
        ax.legend(loc='upper right')
        ax.set_xlim([-100, 100])

        plt.tight_layout()
        plt.savefig(filename)

    
    def build_um_profile_symmetric_vs_nonsymmetric(self, CI, yudelta, um_nonsym, um_sym):

        filename = os.path.join(self.postRlzDir, "u_mean_symmetric_vs_nonsymmetric")
        print(f"\nMAKING PLOT OF UM+ ORIGINAL NON-SYMMETRIC PROFILE vs SYMMETRIC PROFILE in {filename}")

        fig, ax = plt.subplots()
        ax.plot(yudelta, um_nonsym,  'k-',  label=r"$\overline{u}^{+}$ non-sym (original)")
        ax.plot(yudelta, um_sym,     'b--', label=r"$\overline{u}^{+}$ symmetric")
        ax.set_xlabel(r'$y/\delta$')
        ax.set_ylabel(r"$\overline{u}^{+}$")
        ax.set_title(f"um+ original (non-sym) vs. symmetric \nCI = {CI:.3f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename)


    def build_um_profile_symmetric_vs_nonsymmetric_odt_convergence(self, CI, yudelta, um_nonsym, um_sym, averaging_times):

        filename = os.path.join(self.postRlzDir, "u_mean_symmetric_vs_nonsymmetric_odt_convergence")
        print(f"\nMAKING PLOT OF UM+ ORIGINAL NON-SYMMETRIC PROFILE vs SYMMETRIC PROFILE for ODT CONVERGENCE in {filename}")

        num_profiles = um_nonsym.shape[1]

        fig, ax = plt.subplots()
        for p in range(num_profiles):
            ax.plot(yudelta, um_nonsym[:,p], '--', label=f"um+ non-sym:  t = {averaging_times[p]:.0f}, CI = {CI[p]:.1f}")
        ax.plot(yudelta, um_sym[:,-1], '-k',  label=f"um+ symmetric: t = {averaging_times[-1]:.0f}")

        ax.set_xlabel(r'$y/\delta$')
        ax.set_ylabel(r"$\overline{u}^{+}$")
        if num_profiles < 10:
            ax.legend(loc='lower center', ncol = 2, fontsize=8)
        plt.tight_layout()
        plt.savefig(filename)


    def build_CI_evolution(self, time, CI):
        # todo: include time data in the x axis, by now it is just index position in the CI list

        filename = os.path.join(self.postRlzDir, "CI_vs_time")
        print(f"\nMAKING PLOT OF CI EVOLUTION ALONG TIME in {filename}")

        fig, ax = plt.subplots()
        ax.plot(time, CI)
        ax.set_xlabel("Time (t) [s]")
        ax.set_ylabel("Convergence Indicator (CI)")
        #ax.set_ylim([0, 3])

        plt.grid()
        plt.tight_layout()
        plt.savefig(filename)

# --------------------- u-velocity vs reference as gif evolution ------------------------

    def build_vel_avg_fig(self, yplus_RL, yplus_nonRL, yplus_ref, vel_avg_RL, vel_avg_nonRL, vel_avg_ref, avg_time_RL, avg_time_nonRL, global_step, vel_name='u', ylim=None, x_actuator_boundaries=None):
        fig, ax = plt.subplots()
        # vlines for actuator boundaries
        if x_actuator_boundaries is not None:
            if ylim is None:
                ymin = np.min([np.min(vel_avg_ref),np.min(vel_avg_RL)])
                ymax = np.min([np.max(vel_avg_ref),np.max(vel_avg_RL)])
            else:
                ymin, ymax = ylim
            for i in range(len(x_actuator_boundaries)):
                plt.vlines(x_actuator_boundaries[i], ymin, ymax, colors='gray', linestyle='--')
        # plot data
        plt.semilogx(yplus_ref,   vel_avg_ref,   '-',  color="black",     lw=2, label=r"Reference")
        plt.semilogx(yplus_nonRL, vel_avg_nonRL, '--', color="tab:blue",  lw=2, label=r"Uncontrolled")
        plt.semilogx(yplus_RL,    vel_avg_RL,    ':',  color="tab:green", lw=2, label=r"RL Framework")
        # plot parameters
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel(r"$y^{+}$")
        plt.ylabel(rf"$\overline{{{vel_name}}}^{{+}}$")
        plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$\\ RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = {global_step}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    

    def build_vel_avg_frame(self, frames, yplus_RL, yplus_nonRL, yplus_ref, vel_avg_RL, vel_avg_nonRL, vel_avg_ref, avg_time_RL, avg_time_nonRL, global_step, 
                       vel_name='u', ylim=None, x_actuator_boundaries=None):
        fig = self.build_vel_avg_fig(yplus_RL, yplus_nonRL, yplus_ref, vel_avg_RL, vel_avg_nonRL, vel_avg_ref, avg_time_RL, avg_time_nonRL, global_step, vel_name, ylim, x_actuator_boundaries)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames


    def build_vel_rmsf_fig(self, yplus_RL, yplus_nonRL, yplus_ref, vel_rmsf_RL, vel_rmsf_nonRL, vel_rmsf_ref, avg_time_RL, avg_time_nonRL, global_step, 
                           vel_name='u', ylim=None, x_actuator_boundaries=None):
        fig, ax = plt.subplots()
        # vlines for actuator boundaries
        if x_actuator_boundaries is not None:
            if ylim is None:
                ymin = np.min([np.min(vel_rmsf_ref),np.min(vel_rmsf_RL)])
                ymax = np.min([np.max(vel_rmsf_ref),np.max(vel_rmsf_RL)])
            else:
                ymin, ymax = ylim
            for i in range(len(x_actuator_boundaries)):
                plt.vlines(x_actuator_boundaries[i], ymin, ymax, colors='gray', linestyle='--')
        # plot data
        plt.semilogx(yplus_ref,   vel_rmsf_ref, '-',    color='black',     label=r"Reference")
        plt.semilogx(yplus_nonRL, vel_rmsf_nonRL, '--', color='tab:blue',  label=r"Uncontrolled")
        plt.semilogx(yplus_RL,    vel_rmsf_RL, ':',     color='tab:green', label=r"RL Framework")
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel(r"$y^{+}$")
        plt.ylabel(rf"${vel_name}^{{+}}_{{\textrm{{rmsf}}}}$")
        plt.grid(axis="y")
        plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$\\ RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = {global_step}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    

    def build_vel_rmsf_frame(self, frames, yplus_RL, yplus_nonRL, yplus_ref, vel_rmsf_RL, vel_rmsf_nonRL, vel_rmsf_ref, avg_time_RL, avg_time_nonRL, global_step, 
                             vel_name='u', ylim=None, x_actuator_boundaries=None):
        
        fig = self.build_vel_rmsf_fig(yplus_RL, yplus_nonRL, yplus_ref, vel_rmsf_RL, vel_rmsf_nonRL, vel_rmsf_ref, avg_time_RL, avg_time_nonRL, global_step, vel_name, ylim, x_actuator_boundaries)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames


    def build_um_fig_RL_nonRL_ref(self, yplus_RL, yplus_nonRL, yplus_ref, um_RL, um_nonRL, um_ref, time_RL, time_nonRL, time_ref, color_RL="tab:green", ylim=None):
        
        fig, ax = plt.subplots()
        plt.semilogx(yplus_ref,   um_ref,   '-',  color="black",      lw=2, label=rf"Reference $t^+={time_ref:.1f}$")
        plt.semilogx(yplus_nonRL, um_nonRL, '-.', color="tab:orange", lw=2, label=rf"Non-RL    $t^+={time_nonRL:.1f}$")
        plt.semilogx(yplus_RL,    um_RL,    '--', color=color_RL,     lw=2, label=rf"RL        $t^+={time_RL:.1f}$")

        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel(r"$y^{+}$")
        plt.ylabel(r"$\overline{u}^{+}$")
        plt.yticks([0.0, 5.0, 10.0, 15.0, 20.0])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    
    
    def build_um_frame_RL_nonRL_ref(self, frames, 
                                    yplus_RL, yplus_nonRL, yplus_ref, 
                                    um_RL, um_nonRL, um_ref,
                                    time_RL, time_nonRL, time_ref,
                                    color_RL="tab:green",
                                    ylim=None):    

        fig = self.build_um_fig_RL_nonRL_ref(yplus_RL, yplus_nonRL, yplus_ref, um_RL, um_nonRL, um_ref, time_RL, time_nonRL, time_ref, color_RL, ylim)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

# --------------------- anisotropy tensor barycentric map in barycentric realizable triangle ------------------------

    def build_anisotropy_tensor_barycentric_xmap_triang(self, y_delta, bar_map_x, bar_map_y, avg_time, filename):
        
        plt.figure()

        # Plot markers Barycentric map
        #cmap = cm.get_cmap( 'Greys' ) ## deprecated from matplotlib 3.7
        cmap  = matplotlib.colormaps['Greys']
        norm  = colors.Normalize(vmin = 0, vmax = 1.0)

        # Plot data into the barycentric map
        plt.scatter( bar_map_x, bar_map_y, c = y_delta, cmap = cmap, norm=norm, zorder = 3, marker = 'o', s = 85, edgecolor = 'black', linewidth = 0.8 )

        # Plot barycentric map lines
        plt.plot( [self.x1c[0], self.x2c[0]],[self.x1c[1], self.x2c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
        plt.plot( [self.x2c[0], self.x3c[0]],[self.x2c[1], self.x3c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
        plt.plot( [self.x3c[0], self.x1c[0]],[self.x3c[1], self.x1c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )

        # Configure plot
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.axis( 'off' )
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.text( 1.0047, -0.025, r'$\textbf{x}_{1_{c}}$' )
        plt.text( -0.037, -0.025, r'$\textbf{x}_{2_{c}}$' )
        plt.text( 0.4850, 0.9000, r'$\textbf{x}_{3_{c}}$' )
        cbar = plt.colorbar()
        cbar.set_label( r'$y/\delta$' )
        plt.title(rf"$t_{{\textrm{{avg}}}}^{{+}} = {avg_time:.2f}$")
        plt.tight_layout()
        
        # save figure
        filepath = os.path.join(self.postRlzDir, f"{filename}.{self.format}")
        print(f"\nMAKING PLOT OF BARYCENTRIC MAP OF ANISOTROPY TENSOR in {filepath}" )
        plt.savefig(filepath, format=self.format)


    def build_anisotropy_tensor_barycentric_xmap_triang_frame(self, frames, ydelta_RL, ydelta_nonRL, ydelta_ref, xmap1_RL,  xmap1_nonRL,  xmap1_ref,  xmap2_RL, xmap2_nonRL, xmap2_ref, \
                                                              avg_time_RL, avg_time_nonRL, global_step):
        plt.figure()

        # Plot markers Barycentric map
        #cmap = cm.get_cmap( 'Greys' ) ## deprecated from matplotlib 3.7
        cmap  = matplotlib.colormaps['Greys']
        norm  = colors.Normalize(vmin = 0, vmax = 1.0)

        # Plot data into the barycentric map
        plt.scatter( xmap1_ref,   xmap2_ref,   c = ydelta_ref,   cmap = cmap, norm = norm, zorder = 3, marker = 'o', s = 85, edgecolor = 'black', linewidth = 0.8, label="Reference" )
        plt.scatter( xmap1_nonRL, xmap2_nonRL, c = ydelta_nonRL, cmap = cmap, norm = norm, zorder = 3, marker = 's', s = 85, edgecolor = 'black', linewidth = 0.8, label="Uncontrolled" )
        plt.scatter( xmap1_RL,    xmap2_RL,    c = ydelta_RL,    cmap = cmap, norm = norm, zorder = 3, marker = '^', s = 85, edgecolor = 'black', linewidth = 0.8, label="RL Framework" )

        # Plot barycentric map lines
        plt.plot( [self.x1c[0], self.x2c[0]],[self.x1c[1], self.x2c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
        plt.plot( [self.x2c[0], self.x3c[0]],[self.x2c[1], self.x3c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
        plt.plot( [self.x3c[0], self.x1c[0]],[self.x3c[1], self.x1c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )

        # Configure plot
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.axis( 'off' )
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.text( 1.02,   -0.05,  r'$\textbf{x}_{1_{c}}$' )
        plt.text( -0.065, -0.05,  r'$\textbf{x}_{2_{c}}$' )
        plt.text( 0.45,   0.9000, r'$\textbf{x}_{3_{c}}$' )
        cbar = plt.colorbar()
        cbar.set_label( r'$y/\delta$' )
        plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$\\ RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = {global_step}")
        plt.legend(loc='upper right')
        plt.tight_layout()
        ###plt.clim( 0.0, 20.0 )

        # ------ gif frame by pillow ---------
        # Save the current figure as an image frame
        fig = plt.gcf()
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        dirname = os.path.join(self.postRlzDir, "anisotropy_tensor_barycentric_xmap_triang_frames")
        os.makedirs(dirname, exist_ok=True)
        filename = os.path.join(dirname, f"anisotropy_tensor_barycentric_xmap_triang_{avg_time_RL:.3f}_{avg_time_nonRL:.3f}_{global_step}.{self.format}")
        plt.savefig(filename, format=self.format)
        plt.close()

        return frames


# ------------------- anisotropy tensor barycentric map coordinates 'xmap_i' -------------------
    
    def build_reynolds_stress_tensor_trace_fig(self, ydelta_RL, ydelta_nonRL, ydelta_ref, Rkk_RL, Rkk_nonRL, Rkk_ref, \
                                               avg_time_RL, avg_time_nonRL, global_step):
        fig, ax = plt.subplots()
        plt.plot(ydelta_ref,   Rkk_ref,   '-',  color='black',     label=r"Reference")
        plt.plot(ydelta_nonRL, Rkk_nonRL, '--', color='tab:blue',  label=r"Uncontrolled")
        plt.plot(ydelta_RL,    Rkk_RL,    ':',  color='tab:green', label=r"RL Framework")
        plt.xlim([0.0, 1.0])
        plt.xlabel(r"$y/\delta$")
        plt.ylabel(r"Reynolds Stress Trace $R_{kk}$")
        plt.grid(axis="y")
        plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$\\ RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = {global_step}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    
    def build_reynolds_stress_tensor_trace_frame(self, frames, ydelta_RL, ydelta_nonRL, ydelta_ref, Rkk_RL, Rkk_nonRL, Rkk_ref, \
                                                 avg_time_RL, avg_time_nonRL, global_step):        
        fig = self.build_reynolds_stress_tensor_trace_fig(ydelta_RL, ydelta_nonRL, ydelta_ref, Rkk_RL, Rkk_nonRL, Rkk_ref, avg_time_RL, avg_time_nonRL, global_step)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

# ------------------- anisotropy tensor barycentric map coordinates 'xmap_i' -------------------

    def build_anisotropy_tensor_barycentric_xmap_coord_fig(self, ydelta_RL, ydelta_nonRL, ydelta_ref, xmap1_RL, xmap1_nonRL, xmap1_ref, xmap2_RL, xmap2_nonRL, xmap2_ref, \
                                                           avg_time_RL, avg_time_nonRL, global_step):
        fig, ax = plt.subplots()
        plt.plot(ydelta_ref,   xmap1_ref,   '-',  color='black',    label=r"Reference $x_1$")
        plt.plot(ydelta_ref,   xmap2_ref,   '-',  color='tab:blue', label=r"Reference $x_2$")
        plt.plot(ydelta_nonRL, xmap1_nonRL, '--', color='black',    label=r"non-RL $x_1$")
        plt.plot(ydelta_nonRL, xmap2_nonRL, '--', color='tab:blue', label=r"non-RL $x_2$")
        plt.plot(ydelta_RL,    xmap1_RL,    ':',  color='black',    label=r"RL $x_1$")
        plt.plot(ydelta_RL,    xmap2_RL,    ':',  color='tab:blue', label=r"RL $x_2$")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(r"$y/\delta$")
        plt.ylabel(r"barycentric coordinates $x_i$")
        plt.grid(axis="y")
        plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$\\ RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = {global_step}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        #fig = plt.gcf()
        return fig

    def build_anisotropy_tensor_barycentric_xmap_coord_frame(self, frames, ydelta_RL, ydelta_nonRL, ydelta_ref, xmap1_RL, xmap1_nonRL, xmap1_ref, xmap2_RL, xmap2_nonRL, xmap2_ref, \
                                                             avg_time_RL, avg_time_nonRL, global_step):
        fig = self.build_anisotropy_tensor_barycentric_xmap_coord_fig(ydelta_RL, ydelta_nonRL, ydelta_ref, xmap1_RL, xmap1_nonRL, xmap1_ref, xmap2_RL, xmap2_nonRL, xmap2_ref, avg_time_RL, avg_time_nonRL, global_step)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

# ------------------- anisotropy tensor eigenvalues 'lambda_i' -------------------

    def build_anisotropy_tensor_eigenvalues_fig(self, ydelta_RL, ydelta_nonRL, ydelta_ref, eigval_RL, eigval_nonRL, eigval_ref, \
                                                avg_time_RL, avg_time_nonRL, global_step):
        fig, ax = plt.subplots()
        plt.plot(ydelta_ref,   eigval_ref[:,0],   '-',  color='black',    label=r"Reference $\lambda_0$")
        plt.plot(ydelta_ref,   eigval_ref[:,1],   '-',  color='tab:blue', label=r"Reference $\lambda_1$")
        plt.plot(ydelta_ref,   eigval_ref[:,2],   '-',  color='tab:green',label=r"Reference $\lambda_2$")
        plt.plot(ydelta_nonRL, eigval_nonRL[:,0], '--', color='black',    label=r"non-RL $\lambda_0$")
        plt.plot(ydelta_nonRL, eigval_nonRL[:,1], '--', color='tab:blue', label=r"non-RL $\lambda_1$")
        plt.plot(ydelta_nonRL, eigval_nonRL[:,2], '--', color='tab:green',label=r"non-RL $\lambda_2$")
        plt.plot(ydelta_RL,    eigval_RL[:,0],    ':',  color='black',    label=r"RL $\lambda_0$")
        plt.plot(ydelta_RL,    eigval_RL[:,1],    ':',  color='tab:blue', label=r"RL $\lambda_1$")
        plt.plot(ydelta_RL,    eigval_RL[:,2],    ':',  color='tab:green',label=r"RL $\lambda_2$")

        plt.xlim([0.0, 1.0])
        plt.ylim([-0.5, 1.0])
        plt.yticks([-2/3, -1/3, 0, 1/3, 2/3], labels = ["-2/3", "-1/3", "0", "1/3", "2/3"])
        plt.xlabel(r"$y/\delta$")
        plt.ylabel(r"anisotropy tensor eigenvalues $\lambda_i$")
        plt.grid(axis="y")
        plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$\\ RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = {global_step}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    
    def build_anisotropy_tensor_eigenvalues_frame(self, frames, ydelta_RL, ydelta_nonRL, ydelta_ref, eigval_RL, eigval_nonRL, eigval_ref, \
                                                  avg_time_RL, avg_time_nonRL, global_step):
        fig = self.build_anisotropy_tensor_eigenvalues_fig(ydelta_RL, ydelta_nonRL, ydelta_ref, eigval_RL, eigval_nonRL, eigval_ref, avg_time_RL, avg_time_nonRL, global_step)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

# ------------------- multiple-dimensions RL actions  -------------------

    def build_actions_plot(self, avg_time, actions, global_step, ylim, ylabel_name=None):
        """
        avg_time: np.array([num_time_steps])
        actions:  np.array([num_time_steps, action_dim, rl_n_envs])
        """
        action_dict = {0: r"$\Delta Rkk$", 1:r"$\Delta \theta_z$", 2:r"$\Delta \theta_y$", 3:r"$\Delta \theta_x$", 4:r"$\Delta x_1$", 5:r"$\Delta x_2$" }
        action_dim = actions.shape[1]
        rl_n_envs  = actions.shape[2]
        figs_dict  = {}
        for i_act in range(action_dim):
            fig, ax = plt.subplots()
            for i_env in range(rl_n_envs):
                plt.plot(avg_time, actions[:,i_act,i_env], linewidth=2, label=rf"RL env {i_env}")
            plt.ylim(ylim)
            plt.xlabel("Averaging time", fontsize=16)
            if ylabel_name is None:
                plt.ylabel("Action " + action_dict[i_act], fontsize=16)
            else:
                plt.ylabel(ylabel_name, fontsize=16)
            plt.grid(axis="y")
            plt.title(f"RL step = {global_step}", fontsize=16)
            if (rl_n_envs < 15):
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            figs_dict[i_act] = fig
            plt.close()
        return figs_dict

    def build_actions_scatter(self, avg_time, actions, avg_time_lim, actions_lim, global_step):
        """
        avg_time: np.array([num_time_steps])
        actions:  np.array([num_time_steps, action_dim, rl_n_envs])
        """
        action_dict = {0: r"$\Delta Rkk$", 1:r"$\Delta \theta_z$", 2:r"$\Delta \theta_y$", 3:r"$\Delta \theta_x$", 4:r"$\Delta x_1$", 5:r"$\Delta x_2$" }
        action_dim = actions.shape[1]
        rl_n_envs  = actions.shape[2]
        figs_dict  = {}
        for i_act in range(action_dim):
            fig, ax = plt.subplots()
            for i_env in range(rl_n_envs):
                plt.scatter(avg_time, actions[:,i_act,i_env], label=rf"RL env {i_env}")
            plt.xlim(avg_time_lim)
            plt.ylim(actions_lim)
            plt.xlabel("Averaging time", fontsize=16)
            plt.ylabel("Action " + action_dict[i_act], fontsize=16)
            plt.grid(axis="y")
            plt.title(f"RL step = {global_step}", fontsize=16)
            if (rl_n_envs < 15):
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            figs_dict[i_act] = fig
            plt.close()
        return figs_dict
    
    def build_actions_pdf(self, actions, actions_lim, global_step):
        """
        avg_time: np.array([num_time_steps])
        actions:  np.array([num_time_steps, action_dim, rl_n_envs])
        """
        action_dict = {0: r"$\Delta Rkk$", 1:r"$\Delta \theta_z$", 2:r"$\Delta \theta_y$", 3:r"$\Delta \theta_x$", 4:r"$\Delta x_1$", 5:r"$\Delta x_2$" }
        action_dim = actions.shape[1]
        rl_n_envs  = actions.shape[2]
        figs_dict  = {}
        for i_act in range(action_dim):
            fig, ax = plt.subplots()
            for i_env in range(rl_n_envs):
                act = actions[:, i_act, i_env]
                mean = np.mean(act)
                std_dev = np.std(act)
                sns.kdeplot(
                    act, 
                    ax=ax, 
                    fill=True, 
                    common_norm=False,  # Each env's KDE is independently normalized
                    bw_adjust=1.0,
                    label=rf"RL env {i_env}: $\mu$={mean:.2f}, $\sigma$={std_dev:.2f}"
                )
            plt.xlim(actions_lim)
            #plt.ylim([0,1])
            plt.xlabel("Action " + action_dict[i_act], fontsize=16)
            plt.ylabel("PDF", fontsize=16)
            plt.title(f"RL step = {global_step}", fontsize=16)
            if (rl_n_envs < 15):
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            figs_dict[i_act] = fig
            plt.close()
        return figs_dict
    
    def build_actions_frames(self, frames_dict_scatter, frames_dict_pdf, avg_time, actions, avg_time_lim, actions_lim, global_step):
        """
        frames_dict_scatter, frames_dict_pdf: dictionaries with keys 0, 1, ..., action_dim-1
        avg_time: np.array([num_time_steps])
        actions:  np.array([num_time_steps, action_dim, rl_n_envs])
        """
        figs_dict_scatter = self.build_actions_scatter(avg_time, actions, avg_time_lim, actions_lim, global_step)
        figs_dict_pdf     = self.build_actions_pdf(actions, actions_lim, global_step)
        action_dim = actions.shape[1]
        for i_act in range(action_dim):
            # Scatter frame
            fig_scatter = figs_dict_scatter[i_act]
            fig_scatter.canvas.draw()
            img_scatter = Image.frombytes("RGB", fig_scatter.canvas.get_width_height(), fig_scatter.canvas.tostring_rgb())
            frames_dict_scatter[i_act].append(img_scatter)
            # Pdf frame
            fig_pdf     = figs_dict_pdf[i_act]
            fig_pdf.canvas.draw()
            img_pdf = Image.frombytes("RGB", fig_pdf.canvas.get_width_height(), fig_pdf.canvas.tostring_rgb())
            frames_dict_pdf[i_act].append(img_pdf)
        return frames_dict_scatter, frames_dict_pdf
    
    def build_action_gifs_from_frames(self, frames_dict_scatter, frames_dict_pdf, action_dim):
        for i_act in range(action_dim):
            # scatter gif
            filename = os.path.join(self.postRlzDir, f"action{i_act}_scatter_vs_global_steps.gif")
            print(f"\nMAKING GIF SCATTER action {i_act} along RL GLOBAL STEPS in {filename}" )
            frames_dict_scatter[i_act][0].save(filename, save_all=True, append_images=frames_dict_scatter[i_act][1:], duration=1000, loop=0)    
            # pdf gif
            filename = os.path.join(self.postRlzDir, f"action{i_act}_pdf_vs_global_steps.gif")
            print(f"\nMAKING GIF PDF action {i_act} along RL GLOBAL STEPS in {filename}" )
            frames_dict_pdf[i_act][0].save(filename, save_all=True, append_images=frames_dict_pdf[i_act][1:], duration=1000, loop=0)    

# ------------------------------------------------------------------------

    def build_states_plot(self, avg_time, states, avg_time_lim, states_lim, global_step):
        """
        avg_time: np.array([num_time_steps])
        states:  np.array([num_time_steps, state_dim, rl_n_envs])
        """
        state_dict = {0: r"$\Delta \overline{u}$", 1:r"$\Delta u_{\textrm{rms}}$", 2:r"$\Delta v_{\textrm{rms}}$", 3:r"$\Delta w_{\textrm{rms}}$", 4:r"$x/L_x$", 5:r"$y/L_y$", 6:r"$z/L_z$"}
        state_dim  = states.shape[1]
        rl_n_envs  = states.shape[2]
        figs_dict  = {}
        for i_state in range(state_dim):
            fig, ax = plt.subplots()
            for i_env in range(rl_n_envs):
                plt.plot(avg_time, states[:,i_state,i_env], linewidth=2, label=rf"RL env {i_env}")
            plt.xlim(avg_time_lim)
            plt.ylim(states_lim)
            plt.xlabel("Averaging time", fontsize=16)
            plt.ylabel("State " + state_dict[i_state], fontsize=16)
            plt.grid(axis="y")
            plt.title(f"RL step = {global_step}", fontsize=16)
            if (rl_n_envs < 15):
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            figs_dict[i_state] = fig
            plt.close()
        return figs_dict
    
    def build_states_pdf(self, states, states_lim, global_step):
        """
        avg_time: np.array([num_time_steps])
        states:  np.array([num_time_steps, state_dim, rl_n_envs])
        """
        state_dict = {0: r"$\Delta \overline{u}$", 1:r"$\Delta u_{\textrm{rms}}$", 2:r"$\Delta v_{\textrm{rms}}$", 3:r"$\Delta w_{\textrm{rms}}$", 4:r"$x/L_x$", 5:r"$y/L_y$", 6:r"$z/L_z$"}
        state_dim  = states.shape[1]
        rl_n_envs  = states.shape[2]
        figs_dict  = {}
        for i_state in range(state_dim):
            fig, ax = plt.subplots()
            for i_env in range(rl_n_envs):
                act = states[:, i_state, i_env]
                mean = np.mean(act)
                std_dev = np.std(act)
                sns.kdeplot(
                    act, 
                    ax=ax, 
                    fill=True, 
                    common_norm=False,  # Each env's KDE is independently normalized
                    bw_adjust=1.0,
                    label=rf"RL env {i_env}: $\mu$={mean:.2f}, $\sigma$={std_dev:.2f}"
                )
            plt.xlim(states_lim)
            #plt.ylim([0,1])
            plt.xlabel("State " + state_dict[i_state], fontsize=16)
            plt.ylabel("PDF", fontsize=16)
            plt.title(f"RL step = {global_step}", fontsize=16)
            if (rl_n_envs < 15):
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            figs_dict[i_state] = fig
            plt.close()
        return figs_dict 

    def build_states_ensemble_average(self, avg_time, states, avg_time_lim, states_lim, global_step):
        state_dict = {0: r"$\Delta \overline{u}$", 1:r"$\Delta u_{\textrm{rms}}$", 2:r"$\Delta v_{\textrm{rms}}$", 3:r"$\Delta w_{\textrm{rms}}$", 4:r"$x/L_x$", 5:r"$y/L_y$", 6:r"$z/L_z$"}
        state_dim  = states.shape[1]
        rl_n_envs  = states.shape[2]
        figs_dict  = {}
        for i_state in range(state_dim):
            fig = plt.figure()
            state = states[:, i_state, :]
            # Calculate mean, standard deviation, min, max values across all agents
            mean_state = np.mean(state, axis=1)
            std_state  = np.std(state, axis=1)
            min_state  = np.min(state, axis=1)
            max_state  = np.max(state, axis=1)
            # Build plot
            plt.plot(avg_time, mean_state,                             color = plt.cm.tab10(3), label=r'$\mathbb{E}[s]$')
            plt.plot(avg_time, mean_state + std_state, linestyle='-.', color = plt.cm.tab10(3), label=r'$\pm \mathbb{V}^{1/2}[s]$')
            plt.plot(avg_time, mean_state - std_state, linestyle='-.', color = plt.cm.tab10(3))
            plt.fill_between(avg_time, min_state, max_state, alpha=0.3,    color='gray',        label=r'$\{s\}$')
            plt.xlabel(r'$t_{avg}^+$' )
            plt.ylabel(rf"State $s=${state_dict[i_state]}")
            plt.xlim(avg_time_lim)
            plt.ylim(states_lim)
            plt.title(f"RL step = {global_step}", fontsize=16)
            plt.legend(loc='upper right', frameon=True)
            plt.grid(which='both',axis='y')
            plt.tight_layout()
            figs_dict[i_state] = fig
            plt.close()
        return figs_dict

    def build_states_frames(self, frames_dict_plot, frames_dict_pdf, frames_dict_ensavg, avg_time, states, avg_time_lim, states_lim, global_step):
        """
        frames_dict_plot, frames_dict_pdf, frames_dict_ensavg: dictionaries with keys 0, 1, ..., state_dim-1
        avg_time: np.array([num_time_steps])
        states:  np.array([num_time_steps, state_dim, rl_n_envs])
        """
        figs_dict_plot   = self.build_states_plot(avg_time, states, avg_time_lim, states_lim, global_step)
        figs_dict_pdf    = self.build_states_pdf(states, states_lim, global_step)
        figs_dict_ensavg = self.build_states_ensemble_average(avg_time, states, avg_time_lim, states_lim, global_step)
        state_dim = states.shape[1]
        for i_state in range(state_dim):
            # Plot frame
            fig_plot = figs_dict_plot[i_state]
            fig_plot.canvas.draw()
            img_plot = Image.frombytes("RGB", fig_plot.canvas.get_width_height(), fig_plot.canvas.tostring_rgb())
            frames_dict_plot[i_state].append(img_plot)
            # Pdf frame
            fig_pdf = figs_dict_pdf[i_state]
            fig_pdf.canvas.draw()
            img_pdf = Image.frombytes("RGB", fig_pdf.canvas.get_width_height(), fig_pdf.canvas.tostring_rgb())
            frames_dict_pdf[i_state].append(img_pdf)
            # Ensemble average frame
            fig_ensavg  = figs_dict_ensavg[i_state]
            fig_ensavg.canvas.draw()
            img_ensavg = Image.frombytes("RGB", fig_ensavg.canvas.get_width_height(), fig_ensavg.canvas.tostring_rgb())
            frames_dict_ensavg[i_state].append(img_ensavg)
        return frames_dict_plot, frames_dict_pdf, frames_dict_ensavg
    
    def build_state_gifs_from_frames(self, frames_dict_plot, frames_dict_pdf, frames_dict_ensavg, state_dim):
        for i_state in range(state_dim):
            # Plot gif
            filename = os.path.join(self.postRlzDir, f"state{i_state}_plot_vs_global_steps.gif")
            print(f"\nMAKING GIF SCATTER state {i_state} along RL GLOBAL STEPS in {filename}" )
            frames_dict_plot[i_state][0].save(filename, save_all=True, append_images=frames_dict_plot[i_state][1:], duration=1000, loop=0)    
            # Pdf gif
            filename = os.path.join(self.postRlzDir, f"state{i_state}_pdf_vs_global_steps.gif")
            print(f"\nMAKING GIF PDF state {i_state} along RL GLOBAL STEPS in {filename}" )
            frames_dict_pdf[i_state][0].save(filename, save_all=True, append_images=frames_dict_pdf[i_state][1:], duration=1000, loop=0)   
            # Ensemble average gif
            filename = os.path.join(self.postRlzDir, f"state{i_state}_ensavg_vs_global_steps.gif")
            print(f"\nMAKING GIF ENSEMBLE AVERAGE state {i_state} along RL GLOBAL STEPS in {filename}" )
            frames_dict_ensavg[i_state][0].save(filename, save_all=True, append_images=frames_dict_ensavg[i_state][1:], duration=1000, loop=0)   

# ------------------------------------------------------------------------

    def build_rewards_plot(self, avg_time, rewards, avg_time_lim, rewards_lim, global_step, reward_name):
        """
        avg_time: np.array([num_time_steps])
        rewards:  np.array([num_time_steps, rl_n_envs])
        """
        rl_n_envs  = rewards.shape[1]
        fig, ax = plt.subplots()
        for i_env in range(rl_n_envs):
            plt.plot(avg_time, rewards[:,i_env], linewidth=2, label=rf"RL env {i_env}")
        plt.xlim(avg_time_lim)
        plt.ylim(rewards_lim)
        plt.xlabel("Averaging time", fontsize=16)
        plt.ylabel(reward_name, fontsize=16)
        plt.grid(axis="y")
        plt.title(f"RL step = {global_step}", fontsize=16)
        if (rl_n_envs < 15):
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        fig_plot = fig
        plt.close()
        return fig_plot

    def build_rewards_pdf(self, rewards, rewards_lim, global_step, reward_name):
        """
        avg_time: np.array([num_time_steps])
        rewards:  np.array([num_time_steps, rl_n_envs])
        """
        import seaborn as sns
        rl_n_envs  = rewards.shape[1]
        fig, ax = plt.subplots()
        for i_env in range(rl_n_envs):
            rew = rewards[:, i_env]
            mean = np.mean(rew)
            std_dev = np.std(rew)
            sns.kdeplot(
                rew, 
                ax=ax, 
                fill=True, 
                common_norm=False,  # Each env's KDE is independently normalized
                bw_adjust=1.0,
                label=rf"RL env {i_env}: $\mu$={mean:2.2f}, $\sigma$={std_dev:2.2f}"
            )
        plt.xlim(rewards_lim)
        plt.ylim([0,0.5])
        plt.xlabel(reward_name, fontsize=16)
        plt.ylabel("PDF", fontsize=16)
        plt.title(f"RL step = {global_step}", fontsize=16)
        if (rl_n_envs < 15):
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        fig_pdf = fig
        plt.close()
        return fig_pdf

    def build_rewards_ensemble_average(self, avg_time, rewards, avg_time_lim, rewards_lim, global_step, reward_name):
        # Calculate mean, standard deviation, min, max values across all agents
        mean_rew = np.mean(rewards, axis=1)
        std_rew  = np.std(rewards, axis=1)
        min_rew  = np.min(rewards, axis=1)
        max_rew  = np.max(rewards, axis=1)
        # Build plot
        fig_ensavg = plt.figure()
        plt.plot(avg_time, mean_rew,                           color = plt.cm.tab10(3), label=r'$\mathbb{E}[r]$')
        plt.plot(avg_time, mean_rew + std_rew, linestyle='-.', color = plt.cm.tab10(3), label=r'$\pm \mathbb{V}^{1/2}[r]$')
        plt.plot(avg_time, mean_rew - std_rew, linestyle='-.', color = plt.cm.tab10(3))
        plt.fill_between(avg_time, min_rew, max_rew, alpha=0.3,    color='gray',        label=r'$\{r\}$')
        plt.xlabel(r'$t_{avg}^+$' )
        plt.ylabel(r"Reward $r$")
        plt.xlim(avg_time_lim)
        plt.ylim(rewards_lim)
        plt.title(f"RL step = {global_step}", fontsize=16)
        plt.legend(loc='upper right', frameon=True)
        plt.grid(which='both',axis='y')
        plt.tight_layout()
        plt.close()
        return fig_ensavg

    def build_rewards_frames(self, frames_plot, frames_pdf, frames_ensavg, avg_time, rewards, avg_time_lim, rewards_lim, global_step, reward_name = 'Local Reward'):
        """
        frames_plot, frames_pdf: lists of frames of different plots
        avg_time: np.array([num_time_steps])
        rewards:  np.array([num_time_steps, rl_n_envs])
        """
        fig_plot   = self.build_rewards_plot(avg_time, rewards, avg_time_lim, rewards_lim, global_step, reward_name)
        fig_pdf    = self.build_rewards_pdf(rewards, rewards_lim, global_step, reward_name)
        fig_ensavg = self.build_rewards_ensemble_average(avg_time, rewards, avg_time_lim, rewards_lim, global_step, reward_name)
        # Plot frame
        fig_plot.canvas.draw()
        img_plot = Image.frombytes("RGB", fig_plot.canvas.get_width_height(), fig_plot.canvas.tostring_rgb())
        frames_plot.append(img_plot)
        # Pdf frame
        fig_pdf.canvas.draw()
        img_pdf = Image.frombytes("RGB", fig_pdf.canvas.get_width_height(), fig_pdf.canvas.tostring_rgb())
        frames_pdf.append(img_pdf)
        # Ensemble average frame
        fig_ensavg.canvas.draw()
        img_ensavg = Image.frombytes("RGB", fig_ensavg.canvas.get_width_height(), fig_ensavg.canvas.tostring_rgb())
        frames_ensavg.append(img_ensavg)
        return frames_plot, frames_pdf, frames_ensavg



    def build_rewards_gifs_from_frames(self, frames_plot, frames_pdf, frames_ensavg, reward_name = "local_reward"):
        # plot gif
        filename = os.path.join(self.postRlzDir, f"{reward_name}_plot_vs_global_steps.gif")
        print(f"\nMAKING GIF PLOT {reward_name} along RL GLOBAL STEPS in {filename}" )
        frames_plot[0].save(filename, save_all=True, append_images=frames_plot[1:], duration=1000, loop=0)    
        # pdf gif
        filename = os.path.join(self.postRlzDir, f"{reward_name}_pdf_vs_global_steps.gif")
        print(f"\nMAKING GIF PDF {reward_name} along RL GLOBAL STEPS in {filename}" )
        frames_pdf[0].save(filename, save_all=True, append_images=frames_pdf[1:], duration=1000, loop=0)    
        # ensemble average gif
        filename = os.path.join(self.postRlzDir, f"{reward_name}_ensavg_vs_global_steps.gif")
        print(f"\nMAKING GIF ENSEMBLE AVERAGE {reward_name} along RL GLOBAL STEPS in {filename}" )
        frames_ensavg[0].save(filename, save_all=True, append_images=frames_ensavg[1:], duration=1000, loop=0)    

# ------------------------------------------------------------------------

    def build_avg_u_bulk_frames(self, frames_plot, avg_time, avg_u_bulk_num, avg_u_bulk_ref, global_step, xlim, ylim):
        # Build figure
        fig, ax = plt.subplots()
        plt.hlines(avg_u_bulk_ref, xlim[0], xlim[1], linestyle = '-', linewidth=2, color='black', label=r"Reference")
        plt.plot(avg_time, avg_u_bulk_num,           linestyle = '-', linewidth=2, color=plt.cm.tab10(1), label=r"RL Framework")
        plt.xlabel( r'Averaging time $t_{avg}^+$', fontsize=16)
        plt.ylabel( r'$\overline{u}^{+}_b$', fontsize=16)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(which='major',axis='y')
        plt.tick_params( axis = 'both', pad = 7.5 )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
        plt.title(f"RL step = {global_step}", fontsize=16)
        plt.tight_layout()        
        # Transform figure to image
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames_plot.append(img)
        plt.close(fig)
        return frames_plot

# ------------------------------------------------------------------------

    def build_main_gifs_from_frames(self, frames_dict):
      
        for k,v in frames_dict.items():
            frames_name = k
            frames_list = v
            filename = os.path.join(self.postRlzDir, f"{frames_name}_vs_global_steps.gif")
            print(f"\nMAKING GIF {frames_name} for RUNTIME calculations along TRAINING GLOBAL STEPS in {filename}" )
            frames_list[0].save(filename, save_all=True, append_images=frames_list[1:], duration=1000, loop=0)    

# ------------------------------------------------------------------------

    def plot_line(self, xdata, ydata, xlim, ylim, xlabel, ylabel, title):

        filename = os.path.join(self.postRlzDir, f"{title}")
        print(f"\nMAKING PLOT of {xlabel} vs. {ylabel} in {filename}" )

        fig, ax = plt.subplots()

        plt.plot(xdata, ydata, linewidth=2)
        plt.xlim(xlim); 
        plt.xlabel(xlabel)
        plt.ylim(ylim); #plt.yticks(yticks); 
        plt.ylabel(ylabel)

        plt.savefig(filename, format=self.format)
        plt.close()


    def plot_pdf(self, xdata, xlim, xlabel, nbins, title):

        filename = os.path.join(self.postRlzDir, f"{title}")
        print(f"\nMAKING PLOT of PDF of {xlabel} in {filename}" )

        # Compute a histogram of the sample
        bins = np.linspace(xlim[0], xlim[1], nbins)
        histogram, bins = np.histogram(xdata, bins=bins, density=True)

        # Compute pdf
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        pdf = stats.norm.pdf(bin_centers)

        plt.figure(figsize=(6, 4))
        plt.plot(bin_centers, pdf)
        plt.xlabel(xlabel)
        plt.xlim(xlim)
        plt.ylim([0,1])

        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()


    def plot_join_pdf(self, x_data, y_data, xlim, ylim, xlabel, ylabel, nbins, title):

        filename = os.path.join(self.postRlzDir, f"{title}")
        print(f"\nMAKING PLOT of JOIN-PDF of {xlabel} vs. {ylabel} in {filename}" )

        fig, ax = plt.subplots()

        # Histogram 2D plot
        x_bins = np.linspace( x_data.min(), x_data.max(), nbins )
        y_bins = np.linspace( y_data.min(), y_data.max(), nbins )
        h, x_edges, y_edges = np.histogram2d( x_data, y_data, bins = [ x_bins, y_bins ], normed = True )
        h = h + 1.0e-12
        h = h.T
        x_centers = ( x_edges[:-1] + x_edges[1:] )/2
        y_centers = ( y_edges[:-1] + y_edges[1:] )/2
        
        # Plot data
        #my_cmap = copy.copy( cm.get_cmap( 'Greys' ) )
        my_cmap = copy.copy( cm.get_cmap( 'pink_r' ) )
        my_cmap.set_under( 'white' )
        cs = ax.contour( x_centers, y_centers, h, colors = 'black', zorder = 2, norm = colors.LogNorm( vmin = 10.0**( int( np.log10( h.max() ) ) - 4 ), vmax = 10.0**( int( np.log10( h.max() ) ) + 1) ), levels = ( 10.0**( int( np.log10( h.max() ) ) - 4 ), 10.0**( int( np.log10( h.max() ) ) - 3 ), 10.0**( int( np.log10( h.max() ) ) - 2 ), 10.0**( int( np.log10( h.max() ) ) - 1 ), 10.0**( int( np.log10( h.max() ) ) + 0 ), 10.0**( int( np.log10( h.max() ) ) + 1 ) ),  linestyles = '--', linewidths = 1.0 )
        cs = ax.contourf(x_centers, y_centers, h, cmap = my_cmap,   zorder = 1, norm = colors.LogNorm( vmin = 10.0**( int( np.log10( h.max() ) ) - 4 ), vmax = 10.0**( int( np.log10( h.max() ) ) + 1) ), levels = ( 10.0**( int( np.log10( h.max() ) ) - 4 ), 10.0**( int( np.log10( h.max() ) ) - 3 ), 10.0**( int( np.log10( h.max() ) ) - 2 ), 10.0**( int( np.log10( h.max() ) ) - 1 ), 10.0**( int( np.log10( h.max() ) ) + 0 ), 10.0**( int( np.log10( h.max() ) ) + 1 ) ) )
        cbar = plt.colorbar(cs, ax=ax, shrink = 0.95, pad = 0.025, format = LogFormatter(10, labelOnlyBase=False))
        cs.set_clim( 10.0**( int( np.log10( h.max() ) ) - 4 ), 10.0**( int( np.log10( h.max() ) ) + 1 ) )
        
        # Configure plot
        ax.set_xlim(xlim); #ax.set_xticks(xticks); 
        ax.set_xlabel(xlabel)
        ax.tick_params( axis = 'x', direction = 'in', bottom = True, top = True, left = True, right = True )
        ax.set_ylim(ylim); #plt.yticks(yticks); 
        ax.set_ylabel(ylabel)
        ax.tick_params( axis = 'y', direction = 'in', bottom = True, top = True, left = True, right = True )
        ax.tick_params(axis = 'both', pad = 5) 	# add padding to both x and y axes, dist between axis ticks and label

        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()

# ------------------------------------ RL Convergence along ODT Realizations -------------------------------------------

    def RL_u_mean_convergence(self, yplus, rlzArr, 
                              um_RL_nonConv, urmsf_RL_nonConv, um_nonRL_nonConv, urmsf_nonRL_nonConv, um_baseline, urmsf_baseline, 
                              um_NRMSE_RL, urmsf_NRMSE_RL, um_NRMSE_nonRL, urmsf_NRMSE_nonRL,
                              time_nonConv_RL, time_nonConv_nonRL, time_baseline):
        # --------- plot um data ---------
        
        filename = os.path.join(self.postRlzDir, f"RL_u_mean_convergence")
        print(f"\nMAKING PLOT of um profile at tEndAvg for multiple realizations in {filename}")

        fig, ax = plt.subplots(1,2, figsize=(10,5))

        # > RL non-converged (at time_nonConv):
        nrlz = um_RL_nonConv.shape[1]
        for irlz in range(nrlz):
            if irlz == nrlz - 1: # last rlz, may have truncated
                ax[0].semilogx(yplus, um_RL_nonConv[:,irlz], label=rf"RL Rlz {rlzArr[irlz]} at $t^+_{{\textrm{{avg}}}}={time_nonConv_RL:.0f}$")
            else:
                ax[0].semilogx(yplus, um_RL_nonConv[:,irlz], label=rf"RL Rlz {rlzArr[irlz]} at $t^+_{{\textrm{{avg}}}}={time_nonConv_nonRL:.0f}$")
        # > non-RL non-converged (at time_nonConv):
        ax[0].semilogx(yplus, um_nonRL_nonConv, '--k', label=rf"Non-RL  at $t^+_{{\textrm{{avg}}}}={time_nonConv_nonRL:.0f}$")
        # > non-RL baseline (at time_baseline)
        ax[0].semilogx(yplus, um_baseline, '-k', label=rf"Reference at $t^+_{{\textrm{{avg}}}}={time_baseline:.0f}$")
        ax[0].set_xlabel(r'$y^{+}$')
        ax[0].set_ylabel(r'$\overline{u}^{+}$')
        if nrlz < 10:
            ax[0].legend(loc='upper left', fontsize=12)

        # Error variable vs. realization *per grid point*
        num_points = len(um_baseline)
        absErr_RL = np.zeros([num_points, nrlz])
        for irlz in range(nrlz):
            absErr_RL[:,irlz] = np.abs(um_RL_nonConv[:,irlz] - um_baseline)
        absErr_nonRL = np.abs(um_nonRL_nonConv - um_baseline)
        # > RL non-converged (at time_nonConv):
        for irlz in range(nrlz):
            ax[1].loglog(yplus, absErr_RL[:,irlz], label=rf"RL Rlz {rlzArr[irlz]} at $t^+_{{\textrm{{avg}}}}={time_nonConv_RL:.0f}$")
        # > non-RL non-converged (at time_nonConv):
        ax[1].loglog(yplus, absErr_nonRL, '--k', label=rf"Non-RL at $t^+_{{\textrm{{avg}}}}={time_nonConv_nonRL:.0f}$")
        ax[1].set_xlabel(r'$y^{+}$')
        ax[1].set_ylabel(r'$| \overline{u}^{+} - \overline{u}_{\textrm{ref}}^{+} |$')
        if nrlz < 10:
            ax[1].legend(loc='lower left', fontsize=12)

        ### # NRMSE variable vs. realization
        ### ax[2].semilogy(rlzArr, um_NRMSE_RL, '-o', label="RL Framework")
        ### ax[2].semilogy(rlzArr, um_NRMSE_nonRL * np.ones(nrlz), '--k', label=f"non-RL ({um_NRMSE_nonRL:.3e})")
        ### ax[2].set_xlabel('Rlz')
        ### ax[2].set_ylabel(r'NRMSE($\overline{u}^{+}$)')
        ### ax[2].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # --------- plot urmsf data ---------
        
        filename = os.path.join(self.postRlzDir, f"RL_u_rmsf_convergence")
        print(f"\nMAKING PLOT of urmsf profile at tEndAvg for multiple realizations in {filename}")

        fig, ax = plt.subplots(1,2, figsize=(10,5))

        # > RL non-converged (at time_nonConv):
        nrlz = um_RL_nonConv.shape[1]
        for irlz in range(nrlz):
            if irlz == nrlz - 1:  # last rlz, may be truncated
                ax[0].semilogx(yplus, urmsf_RL_nonConv[:,irlz], label=rf"RL Rlz {rlzArr[irlz]} at $t^+_{{\textrm{{avg}}}}={time_nonConv_RL:.0f}$")
            else:
                ax[0].semilogx(yplus, urmsf_RL_nonConv[:,irlz], label=rf"RL Rlz {rlzArr[irlz]} at $t^+_{{\textrm{{avg}}}}={time_nonConv_nonRL:.0f}$")
        # > non-RL non-converged (at time_nonConv):
        ax[0].semilogx(yplus, urmsf_nonRL_nonConv, '--k', label=rf"Non-RL at $t^+_{{\textrm{{avg}}}}={time_nonConv_nonRL:.0f}$")
        # > non-RL baseline (at time_baseline)
        ax[0].semilogx(yplus, urmsf_baseline, '-k', label=rf"Reference at $t^+_{{\textrm{{avg}}}}={time_baseline:.0f}$")
        ax[0].set_xlabel(r'$y^{+}$')
        ax[0].set_ylabel(r'$u^{+}_{\textrm{rms}}$')
        if nrlz <= 15:
            ax[0].legend(fontsize=12)

        # Error variable vs. realization *per grid point*
        num_points = len(urmsf_baseline)
        absErr_RL = np.zeros([num_points, nrlz])
        for irlz in range(nrlz):
            absErr_RL[:,irlz] = np.abs(urmsf_RL_nonConv[:,irlz] - urmsf_baseline)
        absErr_nonRL = np.abs(urmsf_nonRL_nonConv - urmsf_baseline)
        # > RL non-converged (at time_nonConv):
        for irlz in range(nrlz):
            if irlz == nrlz - 1: # last rlz, may be truncated
                ax[1].loglog(yplus, absErr_RL[:,irlz], label=rf"RL Rlz {rlzArr[irlz]} at $t^+_{{\textrm{{avg}}}}={time_nonConv_RL:.0f}$")
            else:
                ax[1].loglog(yplus, absErr_RL[:,irlz], label=rf"RL Rlz {rlzArr[irlz]} at $t^+_{{\textrm{{avg}}}}={time_nonConv_nonRL:.0f}$")
        # > non-RL non-converged (at time_nonConv):
        ax[1].loglog(yplus, absErr_nonRL, '--k', label=rf"Non-RL at $t^+_{{\textrm{{avg}}}}={time_nonConv_nonRL:.0f}$")
        ax[1].set_xlabel(r'$y^{+}$')
        ax[1].set_ylabel(r"Absolute Error $| u^{+}_{\textrm{rms}} - u^{+}_{\textrm{rms, ref}} |$")
        if nrlz < 10:
            ax[1].legend(loc='lower left', fontsize=12)

        ### # NRMSE variable vs. realization
        ### ax[2].semilogy(rlzArr, urmsf_NRMSE_RL, '-o', label="RL Framework")
        ### ax[2].semilogy(rlzArr, urmsf_NRMSE_nonRL * np.ones(nrlz), '--k', label=f"non-RL ({urmsf_NRMSE_nonRL:.3e})")
        ### ax[2].set_xlabel('Rlz')
        ### ax[2].set_ylabel(r"NRMSE $u^{+}_{\textrm{rms}}$")
        ### ax[2].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    def RL_variable_convergence_along_ydelta(self, filename, ylabel, rlzArr, ydelta, var_RL_nonConv, var_nonRL_nonConv, var_baseline, time_nonConv, time_baseline):
        
        filename = os.path.join(self.postRlzDir, f"{filename}.{self.format}")
        print(f"\nMAKING PLOT {filename}")

        fig, ax = plt.subplots(1, 2, figsize=(10,5))

        # variable vs. y-coordinate, for each realization
        # > RL non-converged (at time_nonConv):
        nrlz   = len(rlzArr)
        if nrlz <= 10:
            colors = tab_colors
        else:
            colors = plt.cm.viridis_r(np.linspace(0, 1, nrlz))
        for irlz in range(nrlz):
            ax[0].plot(ydelta, var_RL_nonConv[:,irlz], color=colors[irlz], label=rf"RL Rlz {rlzArr[irlz]} at $t^+_{{\textrm{{avg}}}}={time_nonConv:.0f}$")
        # > non-RL non-converged (at time_nonConv):
        ax[0].plot(ydelta, var_nonRL_nonConv, '--k', label=rf"Non-RL at $t^+_{{\textrm{{avg}}}}={time_nonConv:.0f}$")
        # > non-RL baseline (at time_baseline)
        ax[0].plot(ydelta, var_baseline, '-k', label=rf"Reference at $t^+_{{\textrm{{avg}}}}={time_baseline:.0f}$")
        ax[0].set_xlabel(r"$y/\delta$")
        ax[0].set_ylabel(ylabel)
        if nrlz < 10:
            ax[0].legend(frameon=True, fontsize=10)

        # NRMSE variable vs. realization
        NRMSE_RL = np.zeros(nrlz)
        for irlz in range(nrlz):
            NRMSE_RL[irlz] = np.linalg.norm(var_RL_nonConv[:,irlz] - var_baseline, 2) \
                             / np.linalg.norm(var_baseline, 2)
        NRMSE_nonRL = np.linalg.norm(var_nonRL_nonConv - var_baseline, 2) \
                      / np.linalg.norm(var_baseline, 2)
        ax[1].plot(rlzArr, NRMSE_RL, '-o', label="RL Framework")
        ax[1].plot(rlzArr, NRMSE_nonRL * np.ones(nrlz), '--k', label="Uncontrolled")
        ax[1].set_xlabel("Rlz")
        ax[1].set_ylabel(f"NRMSE {ylabel}")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()

    def RL_variable_convergence_along_time(self, filename, ylabel, rlzArr, timeArr, var_RL):
        
        filename = os.path.join(self.postRlzDir, f"{filename}.{self.format}")
        print(f"\nMAKING PLOT {filename}")

        fig, ax = plt.subplots(1,3,figsize=(15,5))
        nrlz = len(rlzArr)
        for irlz in range(nrlz):
            ax[0].plot(timeArr, var_RL[irlz,:], label=f"RL Rlz {rlzArr[irlz]}")
        ax[0].set_xlabel("FTT")
        ax[0].set_ylabel(ylabel)
        if nrlz < 10:
            ax[0].legend(frameon=True, fontsize=10)

        # Mean variable vs. realization
        mean_RL = np.zeros(nrlz)
        for irlz in range(nrlz):
            mean_RL[irlz] = np.mean(var_RL[irlz,:])
        ax[1].plot(rlzArr, mean_RL, '-o')
        ax[1].set_xlabel("Rlz")
        ax[1].set_ylabel(f"Mean {ylabel}")

        # Final value variable vs. realization
        ax[2].plot(rlzArr, var_RL[:,-1], '-o')
        ax[2].set_xlabel("Rlz")
        ax[2].set_ylabel(f"Terminal value {ylabel}")
        
        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()

    
    def RL_variable_convergence_along_time_and_pdf(self, filename, ylabel, rlzArr, timeArr, var_RL):
        filename = os.path.join(self.postRlzDir, f"{filename}.{self.format}")
        print(f"\nMAKING PLOT {filename}")
        
        fig, ax = plt.subplots(1,3, figsize=(15,5))

        # var_RL vs timeArr
        nrlz = len(rlzArr)
        for irlz in range(nrlz):
            ax[0].plot(timeArr, var_RL[irlz,:], label=f"RL Rlz {rlzArr[irlz]}")
        ax[0].set_xlabel("FTT")
        ax[0].set_ylabel(ylabel)
        if nrlz < 10:
            ax[0].legend(frameon=True, fontsize=10)

        # var_RL kde or pdf for each rlz
        for irlz in range(nrlz):
            sns.kdeplot(y=var_RL[irlz,:], label=f"RL Rlz {rlzArr[irlz]}", ax=ax[1], cut=0, warn_singular=False)
        ax[1].set_xlabel("PDF")
        ax[1].set_ylabel(ylabel)

        # var_RL mean for each rlz
        var_mean = np.mean(var_RL, axis=1)
        ax[2].plot(rlzArr, var_mean, '-o')
        ax[2].set_xlabel('Realization Id.')
        ax[2].set_ylabel(f'{ylabel} mean')

        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()


    def RL_Rij_convergence(self, ydelta, rlzArr, 
                           Rkk_RL_nonConv,    lambda1_RL_nonConv,    lambda2_RL_nonConv,    lambda3_RL_nonConv,    xmap1_RL_nonConv,    xmap2_RL_nonConv,
                           Rkk_nonRL_nonConv, lambda1_nonRL_nonConv, lambda2_nonRL_nonConv, lambda3_nonRL_nonConv, xmap1_nonRL_nonConv, xmap2_nonRL_nonConv,
                           Rkk_baseline,      lambda1_baseline,      lambda2_baseline,      lambda3_baseline,      xmap1_baseline,      xmap2_baseline,
                           time_nonConv, time_baseline):
        
        # --------- plot trace Rkk ---------
        self.RL_variable_convergence_along_ydelta("RL_Rkk_convergence",     r"$R_{kk}$",      rlzArr, ydelta, Rkk_RL_nonConv, Rkk_nonRL_nonConv, Rkk_baseline, time_nonConv, time_baseline)
        # --------- plot eigenvalues lambda_i ---------
        self.RL_variable_convergence_along_ydelta("RL_lambda1_convergence", r"$\lambda_{1}$", rlzArr, ydelta, lambda1_RL_nonConv, lambda1_nonRL_nonConv, lambda1_baseline, time_nonConv, time_baseline)
        self.RL_variable_convergence_along_ydelta("RL_lambda2_convergence", r"$\lambda_{2}$", rlzArr, ydelta, lambda2_RL_nonConv, lambda2_nonRL_nonConv, lambda2_baseline, time_nonConv, time_baseline)
        self.RL_variable_convergence_along_ydelta("RL_lambda3_convergence", r"$\lambda_{3}$", rlzArr, ydelta, lambda3_RL_nonConv, lambda3_nonRL_nonConv, lambda3_baseline, time_nonConv, time_baseline)
        # --------- plot barycentric map coordinates xmap_i ---------
        self.RL_variable_convergence_along_ydelta("RL_xmap1_convergence",   r"$x_{1}$",    rlzArr, ydelta, xmap1_RL_nonConv, xmap1_nonRL_nonConv, xmap1_baseline, time_nonConv, time_baseline)
        self.RL_variable_convergence_along_ydelta("RL_xmap2_convergence",   r"$x_{2}$",    rlzArr, ydelta, xmap2_RL_nonConv, xmap2_nonRL_nonConv, xmap2_baseline, time_nonConv, time_baseline)


    def RL_err_convergence(self, rlzArr, err_RL, err_nonRL, time_nonConv, error_name):
        """
        Params:
        - err_RL:    np.array, shape [n_realizations]
        - err_nonRL: np.array, shape [] (scalar)
        """
        filename = os.path.join(self.postRlzDir, f"RL_error_{error_name}_convergence.{self.format}")
        print(f"\nMAKING PLOT of error {error_name} profile at tEndAvg for multiple realizations in {filename}")

        nrlz = len(rlzArr)
        plt.figure()
        # > RL non-converged (at time_nonConv):
        plt.semilogy(rlzArr, err_RL, '-o', label=f"RL: t={time_nonConv}")
        # > non-RL non-converged (at time_nonConv):
        plt.semilogy(rlzArr, np.ones(nrlz) * err_nonRL, '--k', label=f"Non-RL:  t={time_nonConv}")
        
        # configure plot
        plt.xlabel('Realization Id.')
        plt.ylabel(f'Error {error_name}')
        if nrlz < 20:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()


    def RL_err_convergence_along_time(self, rlzArr, err_RL, err_ref, averaging_times_RL, averaging_times_ref, info, tEndAvgRef=200):
        
        filename = os.path.join(self.postRlzDir, f"RL_error_{info['title']}_temporal_convergence.{self.format}")
        print(f"\nMAKING PLOT of error {info['title']} profile for chosen times for multiple realizations in {filename}")

        plt.figure()
        # nonRL references
        nRef = err_ref.shape[0]
        # chose reference averaging time < tEndAvgRef
        idxRef = np.where(averaging_times_ref<tEndAvgRef)[0]
        idxRef = idxRef[1:] # remove 1st averaging time = 0, where relative error = 1 because statistics have just been initialize to = 0 everywhere 
        for id_ref in range(nRef):
            # plot errors 
            if id_ref == 0:
                plt.semilogy(averaging_times_ref[idxRef], err_ref[id_ref,idxRef], color="gray", alpha = 0.3, label=f"Reference Realizations")
            else:
                plt.semilogy(averaging_times_ref[idxRef], err_ref[id_ref,idxRef], color="gray", alpha = 0.3)
        rlzAvg_err_ref = np.mean(err_ref, axis=0)
        plt.semilogy(averaging_times_ref[idxRef], rlzAvg_err_ref[idxRef], color = "black", linewidth = 2, label=f"Reference Rlz-Average")
        # log error values 
        ### print("\nRlz-Avg Reference: List of (averaging_time,", info['title'], ")")
        ### for time, err in zip(averaging_times_ref[idxRef], rlzAvg_err_ref[idxRef]):
        ###     print(f"({time:.2f}, {err:.4f})", end=', ')

        # RL non-converged:
        nrlz   = len(rlzArr)
        if nrlz <= 10:
            colors = tab_colors
        else:      
            colors = plt.cm.viridis_r(np.linspace(0, 1, nrlz))
        for irlz in range(nrlz):
            # eliminate the err_RL[i] = 1, which corresponds to time instances with no statistics data (u-mean not updated after initialized as zeros, thus relative error = 1) (also happens for t=tBeginAvg, where u-mean=0 everywhere)
            updatedErrorIdx_irlz     = np.where(err_RL[:,irlz]!=1.0)[0]
            err_RL_irlz              = err_RL[updatedErrorIdx_irlz, irlz]
            averagings_times_RL_irlz = averaging_times_RL[updatedErrorIdx_irlz]
            plt.semilogy(averagings_times_RL_irlz, err_RL_irlz, linewidth = 2, color=colors[irlz], label=f"RL Rlz {rlzArr[irlz]}")
            # log error values
            ### print(f"\nRL Rlz = {irlz}:")
            ### for time, err in zip(averagings_times_RL_irlz, err_RL_irlz):
            ###     print(f"({time:.3f}, {err:.3f})", end=', ')
        
        # add um target, if necessary
        if info['title'] == 'NRMSE_umean':
            ntk = len(averaging_times_ref[idxRef])
            plt.semilogy(averaging_times_ref[idxRef], um_target*np.ones(ntk), '--', color="tab:red", label=rf"Target $\textrm{{NRMSE}}(\overline{{u}}^+)={um_target:.1e}$")
        
        # configure plot
        plt.xlabel(r'$t^{+}$')
        plt.ylabel(info['ylabel'])
        if nrlz < 15:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()


    def build_RL_rewards_convergence(self, rlzArr, timeArr, rewards_total, rewards_err_umean, rewards_rhsfRatio):
        # Plot RL rewards along time, for each realization
        self.RL_variable_convergence_along_time("RL_rewards_total_convergence", "Reward", rlzArr, timeArr, rewards_total)
        self.RL_variable_convergence_along_time("RL_rewards_term_relL2Err_umean_convergence", r"$\textrm{NRMSE}(\overline{u}^{+})$", rlzArr, timeArr, rewards_err_umean)
        self.RL_variable_convergence_along_time("RL_rewards_term_rhsfRatio_convergence", "abs(RHS-f Ratio - 1)", rlzArr, timeArr, rewards_rhsfRatio)

    def build_RL_rewards_convergence_v2(self, rlzArr, timeArr, rewards_total, rewards_err_umean, rewards_err_rmsf, rewards_rhsfRatio):
        # Plot RL rewards along time, for each realization
        self.RL_variable_convergence_along_time("RL_rewards_total_convergence", "Reward", rlzArr, timeArr, rewards_total)
        self.RL_variable_convergence_along_time("RL_rewards_term_relL2Err_umean_convergence", r"$\textrm{NRMSE}(\overline{u}^{+})$", rlzArr, timeArr, rewards_err_umean)
        self.RL_variable_convergence_along_time("RL_rewards_term_relL2Err_urmsf_convergence", "u' relative L2 Error", rlzArr, timeArr, rewards_err_rmsf)
        self.RL_variable_convergence_along_time("RL_rewards_term_rhsfRatio_convergence", "abs(RHS-f Ratio - 1)", rlzArr, timeArr, rewards_rhsfRatio)


    def build_RL_actions_convergence(self, rlzArr, timeArr, actions):
        # Plot RL actions along time, for each realization
        nActDof = actions.shape[2]
        assert nActDof == 6, "visualizer.build_RL_actions_convergence method only works with actions of 6 d.o.f, specifically: DeltaRkk, DeltaTheta_z, DeltaTheta_y, DeltaTheta_x, DeltaXmap1, DeltaXmap2"
        dofNames      = ["DeltaRkk", "DeltaThetaZ", "DeltaThetaY", "DeltaThetaX", "DeltaXmap1", "DeltaXmap2"]
        dofNamesLatex = [r"$\Delta R_{kk}$", r"$\Delta \theta_{z}$", r"$\Delta \theta_{y}$", r"$\Delta \theta_{x}$", r"$\Delta x_{1}$", r"$\Delta x_{2}$"]
        for iActDof in range(nActDof):
            self.RL_variable_convergence_along_time_and_pdf(f"RL_actions_convergence_{dofNames[iActDof]}", dofNamesLatex[iActDof], rlzArr, timeArr, actions[:,:,iActDof])


    def build_RL_rewards_convergence_nohup(self, rewards_total, RL_rewards_term_relL2Err, rewards_term_rhsfRatio, inputRL_filepath=None):
        # Plot RL rewards along simulation steps from nohup information

        if inputRL_filepath is None:
            nax     = 3
        else:
            nax     = 4
        fig, ax = plt.subplots(1,nax,figsize=(5*nax,5))

        filename = os.path.join(self.postRlzDir, f"RL_rewards.{self.format}")
        print(f"\nMAKING PLOT {filename}")

        ax[0].plot(rewards_total)
        ax[0].set_ylabel("Total Reward")
        
        ax[1].semilogy(RL_rewards_term_relL2Err)
        ax[1].set_ylabel("Relative L2 Error (NRMSE)")

        ax[2].semilogy(rewards_term_rhsfRatio)
        ax[2].set_ylabel("abs(RHS-f Ratio - 1)")
        
        if inputRL_filepath is not None:
            config = configparser.ConfigParser()
            config.read(inputRL_filepath)
            rew_relL2Err_weight  = float(config.get("runner", "rew_umean_weight"))
            rew_rhsfRatio_weight = float(config.get("runner", "rew_rhsf_ratio_weight"))
            ax[3].semilogy(rew_relL2Err_weight  * RL_rewards_term_relL2Err, color='tab:orange', label="weighted Relative L2 Penalty Term (aim: 0)")
            ax[3].semilogy(rew_rhsfRatio_weight * rewards_term_rhsfRatio,   color='tab:green',  label="weighted RHS-F Ratio Penalty Term (aim: 0)")
            ax[3].legend(loc='lower right', frameon=True, fontsize=10)

        for i in range(nax):
            ax[i].set_xlabel("number simulation steps")
            ax[i].grid()

        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()


    def build_RL_rewards_convergence_nohup_v2(self, rewards_total, RL_rewards_term_relL2Err_umean, RL_rewards_term_relL2Err_urmsf, rewards_term_rhsfRatio, inputRL_filepath=None):
        # Plot RL rewards along simulation steps from nohup information

        if inputRL_filepath is None:
            nax     = 4
        else:
            nax     = 5
        fig, ax = plt.subplots(1,nax,figsize=(5*nax,5))

        filename = os.path.join(self.postRlzDir, f"RL_rewards.{self.format}")
        print(f"\nMAKING PLOT {filename}")

        ax[0].plot(rewards_total)
        ax[0].set_ylabel("Total Reward")
        
        ax[1].semilogy(RL_rewards_term_relL2Err_umean)
        ax[1].set_ylabel(r"$\textrm{NRMSE}(\overline{u}^{+})$")

        ax[2].semilogy(RL_rewards_term_relL2Err_urmsf)
        ax[2].set_ylabel(r"$\textrm{NRMSE}(u^{+}_{\textrm{rms}})$")

        ax[3].semilogy(rewards_term_rhsfRatio)
        ax[3].set_ylabel("abs(RHS-f Ratio - 1)")
        
        if inputRL_filepath is not None:
            config = configparser.ConfigParser()
            config.read(inputRL_filepath)
            rew_relL2Err_umean_weight = float(config.get("runner", "rew_umean_weight"))
            rew_relL2Err_urmsf_weight = float(config.get("runner", "rew_urmsf_weight"))
            rew_rhsfRatio_weight = float(config.get("runner", "rew_rhsf_ratio_weight"))
            ax[4].semilogy(rew_relL2Err_umean_weight  * RL_rewards_term_relL2Err_umean, color='tab:orange', label=r"weighted $\textrm{NRMSE}(\overline{u}^{+})$")
            ax[4].semilogy(rew_relL2Err_urmsf_weight  * RL_rewards_term_relL2Err_urmsf, color='tab:blue',   label=r"weighted $\textrm{NRMSE}(u^{+}_{\textrm{rms}})$")
            ax[4].semilogy(rew_rhsfRatio_weight * rewards_term_rhsfRatio,   color='tab:green',  label="weighted RHS-F Ratio Penalty Term")
            ax[4].legend(loc='lower right', frameon=True, fontsize=10)

        for i in range(nax):
            ax[i].set_xlabel("number simulation steps")
            ax[i].grid()

        plt.tight_layout()
        plt.savefig(filename, format=self.format)
        plt.close()


    def build_RL_actions_convergence_nohup(self, actions, actions_avg_freq = 500):
        # Plot RL actions along simulation steps from nohup information
        nSteps  = actions.shape[0]
        nActDof = actions.shape[1]
        assert nActDof == 6, "visualizer.build_RL_actions_convergence method only works with actions of 6 d.o.f, specifically: DeltaRkk, DeltaTheta_z, DeltaTheta_y, DeltaTheta_x, DeltaXmap1, DeltaXmap2"
        dofNames      = ["DeltaRkk", "DeltaThetaZ", "DeltaThetaY", "DeltaThetaX", "DeltaXmap1", "DeltaXmap2"]
        dofNamesLatex = [r"$\Delta R_{kk}$", r"$\Delta \theta_{z}$", r"$\Delta \theta_{y}$", r"$\Delta \theta_{x}$", r"$\Delta x_{1}$", r"$\Delta x_{2}$"]
        avgSteps  = np.arange(0, nSteps, actions_avg_freq)
        nAvgSteps = len(avgSteps) - 1

        # plot each action degree of freedom
        for iActDof in range(nActDof):
            filename = os.path.join(self.postRlzDir, f"RL_actions_convergence_{dofNames[iActDof]}.{self.format}")
            print(f"\nMAKING PLOT {filename}")

            fig, ax = plt.subplots(1,2,figsize=(10,5))
            ax[0].plot(actions[:,iActDof],'o',markersize=1)
            ax[0].set_xlabel("Number simulation step")
            ax[0].set_ylabel(dofNamesLatex[iActDof])

            for iAvgStep in range(nAvgSteps):
                startAvgIdx = avgSteps[iAvgStep]
                endAvgIdx   = avgSteps[iAvgStep+1]
                sns.kdeplot(y=actions[startAvgIdx:endAvgIdx,iActDof], label=f"Simulation steps {startAvgIdx}-{endAvgIdx}", ax=ax[1], cut=0, warn_singular=False, bw_method=0.01)
            ax[1].set_xlabel("KDE")
            ax[1].set_ylabel(dofNamesLatex[iActDof])
            ax[1].legend(loc='center right', fontsize=8)

            plt.tight_layout()
            plt.savefig(filename, format=self.format)
            plt.close()


# ------------------------------------ Energy Spectra -------------------------------------------

    def plot_spectral_turbulent_kinetic_energy_density_streamwise_velocity(
        self, file_details, tavg0, avg_y, avg_y_plus, avg_k, avg_k_plus, avg_lambda, avg_lambda_plus, avg_Euu, avg_Euu_plus,
    ):
        ls = ['-','--','-.',':']
        n_avg_probes = len(avg_y)

        ### # Plot kEuu vs lambda
        ### fname = os.path.join(self.postRlzDir, f"spectral_kEuu+_vs_lambda+_{file_details}.jpg")
        ### plt.figure(figsize=(12, 6))
        ### for i_avg_probe in range(n_avg_probes):
        ###     plt.loglog(avg_lambda_plus[i_avg_probe,:], avg_k[i_avg_probe,:] * avg_Euu_plus[i_avg_probe,:], lw=2, label=rf"$y^+={avg_y_plus[i_avg_probe]:.2f}$", color='k', linestyle=ls[i_avg_probe])
        ### #plt.xlabel(r"Wavelength, $\lambda_x^+$")
        ### plt.xlabel(r"$\lambda_x^+$")
        ### #plt.ylabel(r"Premultiplied Spectral Turbulent Kinetic Energy Density of Streamwise Velocity, $k_x\,E_{uu}^+$")
        ### plt.ylabel(r"$k_x\,E_{uu}^+$")
        ### plt.xscale('log')
        ### plt.grid(True)
        ### plt.legend()
        ### title_str = r"$t_{avg}^+=$" + rf"${tavg0:.0f}$"
        ### plt.title(title_str)
        ### plt.tight_layout()
        ### plt.savefig(fname)
        ### print(f"\nPlot kEuu+ vs. lambda+: {fname}")

        ### # Plot kEuu vs k
        ### fname = os.path.join(self.postRlzDir, f"spectral_kEuu+_vs_k_{file_details}.jpg")
        ### plt.figure(figsize=(12, 6))
        ### for i_avg_probe in range(n_avg_probes):
        ###     plt.loglog(avg_k[i_avg_probe,:], avg_k[i_avg_probe,:] * avg_Euu_plus[i_avg_probe,:], lw=2, label=rf"$y^+={avg_y_plus[i_avg_probe]:.2f}$", color='k', linestyle=ls[i_avg_probe])
        ### #plt.xlabel(r"Wavenumber, $k_x$")
        ### plt.xlabel(r"$k_x$")
        ### #plt.ylabel(r"Premultiplied Spectral Turbulent Kinetic Energy Density of Streamwise Velocity, $k_x\,E_{uu}^+$")
        ### plt.ylabel(r"$k_x\,E_{uu}^+$")
        ### plt.xscale('log')
        ### plt.grid(True)
        ### plt.legend()
        ### title_str = r"$t_{avg}^+=$" + rf"${tavg0:.0f}$"
        ### plt.title(title_str)
        ### plt.tight_layout()
        ### plt.savefig(fname)
        ### print(f"\nPlot kEuu+ vs. k: {fname}")
            
        # Plot Euu vs kplus
        fname = os.path.join(self.postRlzDir, f"spectral_Euu+_vs_k+_{file_details}")
        plt.figure(figsize=(12, 6))
        for i_avg_probe in range(n_avg_probes):
            plt.loglog(avg_k_plus[i_avg_probe,:], avg_Euu_plus[i_avg_probe,:], lw=2, label=rf"$y^+={avg_y_plus[i_avg_probe]:.2f}$", color='k', linestyle=ls[i_avg_probe])
        # Theoretical decay: Euu decays as k^(-5/3) -> slope Euu/k decays ~ 1^(-5/3) -> slope log(Euu)/log(k) ~ (-5/3)
        k_plus_slope   = np.linspace(10**(-3.0), 10**(-1.9), 50)
        Euu_plus_slope = 1e-6*k_plus_slope**(-5.0/3.0)
        plt.loglog(k_plus_slope, Euu_plus_slope, '--', color="tab:blue", lw=2)
        plt.text(10**(-2.0), 10**(-2.2), r"$\sim k_x^{+(-5/3)}$", fontsize=18)
        #plt.xlabel(r"Wavenumber, $k_x$")
        plt.xlabel(r"$k_x^+$")
        #plt.ylabel(r"Premultiplied Spectral Turbulent Kinetic Energy Density of Streamwise Velocity, $k_x\,E_{uu}^+$")
        plt.ylabel(r"$E_{uu}^+$")
        plt.xscale('log')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        title_str = r"$t_{avg}^+=$" + rf"${tavg0:.0f}$"
        plt.title(title_str)
        plt.tight_layout()
        plt.savefig(fname, format=self.format)
        print(f"\nPlot KEuu+ vs. k+: {fname}")

    
    def build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_fig(self, avg_y_plus, avg_k_plus_RL, avg_k_plus_nonRL, avg_k_plus_ref, avg_Euu_plus_RL, avg_Euu_plus_nonRL, avg_Euu_plus_ref, avg_time_RL, avg_time_nonRL, global_step, ylim=[10**(-7.5),1.0]):
        colors = ['black','tab:blue','tab:green','tab:orange']
        n_avg_probes = len(avg_y_plus)
        # Plot Euu vs kplus
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(n_avg_probes):
            plt.loglog(avg_k_plus_ref[i],   avg_Euu_plus_ref[i],   color=colors[i], linestyle='-',  lw=2, label=rf"Reference, $y^+={avg_y_plus[i]:.2f}$")
            plt.loglog(avg_k_plus_nonRL[i], avg_Euu_plus_nonRL[i], color=colors[i], linestyle='--', lw=2, label=rf"non-RL, $y^+={avg_y_plus[i]:.2f}$")
            plt.loglog(avg_k_plus_RL[i],    avg_Euu_plus_RL[i],    color=colors[i], linestyle=':',  lw=2, label=rf"RL, $y^+={avg_y_plus[i]:.2f}$")
        # Theoretical decay: Euu decays as k^(-5/3) -> slope Euu/k decays ~ 1^(-5/3) -> slope log(Euu)/log(k) ~ (-5/3)
        k_plus_slope   = np.linspace(10**(-3.0), 10**(-1.9), 50)
        Euu_plus_slope = 1e-9*k_plus_slope**(-5.0/3.0)
        plt.loglog(k_plus_slope, Euu_plus_slope, '-.', color="tab:gray", lw=2, label=r"$\sim k_x^{+(-5/3)}$")
        ###plt.text(10**(-2.0), 10**(-2.2), r"$\sim k_x^{+(-5/3)}$", fontsize=18)
        # plot parameters
        plt.xlabel(r"$k_x^+$")      #plt.xlabel(r"Wavenumber, $k_x$")
        plt.ylabel(r"$E_{uu}^+$")   #plt.ylabel(r"Premultiplied Spectral Turbulent Kinetic Energy Density of Streamwise Velocity, $k_x\,E_{uu}^+$")
        plt.ylim(ylim)
        plt.xscale('log')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$\\ RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = {global_step}")
        plt.tight_layout()
        return fig

    def build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_frame(self, frames, avg_y_plus, avg_k_plus_RL, avg_k_plus_nonRL, avg_k_plus_ref, avg_Euu_plus_RL, avg_Euu_plus_nonRL, avg_Euu_plus_ref, avg_time_RL, avg_time_nonRL, global_step):
        fig = self.build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_fig(avg_y_plus, avg_k_plus_RL, avg_k_plus_nonRL, avg_k_plus_ref, avg_Euu_plus_RL, avg_Euu_plus_nonRL, avg_Euu_plus_ref, avg_time_RL, avg_time_nonRL, global_step)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

    def build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_fig_from_dicts(self, avg_y_plus_dict, avg_k_plus_RL_dict, avg_k_plus_nonRL_dict, avg_k_plus_ref_dict, avg_Euu_plus_RL_dict, avg_Euu_plus_nonRL_dict, avg_Euu_plus_ref_dict, avg_time_RL, avg_time_nonRL, global_step, ylim=[10**(-7.5),1.0]):
        #import pdb; pdb.set_trace()
        y_coord_name_list = list(avg_y_plus_dict.keys()) 
        n_y_coord         = len(y_coord_name_list)
        colors_list       = ['black','tab:blue','tab:green','tab:orange']
        assert n_y_coord == 4
        colors_dict       = dict(map(lambda k,v : (k,v), y_coord_name_list, colors_list))
        # Plot Euu vs kplus
        fig, ax = plt.subplots(figsize=(12, 6))
        for y_coord in y_coord_name_list:
            plt.loglog(avg_k_plus_ref_dict[y_coord],    avg_Euu_plus_ref_dict[y_coord],   color=colors_dict[y_coord], linestyle='-',  lw=2, label=rf"$y^+={avg_y_plus_dict[y_coord]:.2f}$, Reference")
            plt.loglog(avg_k_plus_nonRL_dict[y_coord],  avg_Euu_plus_nonRL_dict[y_coord], color=colors_dict[y_coord], linestyle='--', lw=2, label=rf"$y^+={avg_y_plus_dict[y_coord]:.2f}$, Uncontrolled")
            if avg_k_plus_RL_dict is not None:
                plt.loglog(avg_k_plus_RL_dict[y_coord], avg_Euu_plus_RL_dict[y_coord],    color=colors_dict[y_coord], linestyle=':',  lw=2, label=rf"$y^+={avg_y_plus_dict[y_coord]:.2f}$, RL Framework")
        # Theoretical decay: Euu decays as k^(-5/3) -> slope Euu/k decays ~ 1^(-5/3) -> slope log(Euu)/log(k) ~ (-5/3)
        k_plus_slope   = np.linspace(10**(-2.0), 10**(-1.0), 50)
        Euu_plus_slope = k_plus_slope**(-5.0/3.0) * (0.1 / k_plus_slope[0]**(-5.0/3.0))
        plt.loglog(k_plus_slope, Euu_plus_slope, '-.', color="tab:gray", lw=2, label=r"$\sim k_x^{+(-5/3)}$")
        ###plt.text(10**(-2.0), 10**(-2.2), r"$\sim k_x^{+(-5/3)}$", fontsize=18)
        # plot parameters
        plt.xlabel(r"$k_x^+$")      #plt.xlabel(r"Wavenumber, $k_x$")
        plt.ylabel(r"$E_{uu}^+$")   #plt.ylabel(r"Premultiplied Spectral Turbulent Kinetic Energy Density of Streamwise Velocity, $k_x\,E_{uu}^+$")
        plt.ylim(ylim)
        plt.xscale('log')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if avg_time_RL is None:
            plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$")
        else:
            plt.title(rf"non-RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_nonRL:.2f}$\\ RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = {global_step}")
        plt.tight_layout()
        return fig

    def build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_frame_from_dicts(self, frames, avg_y_plus_dict, avg_k_plus_RL_dict, avg_k_plus_nonRL_dict, avg_k_plus_ref_dict, avg_Euu_plus_RL_dict, avg_Euu_plus_nonRL_dict, avg_Euu_plus_ref_dict, avg_time_RL, avg_time_nonRL, global_step, ylim=[10**(-7.5),1.0]):
        fig = self.build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_fig_from_dicts(avg_y_plus_dict, avg_k_plus_RL_dict, avg_k_plus_nonRL_dict, avg_k_plus_ref_dict, avg_Euu_plus_RL_dict, avg_Euu_plus_nonRL_dict, avg_Euu_plus_ref_dict, avg_time_RL, avg_time_nonRL, global_step, ylim)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

# ------------------------------------ RHS terms of drhou/dt, drhov/dt, drhow/dt -------------------------------------------

    def build_rhovel_fig_from_dicts(self, ensemble_dict, avg_time_RL, global_step, ylim):

        # Variables keys (used in probes data files)
        time_key         = "# t [s]"
        y_plus_key       = 'y_plus [m]'
        rhou_key         = " u [m/s]"    # Assuming constant rho_0=1
        rhov_key         = " v [m/s]"
        rhow_key         = " w [m/s]"
        rhovel_keys      = [rhou_key, rhov_key, rhow_key]
        rhou_inv_key     = " rhou_inv_flux [kg/m2s2]"
        rhov_inv_key     = " rhov_inv_flux [kg/m2s2]"
        rhow_inv_key     = " rhow_inv_flux [kg/m2s2]"
        rhovel_inv_keys  = [rhou_inv_key, rhov_inv_key, rhow_inv_key]
        rhou_vis_key     = " rhou_vis_flux [kg/m2s2]"
        rhov_vis_key     = " rhov_vis_flux [kg/m2s2]"
        rhow_vis_key     = " rhow_vis_flux [kg/m2s2]"
        rhovel_vis_keys  = [rhou_vis_key, rhov_vis_key, rhow_vis_key]
        f_rhou_key       = " f_rhou [kg/m2s2]"
        f_rhov_key       = " f_rhov [kg/m2s2]"
        f_rhow_key       = " f_rhow [kg/m2s2]"
        f_rhovel_keys    = [f_rhou_key, f_rhov_key, f_rhow_key]
        rl_f_rhou_key    = " rl_f_rhou [kg/m2s2]"
        rl_f_rhov_key    = " rl_f_rhov [kg/m2s2]"
        rl_f_rhow_key    = " rl_f_rhow [kg/m2s2]"
        rl_f_rhovel_keys = [rl_f_rhou_key, rl_f_rhov_key, rl_f_rhow_key]
        vel_names        = ['u', 'v', 'w']
        n_dim            = 3

        #import pdb; pdb.set_trace()
        y_coord_name_list = list(ensemble_dict.keys()) 
        n_y_coord         = len(y_coord_name_list)
        assert n_y_coord == 4
        colors_list       = ['black','tab:blue','tab:orange','tab:green', 'tab:red']
        figs_list         = []
        for dim in range(n_dim):
            vel_name = vel_names[dim]
            fig, ax = plt.subplots(2,2,figsize=(12,6))
            for i in range(n_y_coord):
                y_coord = y_coord_name_list[i]
                y_plus_value = ensemble_dict[y_coord][y_plus_key]
                row, col = divmod(i, 2)
                if i == 0:
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][rhovel_keys[dim]],      color=colors_list[0], lw=2, label=rf"$\rho {vel_name}$ value")
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][rhovel_inv_keys[dim]],  color=colors_list[1], lw=2, label=rf"inv. term")
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][rhovel_vis_keys[dim]],  color=colors_list[2], lw=2, label=rf"vis. term")
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][f_rhovel_keys[dim]],    color=colors_list[3], lw=2, label=rf"forcing term")
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][rl_f_rhovel_keys[dim]], color=colors_list[4], lw=2, label=rf"RL term")
                else:
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][rhovel_keys[dim]],      color=colors_list[0], lw=2)
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][rhovel_inv_keys[dim]],  color=colors_list[1], lw=2)
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][rhovel_vis_keys[dim]],  color=colors_list[2], lw=2)
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][f_rhovel_keys[dim]],    color=colors_list[3], lw=2)
                    ax[row, col].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][rl_f_rhovel_keys[dim]], color=colors_list[4], lw=2)
                ax[row, col].set_title(rf"$y^+={y_plus_value:.2f}$", pad=10)
                if ylim is not None:
                    ax[row, col].set_ylim(ylim)
                ax[row, col].grid()
                #ax[row, col].set_xlabel(r"averaging time [s]", labelpad=10)
            fig.suptitle(rf"RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = ${global_step}$", y=0.97)
            fig.tight_layout()
            fig.subplots_adjust(right=0.80)                         
            fig.legend(loc='center left', bbox_to_anchor=(0.83,0.5))
            figs_list.append(fig)
            plt.close(fig)
        return figs_list

    def build_rhovel_frame_from_dicts(self, frames_list, ensemble_dict, avg_time_RL, global_step, ylim=None):
        figs_list = self.build_rhovel_fig_from_dicts(ensemble_dict, avg_time_RL, global_step, ylim)
        assert len(figs_list) == len(frames_list)
        for i in range(len(figs_list)):
            fig = figs_list[i]
            fig.canvas.draw()
            img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            frames_list[i].append(img)
            plt.close()
        return frames_list


# ------------------------------------ RHS terms of d_DeltaRij_j -------------------------------------------

    def build_d_DeltaRij_j_fig_from_dicts(self, ensemble_dict, avg_time_RL, global_step):

        # Variables keys (used in probes data files)
        time_key         = "# t [s]"
        y_plus_key       = 'y_plus [m]'
        d_DeltaRxj_j_key = " d_DeltaRxj_j [m/s2]"
        d_DeltaRyj_j_key = " d_DeltaRyj_j [m/s2]"
        d_DeltaRzj_j_key = " d_DeltaRzj_j [m/s2]"
        d_DeltaRxx_x_key = " d_DeltaRxx_x [m/s2]"
        d_DeltaRxy_x_key = " d_DeltaRxy_x [m/s2]"
        d_DeltaRxz_x_key = " d_DeltaRxz_x [m/s2]"
        d_DeltaRxy_y_key = " d_DeltaRxy_y [m/s2]"
        d_DeltaRyy_y_key = " d_DeltaRyy_y [m/s2]"
        d_DeltaRyz_y_key = " d_DeltaRyz_y [m/s2]"
        d_DeltaRxz_z_key = " d_DeltaRxz_z [m/s2]"
        d_DeltaRyz_z_key = " d_DeltaRyz_z [m/s2]"
        d_DeltaRzz_z_key = " d_DeltaRzz_z [m/s2]"

        #import pdb; pdb.set_trace()
        y_coord_name_list = list(ensemble_dict.keys()) 
        n_y_coord         = len(y_coord_name_list)
        colors_list       = ['black','tab:blue','tab:orange','tab:green', 'tab:red']
        assert n_y_coord == 4
        fig, ax = plt.subplots(3,4,figsize=(16,10))
        for i in range(n_y_coord):
            y_coord = y_coord_name_list[i]
            y_plus_value = ensemble_dict[y_coord][y_plus_key]
            # d_DeltaRxj_j
            ax[0,0].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRxj_j_key], color=colors_list[i], lw=2, label=rf"$y^+={y_plus_value:.2f}$")
            ax[0,0].set_title(r"$\partial \Delta R_{xj} / \partial x_j$", pad=10)
            try:    # for some cases d_DeltaRxx_x, d_DeltaRxy_y, d_DeltaRxz_z is not stored in probes data
                # d_DeltaRxx_x
                ax[0,1].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRxx_x_key], color=colors_list[i], lw=2)
                ax[0,1].set_title(r"$\partial \Delta R_{xx} / \partial x$", pad=10)
                # d_DeltaRxy_y
                ax[0,2].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRxy_y_key], color=colors_list[i], lw=2)
                ax[0,2].set_title(r"$\partial \Delta R_{xy} / \partial y$", pad=10)
                # d_DeltaRxz_z
                ax[0,3].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRxz_z_key], color=colors_list[i], lw=2)
                ax[0,3].set_title(r"$\partial \Delta R_{xz} / \partial z$", pad=10)
            except KeyError:
                pass
            #####
            # d_DeltaRyj_j
            ax[1,0].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRyj_j_key], color=colors_list[i], lw=2)
            ax[1,0].set_title(r"$\partial \Delta R_{yj} / \partial x_j$", pad=10)
            try: 
                # d_DeltaRyx_x
                ax[1,1].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRxy_x_key], color=colors_list[i], lw=2)
                ax[1,1].set_title(r"$\partial \Delta R_{yx} / \partial x$", pad=10)
                # d_DeltaRyy_y
                ax[1,2].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRyy_y_key], color=colors_list[i], lw=2)
                ax[1,2].set_title(r"$\partial \Delta R_{yy} / \partial y$", pad=10)
                # d_DeltaRyz_z
                ax[1,3].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRyz_z_key], color=colors_list[i], lw=2)
                ax[1,3].set_title(r"$\partial \Delta R_{yz} / \partial z$", pad=10)
            except KeyError:
                pass
            #####
            # d_DeltaRzj_j
            ax[2,0].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRzj_j_key], color=colors_list[i], lw=2)
            ax[2,0].set_title(r"$\partial \Delta R_{zj} / \partial x_j$", pad=10)
            try:
                # d_DeltaRzx_x
                ax[2,1].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRxz_x_key], color=colors_list[i], lw=2)
                ax[2,1].set_title(r"$\partial \Delta R_{zx} / \partial x$", pad=10)
                # d_DeltaRzy_y
                ax[2,2].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRyz_y_key], color=colors_list[i], lw=2)
                ax[2,2].set_title(r"$\partial \Delta R_{zy} / \partial y$", pad=10)
                # d_DeltaRzz_z
                ax[2,3].plot(ensemble_dict[y_coord][time_key], ensemble_dict[y_coord][d_DeltaRzz_z_key], color=colors_list[i], lw=2)
                ax[2,3].set_title(r"$\partial \Delta R_{zz} / \partial z$", pad=10)
            except KeyError:
                pass
        for i in range(12):
            row,col = divmod(i,4)
            ax[row,col].grid()
            #ax[row, col].set_xlabel(r"averaging time [s]", labelpad=10)
        fig.suptitle(rf"RL: $t_{{\textrm{{avg}}}}^{{+}} = {avg_time_RL:.2f}$, train step = ${global_step}$", y=0.97)
        fig.tight_layout()
        fig.subplots_adjust(right=0.85)                         
        fig.legend(loc='center left', bbox_to_anchor=(0.86,0.5))
        return fig

    def build_d_DeltaRij_j_frame_from_dicts(self, frames, ensemble_dict, avg_time_RL, global_step):
        fig = self.build_d_DeltaRij_j_fig_from_dicts(ensemble_dict, avg_time_RL, global_step)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

# ------------------------------------ Bulk and Wall values ------------------------------------

    def plot_bulk_wall_values(self, averaging_time_nonRL, averaging_time_accum_RL, 
                              u_b_ref, u_b_nonRL, u_b_RL,
                              avg_u_b_ref, avg_u_b_nonRL, avg_u_b_RL,
                              tau_w_num_ref, tau_w_num_nonRL, tau_w_num_RL,
                              u_tau_num_ref, u_tau_num_nonRL, u_tau_num_RL):

        print("\nBuiding plots of bulk and wall values...")

        N_nonRL = len(averaging_time_nonRL)

        # --- (inst) u_bulk plot ---
        plt.plot( averaging_time_nonRL,    u_b_ref * np.ones(N_nonRL), linestyle = '-',                                linewidth = 2, color = "k",             label = r'Reference' )
        plt.plot( averaging_time_nonRL,    u_b_nonRL,                  linestyle = '--', marker = 's', markersize = 4, linewidth = 2, color = plt.cm.tab10(0), label = r'Uncontrolled' )
        plt.plot( averaging_time_accum_RL, u_b_RL,                     linestyle = ':',  marker = '^', markersize = 4, linewidth = 2, color = plt.cm.tab10(3), label = r'RL Framework' )
        plt.xlabel( r'Cummulative averaging time $t_{avg}^+$' )
        plt.ylabel( r'Numerical $u^{+}_b$' )
        #plt.ylim(14,14.8)
        #plt.yticks(np.arange(14,14.8,0.1))
        #plt.grid(which='major',axis='y')
        plt.grid(which='both',axis='y')
        plt.tick_params( axis = 'both', pad = 7.5 )
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend(loc='upper right', frameon=True)
        filename = os.path.join(self.postRlzDir, f'numerical_u_bulk.{self.format}')
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

        # --- avg_u_bulk plot ---
        plt.plot( averaging_time_nonRL,    avg_u_b_ref * np.ones(N_nonRL), linestyle = '-',                                linewidth = 2, color = "k",             label = r'Reference' )
        plt.plot( averaging_time_nonRL,    avg_u_b_nonRL,                  linestyle = '--', marker = 's', markersize = 4, linewidth = 2, color = plt.cm.tab10(0), label = r'Uncontrolled' )
        plt.plot( averaging_time_accum_RL, avg_u_b_RL,                     linestyle = ':',  marker = '^', markersize = 4, linewidth = 2, color = plt.cm.tab10(3), label = r'RL Framework' )
        plt.xlabel( r'Cummulative averaging time $t_{avg}^+$' )
        plt.ylabel( r'Numerical $\overline{u}^{+}_b$' )
        plt.ylim(14,14.8)
        plt.yticks(np.arange(14,14.8,0.1))
        plt.grid(which='major',axis='y')
        plt.tick_params( axis = 'both', pad = 7.5 )
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend(loc='upper right', frameon=True)
        filename = os.path.join(self.postRlzDir, f'numerical_avg_u_bulk.{self.format}')
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

        # --- avg_u_bulk & (inst) u_bulk plot ---
        plt.plot( averaging_time_nonRL,    avg_u_b_ref * np.ones(N_nonRL), linestyle = '-',                                zorder = 1, linewidth = 1, color = "k",             label = r'$\overline{u}^{+}_b$ Reference' )
        plt.plot( averaging_time_nonRL,    avg_u_b_nonRL,                  linestyle = '-',  marker = 's', markersize = 4, zorder = 1, linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}_b$ Uncontrolled' )
        plt.plot( averaging_time_accum_RL, avg_u_b_RL,                     linestyle = '-',  marker = 'v', markersize = 4, zorder = 1, linewidth = 1, color = plt.cm.tab10(3), label = r'$\overline{u}^{+}_b$ RL Framework' )
        plt.plot( averaging_time_nonRL,    u_b_ref * np.ones(N_nonRL),     linestyle = '--',                               zorder = 0, linewidth = 1, color = "k",             label = r'${u}^{+}_b$ Reference' )
        plt.plot( averaging_time_nonRL,    u_b_nonRL,                      linestyle = '--', marker = 'o', markersize = 4, zorder = 0, linewidth = 1, color = plt.cm.tab10(0), label = r'${u}^{+}_b$ Uncontrolled' )
        plt.plot( averaging_time_accum_RL, u_b_RL,                         linestyle = '--', marker = '^', markersize = 4, zorder = 0, linewidth = 1, color = plt.cm.tab10(3), label = r'${u}^{+}_b$ RL Framework' )
        plt.xlabel( r'Cummulative averaging time $t_{avg}^+$' )
        plt.ylabel( r'Numerical avg. $\overline{u}^{+}_b$ and inst. $\overline{u}^{+}_b$' )
        plt.ylim(12,17)
        plt.yticks(np.arange(12,17,1.0))
        plt.grid(which='major',axis='y')
        plt.tick_params( axis = 'both', pad = 7.5 )
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend(loc='upper right', frameon=True)
        filename = os.path.join(self.postRlzDir, f'numerical_inst_avg_u_bulk.{self.format}')
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

        # --- tau_w plot ---
        plt.plot( averaging_time_nonRL,    tau_w_num_ref * np.ones(N_nonRL),  linestyle = '-',                            linewidth = 2, color = "k",             label = r'Reference' )
        plt.plot( averaging_time_nonRL,    tau_w_num_nonRL,             linestyle = '--',   marker = 's', markersize = 4, linewidth = 2, color = plt.cm.tab10(0), label = r'Uncontrolled' )
        plt.plot( averaging_time_accum_RL, tau_w_num_RL,                linestyle=':',      marker = '^', markersize = 4, linewidth = 2, color = plt.cm.tab10(3), label = r'RL Framework' )
        plt.xlabel( r'Cummulative averaging time $t_{avg}^+$' )
        plt.ylabel( r'Numerical $\tau_w$' )
        plt.ylim([0.0, 1.1])
        plt.grid(which='both',axis='y')
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend(loc='upper right', frameon=True)
        plt.tick_params( axis = 'both', pad = 7.5 )
        filename = os.path.join(self.postRlzDir, f'numerical_tau_w.{self.format}')
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

        # --- u_tau plot ---
        plt.plot( averaging_time_nonRL,    u_tau_num_ref * np.ones(N_nonRL),  linestyle = '-',                           linewidth = 2, color = "k",             label = r'Reference' )
        plt.plot( averaging_time_nonRL,    u_tau_num_nonRL,             linestyle = '--',  marker = 's', markersize = 4, linewidth = 2, color = plt.cm.tab10(0), label = r'Uncontrolled' )
        plt.plot( averaging_time_accum_RL, u_tau_num_RL,                linestyle=':',     marker = '^', markersize = 4, linewidth = 2, color = plt.cm.tab10(3), label = r'RL Framework' )
        plt.xlabel( r'Cummulative averaging time $t_{avg}^+$' )
        plt.ylabel( r'Numerical $u_\tau$' )
        plt.ylim([0.0, 1.1])
        plt.grid(which='both',axis='y')
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend(loc='upper right', frameon=True)
        plt.tick_params( axis = 'both', pad = 7.5 )
        filename = os.path.join(self.postRlzDir, f'numerical_u_tau.{self.format}')
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

# --------------------------------------- Velocity profiles vs y+ ---------------------------------------

    def build_velocity_profiles(self, y_plus_ref, y_plus_nonRL, y_plus_RL,
                                avg_u_plus_ref, avg_u_plus_nonRL, avg_u_plus_RL,
                                rmsf_u_plus_rer, rmsf_u_plus_nonRL, rmsf_u_plus_RL,
                                rmsf_v_plus_rer, rmsf_v_plus_nonRL, rmsf_v_plus_RL,
                                rmsf_w_plus_rer, rmsf_w_plus_nonRL, rmsf_w_plus_RL,
                                TKE_ref, TKE_nonRL, TKE_RL):

        ### Plot u+ vs. y+
        print("\nBuilding plots...")
        # Clear plot
        plt.clf()
        # Read & Plot data
        plt.plot( y_plus_ref, avg_u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
        for i_RL in range(n_RL):
            if n_RL < 10:
                plt.plot( y_plus_RL[i_RL], avg_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
            else:
                plt.plot( y_plus_RL[i_RL], avg_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
        plt.plot( y_plus_nonRL, avg_u_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s' )
        # Configure plot
        plt.xlim( 1.0, 2.0e2 )
        plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
        plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
        plt.xscale( 'log' )
        plt.xlabel( 'y+' )
        plt.ylim( 0.0, 20.0 )
        plt.yticks( np.arange( 0.0, 20.1, 5.0 ) )
        plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
        #plt.yscale( 'log' )
        plt.ylabel( 'u+')
        legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.tick_params( axis = 'both', pad = 7.5 )
        filename = os.path.join(self.postRlzDir, f"u_plus_vs_y_plus_{iteration}.{self.format}")
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

        ### Plot u-rmsf 
        # Read & Plot data
        plt.plot( y_plus_ref, rmsf_u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label=f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
        for i_RL in range(n_RL):
            if n_RL < 10:
                plt.plot( y_plus_RL[i_RL], rmsf_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
            else:
                plt.plot( y_plus_RL[i_RL], rmsf_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
        plt.plot( y_plus_nonRL, rmsf_u_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s' )
        # Configure plot
        plt.xlim( 1.0, 2.0e2 )
        plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
        plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
        plt.xscale( 'log' )
        plt.xlabel( 'y+' )
        plt.ylim( 0.0, 3.0 )
        plt.yticks( np.arange( 0.0, 3.1, 0.5 ) )
        plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
        #plt.yscale( 'log' )
        plt.ylabel( 'u_rms+' )
        #legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.text( 1.05, 1.0, 'u_rms+' )
        legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.tick_params( axis = 'both', pad = 7.5 )
        filename = os.path.join(self.postRlzDir, f"u_rmsf_plus_vs_y_plus_{iteration}.{self.format}")
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

        ### Plot v-rmsf
        # Read & Plot data
        plt.plot( y_plus_ref, rmsf_v_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
        for i_RL in range(n_RL):
            if n_RL < 10:
                plt.plot( y_plus_RL[i_RL], rmsf_v_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
            else:
                plt.plot( y_plus_RL[i_RL], rmsf_v_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
        plt.plot( y_plus_nonRL, rmsf_v_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
        # Configure plot
        plt.xlim( 1.0, 2.0e2 )
        plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
        plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
        plt.xscale( 'log' )
        plt.xlabel( 'y+' )
        plt.ylim( 0.0, 3.0 )
        plt.yticks( np.arange( 0.0, 3.1, 0.5 ) )
        plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
        #plt.yscale( 'log' )
        plt.ylabel( 'v_rms+' )
        #legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.text( 17.5, 0.2, 'v_rms+' )
        legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.tick_params( axis = 'both', pad = 7.5 )
        filename = os.path.join(self.postRlzDir, f"v_rmsf_plus_vs_y_plus_{iteration}.{self.format}")
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

        ### Plot w-rmsf
        # Read & Plot data
        plt.plot( y_plus_ref, rmsf_w_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
        for i_RL in range(n_RL):
            if n_RL < 10:
                plt.plot( y_plus_RL[i_RL], rmsf_w_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
            else:
                plt.plot( y_plus_RL[i_RL], rmsf_w_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
        plt.plot( y_plus_nonRL, rmsf_w_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
        # Configure plot
        plt.xlim( 1.0, 2.0e2 )
        plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
        plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
        plt.xscale( 'log' )
        plt.xlabel( 'y+' )
        plt.ylim( 0.0, 3.0 )
        plt.yticks( np.arange( 0.0, 3.1, 0.5 ) )
        plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
        #plt.yscale( 'log' )
        plt.ylabel( 'w_rms+' )
        #legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.text( 17.5, 0.2, 'w_rms+' )
        legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.tick_params( axis = 'both', pad = 7.5 )
        filename = os.path.join(self.postRlzDir, f"w_rmsf_plus_vs_y_plus_{iteration}.{self.format}")
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

        ### Plot TKE
        # Read & Plot data
        plt.plot( y_plus_ref, TKE_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
        for i_RL in range(n_RL):
            if n_RL < 10:
                plt.plot( y_plus_RL[i_RL], TKE_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
            else:
                plt.plot( y_plus_RL[i_RL], TKE_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
        plt.plot( y_plus_nonRL, TKE_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
        # Configure plot
        plt.xlim( 1.0, 2.0e2 )
        plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
        plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
        plt.xscale( 'log' )
        plt.xlabel( 'y+' )
        plt.ylim( 0.0, 3.0 )
        plt.yticks( np.arange( 0.0, 5.0, 1.0 ) )
        plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
        #plt.yscale( 'log' )
        plt.ylabel( 'TKE+' )
        #legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.text( 17.5, 0.2, 'TKE+' )
        legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
        plt.tick_params( axis = 'both', pad = 7.5 )
        filename = os.path.join(self.postRlzDir, f"tke_plus_vs_y_plus_{iteration}.{self.format}")
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.clf()
        print(f"\nBuild plot: '{filename}'")

# --------------------- Error plots of velocity components ------------------------

    def build_velocity_error_plot(self, avg_time_nonRL, avg_time_RL, err_avg_nonRL, err_avg_RL, err_rmsf_nonRL, err_rmsf_RL, vel_component='u', error_num='2'):
        plt.clf()
        plt.semilogy( avg_time_nonRL, err_avg_nonRL,  linestyle = '-', marker = 's', markersize = 2, linewidth = 1, color = plt.cm.tab10(0), zorder = 0, label = rf'$\overline{{{vel_component}}}^+$ Uncontrolled' )
        plt.semilogy( avg_time_RL,    err_avg_RL,     linestyle = '-', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(3), zorder = 1, label = rf'$\overline{{{vel_component}}}^+$ RL Framework' )
        plt.semilogy( avg_time_nonRL, err_rmsf_nonRL, linestyle = ':', marker = 'o', markersize = 2, linewidth = 1, color = plt.cm.tab10(0), zorder = 0, label = rf'${vel_component}_\textrm{{rms}}^+$ Uncontrolled' )
        plt.semilogy( avg_time_RL,    err_rmsf_RL,    linestyle = ':', marker = 'v', markersize = 2, linewidth = 1, color = plt.cm.tab10(3), zorder = 1, label = rf'${vel_component}_\textrm{{rms}}^+$ RL Framework' )
        plt.xlabel(r'Cummulative averaging time $t_{avg}^+$' )
        plt.ylabel(rf'$L_{error_num}$ Error' )
        plt.grid(which='both',axis='y')
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # (loc='upper right', frameon=True, framealpha=1.0, fancybox=True)
        plt.legend(loc='upper right', frameon=True)
        plt.tick_params( axis = 'both', pad = 7.5 )
        plt.tight_layout()
        filename = os.path.join(self.postRlzDir, f'L{error_num}_error_{vel_component}.{self.format}')
        plt.savefig( filename, format = self.format, bbox_inches = 'tight' )
        plt.close()
        print(f"\nBuild plot: '{filename}'")