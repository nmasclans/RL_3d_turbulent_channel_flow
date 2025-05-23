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
plt.rc( 'savefig',    format = "jpg", dpi = 600)

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

    def __init__(self, postRlzDir):

        self.postRlzDir = postRlzDir

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

    def build_um_fig(self, yplus_odt, yplus_ref, um_odt, um_ref, avg_time, ylim=None):
        
        fig, ax = plt.subplots()
        plt.semilogx(yplus_ref, um_ref,               '-',  color="black",      lw=2, label=r"Reference $t^+=900$")
        plt.semilogx(yplus_odt, um_odt,               '-.', color="tab:green",  lw=2, label=r"RL Framework")

        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel(r"$y^{+}$")
        plt.ylabel(r"$\overline{u}^{+}$")
        plt.title(rf"$t^+ = {avg_time:.2f}$")
        plt.yticks([0.0, 5.0, 10.0, 15.0, 20.0])
        plt.legend(loc='lower right')
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    

    def build_um_frame(self, frames, yplus_odt, yplus_ref, um_odt, um_ref, avg_time, ylim=None):
        
        fig = self.build_um_fig(yplus_odt, yplus_ref, um_odt, um_ref, avg_time, ylim)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames


    def build_urmsf_fig(self, yplus_odt, yplus_ref, urmsf_odt, urmsf_ref, avg_time, ylim=None):
        
        fig, ax = plt.subplots()
        plt.semilogx(yplus_odt, urmsf_odt, '-k',  label=r"Converging $u^{+}_{\textrm{rmsf}}$")
        plt.semilogx(yplus_ref, urmsf_ref, '--k', label=r"Reference  $u^{+}_{\textrm{rmsf}}$")
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel(r"$y^{+}$")
        plt.ylabel(r"$u^{+}_{\textrm{rmsf}}$")
        plt.grid(axis="y")
        plt.title(rf"$t^+ = {avg_time:.2f}$")
        plt.legend(loc='lower right', ncol = 2, fontsize=12)
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    

    def build_urmsf_frame(self, frames, yplus_odt, yplus_ref, urmsf_odt, urmsf_ref, avg_time, ylim=None):
        
        fig = self.build_urmsf_fig(yplus_odt, yplus_ref, urmsf_odt, urmsf_ref, avg_time, ylim)
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
        plt.legend(loc='upper left')
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

    def build_anisotropy_tensor_barycentric_xmap_triang(self, y_delta, bar_map_x, bar_map_y, avg_time, title):
        
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
        plt.title(rf"$t^+ = {avg_time:.2f}$")
        
        # save figure
        filename = os.path.join(self.postRlzDir, f"{title}")
        print(f"\nMAKING PLOT OF BARYCENTRIC MAP OF ANISOTROPY TENSOR in {filename}" )
        plt.savefig(filename)


    def build_anisotropy_tensor_barycentric_xmap_triang_frame(self, frames, y_delta, bar_map_x, bar_map_y, avg_time):

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
        plt.title(rf"$t^+ = {avg_time:.2f}$")
        ###plt.clim( 0.0, 20.0 )

        # ------ save figure ------
        #filename = os.path.join(self.postRlzDir, f"anisotropy_tensor_barycentric_map_odt_avgTime_{avg_time:.0f}")
        #print(f"\nMAKING PLOT OF BARYCENTRIC MAP OF ANISOTROPY TENSOR from ODT data at Averaging Time = {avg_time:.2f}, in filename: {filename}" )
        #plt.savefig(filename)

        # ------ gif frame by pillow ---------
        # Save the current figure as an image frame
        fig = plt.gcf()
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()

        return frames


# ------------------- anisotropy tensor barycentric map coordinates 'xmap_i' -------------------
    
    def build_reynolds_stress_tensor_trace_fig(self, ydelta_odt, ydelta_ref, Rkk_odt, Rkk_ref, avg_time):
        
        fig, ax = plt.subplots()
        plt.plot(ydelta_odt, Rkk_odt, '-k', label=r"$R_{kk}$")
        plt.plot(ydelta_ref, Rkk_ref, '--k', label=r"Reference $R_{kk}$")
        plt.xlim([0.0, 1.0])
        plt.xlabel(r"$y/\delta$")
        plt.ylabel(r"Reynolds Stress Trace $R_{kk}$")
        plt.grid(axis="y")
        plt.title(rf"$t^+ = {avg_time:.2f}$")
        plt.legend(loc='upper right', ncol = 2, fontsize=12)
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    

    def build_reynolds_stress_tensor_trace(self, ydelta_odt, ydelta_ref, Rkk_odt, Rkk_ref, avg_time):
        
        filename = os.path.join(self.postRlzDir, "reynolds_stress_tensor_trace")
        print(f"\nMAKING PLOT OF Reynolds Stress tensor Trace at tavg = {avg_time} in {filename}")
        fig = self.build_reynolds_stress_tensor_trace_fig(ydelta_odt, ydelta_ref, Rkk_odt, Rkk_ref, avg_time)
        fig.savefig(filename)
        plt.close()


    def build_reynolds_stress_tensor_trace_frame(self, frames, ydelta_odt, ydelta_ref, Rkk_odt, Rkk_ref, avg_time):
        
        fig = self.build_reynolds_stress_tensor_trace_fig(ydelta_odt, ydelta_ref, Rkk_odt, Rkk_ref, avg_time)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

# ------------------- anisotropy tensor barycentric map coordinates 'xmap_i' -------------------

    def build_anisotropy_tensor_barycentric_xmap_coord_fig(self, ydelta_odt, ydelta_ref, xmap1_odt, xmap2_odt, xmap1_ref, xmap2_ref, avg_time):
        
        fig, ax = plt.subplots()
        plt.plot(ydelta_odt, xmap1_odt, '-k', label=r"$x_1$")
        plt.plot(ydelta_odt, xmap2_odt, '-b', label=r"$x_2$")
        plt.plot(ydelta_ref, xmap1_ref, '--k', label=r"Reference $x_1$")
        plt.plot(ydelta_ref, xmap2_ref, '--b', label=r"Reference $x_2$")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(r"$y/\delta$")
        plt.ylabel(r"barycentric coordinates $x_i$")
        plt.grid(axis="y")
        plt.title(rf"$t^+ = {avg_time:.2f}$")
        plt.legend(loc='upper right', ncol = 2, fontsize=12)
        plt.tight_layout()
        #fig = plt.gcf()
        return fig

    def build_anisotropy_tensor_barycentric_xmap_coord(self, ydelta_odt, ydelta_ref, xmap1_odt, xmap2_odt, xmap1_ref, xmap2_ref, avg_time):
        filename = os.path.join(self.postRlzDir, "anisotropy_tensor_barycentric_map_coord")
        print(f"\nMAKING PLOT OF Anisotropy Tensor Barycentric Coordinates at tavg = {avg_time} in {filename}")
        fig = self.build_anisotropy_tensor_barycentric_xmap_coord_fig(ydelta_odt, ydelta_ref, xmap1_odt, xmap2_odt, xmap1_ref, xmap2_ref, avg_time)
        fig.savefig(filename)
        plt.close()


    def build_anisotropy_tensor_barycentric_xmap_coord_frame(self, frames, ydelta_odt, ydelta_ref, xmap1_odt, xmap2_odt, xmap1_ref, xmap2_ref, avg_time):
        
        fig = self.build_anisotropy_tensor_barycentric_xmap_coord_fig(ydelta_odt, ydelta_ref, xmap1_odt, xmap2_odt, xmap1_ref, xmap2_ref, avg_time)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames


# ------------------- anisotropy tensor eigenvalues 'lambda_i' -------------------

    def build_anisotropy_tensor_eigenvalues_fig(self, ydelta_odt, ydelta_ref, eigenvalues_odt, eigenvalues_ref, avg_time):
        
        fig, ax = plt.subplots()
        plt.plot(ydelta_odt, eigenvalues_odt[:,0], '-k', label=r"$\lambda_0$")
        plt.plot(ydelta_odt, eigenvalues_odt[:,1], '-b', label=r"$\lambda_1$")
        plt.plot(ydelta_odt, eigenvalues_odt[:,2], '-r', label=r"$\lambda_2$")
        plt.plot(ydelta_ref, eigenvalues_ref[:,0], '--k', label=r"Reference $\lambda_0$")
        plt.plot(ydelta_ref, eigenvalues_ref[:,1], '--b', label=r"Reference $\lambda_1$")
        plt.plot(ydelta_ref, eigenvalues_ref[:,2], '--r', label=r"Reference $\lambda_2$")
        plt.xlim([0.0, 1.0])
        plt.ylim([-0.5, 1.0])
        plt.yticks([-2/3, -1/3, 0, 1/3, 2/3], labels = ["-2/3", "-1/3", "0", "1/3", "2/3"])
        plt.xlabel(r"$y/\delta$")
        plt.ylabel(r"anisotropy tensor eigenvalues $\lambda_i$")
        plt.grid(axis="y")
        plt.title(rf"$t^+ = {avg_time:.2f}$")
        plt.legend(loc='upper right', ncol = 2, fontsize=12)
        plt.tight_layout()
        #fig = plt.gcf()
        return fig
    

    def build_anisotropy_tensor_eigenvalues(self, ydelta_odt, ydelta_ref, eigenvalues_odt, eigenvalues_ref, avg_time):

        filename = os.path.join(self.postRlzDir, "anisotropy_tensor_eigenvalues")
        print(f"\nMAKING PLOT OF Anisotropy Tensor Eigenvalues at tavg = {avg_time} in {filename}")
        fig = self.build_anisotropy_tensor_eigenvalues_fig(ydelta_odt, ydelta_ref, eigenvalues_odt, eigenvalues_ref, avg_time)
        fig.savefig(filename)
        plt.close()
    
    def build_anisotropy_tensor_eigenvalues_frame(self, frames, ydelta_odt, ydelta_ref, eigenvalues_odt, eigenvalues_ref, avg_time):
        
        fig = self.build_anisotropy_tensor_eigenvalues_fig(ydelta_odt, ydelta_ref, eigenvalues_odt, eigenvalues_ref, avg_time)
        fig.canvas.draw()
        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(img)
        plt.close()
        return frames

# ------------------------------------------------------------------------

    def build_main_gifs_from_frames(self, frames_avg_u, frames_rmsf_u, frames_rkk, frames_eig, frames_xmap_coord, frames_xmap_triang):
      
        filename = os.path.join(self.postRlzDir, "u_mean_global_steps.gif")
        print(f"\nMAKING GIF U-MEAN for RUNTIME calculations along TRAINING GLOBAL STEPS in {filename}" )
        frames_avg_u[0].save(filename, save_all=True, append_images=frames_avg_u[1:], duration=100, loop=0)    

        filename = os.path.join(self.postRlzDir, "u_rmsf_global_steps.gif")
        print(f"\nMAKING GIF U-RMSF for RUNTIME calculations along TRAINING GLOBAL STEPS {filename}" )
        frames_rmsf_u[0].save(filename, save_all=True, append_images=frames_rmsf_u[1:], duration=100, loop=0)    

        filename = os.path.join(self.postRlzDir, "reynolds_stress_tensor_trace_global_steps.gif")
        print(f"\nMAKING GIF TRACE/MAGNITUDE OF REYNOLDS STRESS TENSOR for RUNTIME calculations along TRAINING GLOBAL STEPS {filename}" )
        frames_rkk[0].save(filename, save_all=True, append_images=frames_rkk[1:], duration=100, loop=0)    

        filename = os.path.join(self.postRlzDir, "anisotropy_tensor_eigenvalues_global_steps.gif")
        print(f"\nMAKING GIF EIGENVALUES OF ANISOTROPY TENSOR for RUNTIME calculations along TRAINING GLOBAL STEPS {filename}" )
        frames_eig[0].save(filename, save_all=True, append_images=frames_eig[1:], duration=100, loop=0)

        filename = os.path.join(self.postRlzDir, "anisotropy_tensor_barycentric_map_coord_global_steps.gif")
        print(f"\nMAKING GIF OF BARYCENTRIC MAP COORDINATES OF ANISOTROPY TENSOR for RUNTIME calculations along TRAINING GLOBAL STEPS {filename}" )
        frames_xmap_coord[0].save(filename, save_all=True, append_images=frames_xmap_coord[1:], duration=100, loop=0)

        filename = os.path.join(self.postRlzDir, "anisotropy_tensor_barycentric_map_triang_global_steps.gif")
        print(f"\nMAKING GIF OF BARYCENTRIC MAP REALIZABLE TRIANGLE OF ANISOTROPY TENSOR for RUNTIME calculations along TRAINING GLOBAL STEPS {filename}" )
        frames_xmap_triang[0].save(filename, save_all=True, append_images=frames_xmap_triang[1:], duration=100, loop=0)

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

        plt.savefig(filename)
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
        plt.savefig(filename)
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
        plt.savefig(filename)
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
        
        filename = os.path.join(self.postRlzDir, filename)
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
        plt.savefig(filename)
        plt.close()

    def RL_variable_convergence_along_time(self, filename, ylabel, rlzArr, timeArr, var_RL):
        
        filename = os.path.join(self.postRlzDir, filename)
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
        plt.savefig(filename)
        plt.close()

    
    def RL_variable_convergence_along_time_and_pdf(self, filename, ylabel, rlzArr, timeArr, var_RL):
        filename = os.path.join(self.postRlzDir, filename)
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
        plt.savefig(filename)
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
        filename = os.path.join(self.postRlzDir, f"RL_error_{error_name}_convergence")
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
            plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    def RL_err_convergence_along_time(self, rlzArr, err_RL, err_ref, averaging_times_RL, averaging_times_ref, info, tEndAvgRef=200):
        
        filename = os.path.join(self.postRlzDir, f"RL_error_{info['title']}_temporal_convergence")
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
            plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
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

        filename = os.path.join(self.postRlzDir, "RL_rewards")
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
        plt.savefig(filename)
        plt.close()


    def build_RL_rewards_convergence_nohup_v2(self, rewards_total, RL_rewards_term_relL2Err_umean, RL_rewards_term_relL2Err_urmsf, rewards_term_rhsfRatio, inputRL_filepath=None):
        # Plot RL rewards along simulation steps from nohup information

        if inputRL_filepath is None:
            nax     = 4
        else:
            nax     = 5
        fig, ax = plt.subplots(1,nax,figsize=(5*nax,5))

        filename = os.path.join(self.postRlzDir, "RL_rewards")
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
        plt.savefig(filename)
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
            filename = os.path.join(self.postRlzDir, f"RL_actions_convergence_{dofNames[iActDof]}")
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
            plt.savefig(filename)
            plt.close()