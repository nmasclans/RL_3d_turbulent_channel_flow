# RL_post_process_control_law_visualization.py
import os
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 12, frameon = False)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')
#plt.rc( 'savefig',    format = "jpg", dpi = 600)

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("ControlLawVisualization")
logger.setLevel(logging.DEBUG)

# -------------------------------------------------------------------------------------------------

def cluster_control_analysis(X, U, n_clusters=10, random_state=42, agent_id=None):

    # Step 1: Flatten input for clustering
    n_steps, n_agents, state_dim = X.shape
    n_steps_u, n_agents_u, action_dim = U.shape
    assert n_steps == n_steps_u and n_agents == n_agents_u, "Mismatch in X and U dimensions"

    # Cluster control analysis only for specific agent, if necessary
    if agent_id is not None:
        X = X[:,agent_id,:]
        U = U[:,agent_id,:]
        n_agents = 1

    # Reshape data
    X_flat = X.reshape(n_steps * n_agents, state_dim)
    U_flat = U.reshape(n_steps * n_agents, action_dim)

    # Step 2: KMeans Clustering on State Space
    logger.debug("Fit + Predict k-means clustering")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++')
    labels = kmeans.fit_predict(X_flat)
    centroids = kmeans.cluster_centers_

    # Step 3: Transition Matrix P
    logger.debug("Building transition matrix")
    P = np.zeros((n_clusters, n_clusters))
    labels_reshaped = labels.reshape(n_steps, n_agents)
    for t in range(1, n_steps):
        for a in range(n_agents):
            P[labels_reshaped[t - 1, a], labels_reshaped[t, a]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)

    # Step 4: Average Action & State per Cluster
    logger.debug("Calculating action & state statistics per cluster")
    # Step 4.1: Action statistics
    avg_u = np.zeros((n_clusters, action_dim))
    std_u = np.zeros((n_clusters, action_dim))
    min_u = np.zeros((n_clusters, action_dim))
    max_u = np.zeros((n_clusters, action_dim))
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        avg_u[i] = np.mean(U_flat[cluster_indices], axis=0)
        std_u[i] = np.std( U_flat[cluster_indices], axis=0)
        min_u[i] = np.min( U_flat[cluster_indices], axis=0)
        max_u[i] = np.max( U_flat[cluster_indices], axis=0)
    with open(f"control_law_visualization/data_control_action_statistics_{n_clusters}clusters.csv",'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster", "ActionDim", "Mean", "Std", "Min", "Max"])
        for i in range(n_clusters):
            for d in range(action_dim):
                writer.writerow([i, d, avg_u[i, d], std_u[i, d], min_u[i, d], max_u[i, d]])
    # Step 4.2: State statistics
    avg_x = np.zeros((n_clusters, state_dim))
    std_x = np.zeros((n_clusters, state_dim))
    min_x = np.zeros((n_clusters, state_dim))
    max_x = np.zeros((n_clusters, state_dim))
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        avg_x[i] = np.mean(X_flat[cluster_indices], axis=0)
        std_x[i] = np.std( X_flat[cluster_indices], axis=0)
        min_x[i] = np.min( X_flat[cluster_indices], axis=0)
        max_x[i] = np.max( X_flat[cluster_indices], axis=0)
    with open(f"control_law_visualization/data_control_state_statistics_{n_clusters}clusters.csv",'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster", "ActionDim", "Mean", "Std", "Min", "Max"])
        for i in range(n_clusters):
            for d in range(state_dim):
                writer.writerow([i, d, avg_x[i, d], std_x[i, d], min_x[i, d], max_x[i, d]])
    # Step 4.3: plot data from steps 4 & 4.2
    logger.debug("Building figures state & action statistics per cluster")
    plot_cluster_statistics(avg_x, std_x, min_x, max_x, avg_u, std_u, min_u, max_u, n_clusters)
    plot_cluster_stat_profiles(avg_x, std_x, min_x, max_x, avg_u, std_u, min_u, max_u, n_clusters)

    # Step 5: MDS Projection
    logger.debug("Projecting states into low-dimensional space (MDS)")
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=random_state)
    labels_proj  = labels[::scatter_step]
    X_sample     = X_flat[::scatter_step]
    n_sample     = X_sample.shape[0]
    X_all        = np.vstack([X_sample, centroids])
    X_proj_all   = mds.fit_transform(X_all)
    X_proj       = X_proj_all[:n_sample,:]
    centroids_2d = X_proj_all[n_sample:,:]

    # Step 6: Proximity map figure (MDS + Clusters centroid)
    logger.debug("Building proximity map")
    fig, ax = plt.subplots()
    ax.scatter(
            X_proj[:, 0], X_proj[:, 1], c=labels_proj, cmap='tab10', s=10, alpha=0.6, label='Data Points'
    )
    scatter = ax.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1], c=np.arange(n_clusters), cmap='tab10', s=300, edgecolors='k'
    )
    for i in range(n_clusters):
        ax.text(centroids_2d[i, 0], centroids_2d[i, 1], str(i), fontsize=12, ha='center', va='center', color='white')
    #ax.set_title('Proximity Map of Sensor Signals')
    ax.set_xlabel(r'$\gamma_1$')
    ax.set_ylabel(r'$\gamma_2$')
    plt.tight_layout()
    plt.savefig(f"control_law_visualization/fig_control_law_visualization_{n_clusters}clusters_proximity_map.jpg", dpi=600)
    plt.close()

    # Step 6.2: transition matrix figure
    logger.debug("Building transition matrix")
    fig, ax = plt.subplots()
    im = ax.imshow(P, cmap='viridis')
    #ax.set_title('Transition Matrix')
    ax.set_xlabel('To Cluster')
    ax.set_ylabel('From Cluster')
    # Write transition probability in each cell
    for i in range(n_clusters):
        for j in range(n_clusters):
            prob = P[i,j]
            text_color = "white" if prob < 0.5 * P.max() else "black"
            ax.text(j, i, f"{prob:.2f}", ha="center", va="center", color=text_color)
    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(n_clusters))
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"control_law_visualization/fig_control_law_visualization_{n_clusters}clusters_transition_matrix.jpg", dpi=600)
    plt.close()

    ### # Step 6.3: Control Network figure (Action Norms with Transitions)
    ### fig, ax = plt.subplots()
    ### scatter2 = ax.scatter(
    ###     centroids_2d[:, 0], centroids_2d[:, 1], c=np.linalg.norm(avg_u, axis=1), cmap='viridis', s=300, edgecolors='k'
    ### )
    ### for i in range(n_clusters):
    ###     ax.text(centroids_2d[i, 0], centroids_2d[i, 1], str(i), fontsize=12, ha='center', va='center', color='white')
    ### for i in range(n_clusters):
    ###     for j in range(n_clusters):
    ###         if P[i, j] > 0.05:
    ###             ax.annotate(
    ###                 '',
    ###                 xy=centroids_2d[j],
    ###                 xytext=centroids_2d[i],
    ###                 arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=P[i, j])
    ###             )
    ### for i in range(n_clusters):
    ###     stats_text = "\n".join([
    ###         rf"$\mu{i}: {avg_u[i, d]:.2f}, \sigma: {std_u[i, d]:.2f}$" 
    ###         for d in range(action_dim)
    ###     ])
    ###     ax.text(
    ###         centroids_2d[i, 0], centroids_2d[i, 1], f"{i}\n{stats_text}", fontsize=8,
    ###         ha='center', va='center', color='white', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')
    ###     )
    ### ax.set_title('Control Network')
    ### ax.set_xlabel('MDS Dimension 1')
    ### ax.set_ylabel('MDS Dimension 2')
    ### fig.colorbar(scatter2, ax=ax, label='Avg Actuation Norm per Cluster')
    ### plt.tight_layout()
    ### plt.savefig(f"control_law_visualization/fig_control_law_visualization_{n_clusters}clusters_control_network.jpg", _action_normdpi=600)
    ### plt.close()

    ### # Step 7: Proximity Map (MDS), different subplot for the data projections of each cluster
    ### logger.debug("Building proximity map per cluster")
    ### fig, axes = plt.subplots(nrows=(n_clusters+4)//5, ncols=5, figsize=(18, 3.5*((n_clusters + 4)//5)), sharex=True, sharey=True)
    ### axes = axes.flatten()
    ### xall = np.concatenate([X_proj[:,0], centroids_2d[:,0]])
    ### yall = np.concatenate([X_proj[:,1], centroids_2d[:,1]])
    ### xlim = (xall.min(), xall.max())
    ### ylim = (yall.min(), yall.max())
    ### for i in range(n_clusters):
    ###     ax = axes[i]
    ###     idx = (labels_proj==i)
    ###     ax.scatter(X_proj[idx,0], X_proj[idx,1], s=10, alpha=0.6, c=f"C{i}")
    ###     ax.scatter(*centroids_2d[i], s=100, c='white', edgecolors='k', linewidths=2, zorder=5)
    ###     ax.plot(*centroids_2d[i], marker='x', color='red', markersize=8, zorder=6)
    ###     ax.set_title(f"Cluster {i}")
    ###     ax.set_xlim(xlim)
    ###     ax.set_ylim(ylim)
    ### for i in range(n_clusters, len(axes)):
    ###     fig.delaxes(axes[i])  
    ### fig.suptitle("Cluster-wise Proximity Map (MDS)", fontsize=16)
    ### plt.tight_layout(rect=[0, 0, 1, 0.97])
    ### plt.savefig(f"control_law_visualization/fig_proximity_map_per_cluster_{n_clusters}clusters.jpg", dpi=600)
    ### plt.close()

    return labels, centroids, P, avg_u, centroids_2d

# -------------------------------------------------------------------------------------------------

def plot_cluster_statistics(avg_x, std_x, min_x, max_x, avg_u, std_u, min_u, max_u, n_clusters):
    stats = {
        "Mean":  (avg_x,  avg_u),
        "Std":   (std_x,  std_u),
        "Min":   (min_x,  min_u),
        "Max":   (max_x,  max_u)
    }

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    stat_names = list(stats.keys())

    for row, stat in enumerate(stat_names):
        x_stat, u_stat = stats[stat]

        sns.heatmap(x_stat.T, ax=axes[row][0], cmap="viridis", cbar=True)
        #axes[row][0].set_title(f"State {stat} per Cluster")
        axes[row][0].set_xlabel("Cluster")
        axes[row][0].set_ylabel("State Dimension")

        sns.heatmap(u_stat.T, ax=axes[row][1], cmap="viridis", cbar=True)
        #axes[row][1].set_title(f"Action {stat} per Cluster")
        axes[row][1].set_xlabel("Cluster")
        axes[row][1].set_ylabel("Action Dimension")

    plt.tight_layout()
    plt.savefig(f"control_law_visualization/fig_statistics_summary_{n_clusters}clusters.jpg", dpi=300)
    plt.close()

def plot_cluster_stat_profiles(avg_x, std_x, min_x, max_x, avg_u, std_u, min_u, max_u, n_clusters):
    state_dim = avg_x.shape[1]
    action_dim = avg_u.shape[1]
    width_d = 0.75
    width_c = 0.60 * width_d / n_clusters
    
    fig, ax = plt.subplots()
    for d in range(state_dim):          # state_dim = 9
        for i in range(n_clusters):     # n_clusters = 6
            x = d + (i * width_d / (n_clusters)) - width_d / 4
            # Min-Max gray bar
            ax.add_patch(plt.Rectangle((x - 0.5 * width_c, min_x[i, d]), width_c, max_x[i, d] - min_x[i, d], color='gray', alpha=0.3))
            # Std dev black bar
            ax.plot([x, x], [avg_x[i, d] - std_x[i, d], avg_x[i, d] + std_x[i, d]], color='black', linewidth=2)
            ax.plot([x - 0.5 * width_c, x + 0.5 * width_c], [avg_x[i, d] - std_x[i, d], avg_x[i, d] - std_x[i, d]], color='black', linewidth=2)
            ax.plot([x - 0.5 * width_c, x + 0.5 * width_c], [avg_x[i, d] + std_x[i, d], avg_x[i, d] + std_x[i, d]], color='black', linewidth=2)
            # Mean red line
            ax.plot([x - 0.5 * width_c, x + 0.5 * width_c], [avg_x[i, d], avg_x[i, d]], color='red', linewidth=2)
    #ax.set_title("State Statistics per Cluster")
    ax.set_xlabel("State Dimension")
    ax.set_ylabel("Value")
    ax.grid(axis='y')
    ax.set_xticks(np.arange(state_dim))
    ax.set_xlim(-0.5, state_dim - 0.5)
    plt.tight_layout()
    plt.savefig(f"control_law_visualization/fig_statistics_profiles_state_{n_clusters}clusters.jpg", dpi=300)
    plt.close()

    # Plot ACTION statistics as pseudo-violin style per dimension
    fig, ax = plt.subplots()
    for d in range(action_dim):
        for i in range(n_clusters):
            x = d + (i * width_d / (n_clusters)) - width_d / 4
            # Min-Max gray bar
            ax.add_patch(plt.Rectangle((x - 0.5 * width_c, min_u[i, d]), width_c, max_u[i, d] - min_u[i, d], color='gray', alpha=0.3))
            # Std dev black bar
            ax.plot([x, x], [avg_u[i, d] - std_u[i, d], avg_u[i, d] + std_u[i, d]], color='black', linewidth=2)
            ax.plot([x - 0.5 * width_c, x + 0.5 * width_c], [avg_u[i, d] - std_u[i, d], avg_u[i, d] - std_u[i, d]], color='black', linewidth=2)
            ax.plot([x - 0.5 * width_c, x + 0.5 * width_c], [avg_u[i, d] + std_u[i, d], avg_u[i, d] + std_u[i, d]], color='black', linewidth=2)
            # Mean red line
            ax.plot([x - 0.5 * width_c, x + 0.5 * width_c], [avg_u[i, d], avg_u[i, d]], color='red', linewidth=2)
    #ax.set_title("Action Statistics per Cluster")
    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Value")
    ax.grid(axis='y')
    ax.set_xticks(np.arange(action_dim))
    ax.set_xlim(-0.5, action_dim - 0.5)
    plt.tight_layout()
    plt.savefig(f"control_law_visualization/fig_statistics_profiles_action_{n_clusters}clusters.jpg", dpi=300)
    plt.close()

# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    ensemble   = "0"
    case_dir   = "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau100_S10_5tavg0_1max_9state_17"
    run_mode   = "train"
    train_name = "train_2025-09-04--09-48-06--0c85"
    rl_n_envs  = 160
    step       = "004320"
    scatter_step = 2
    
    time_data_dir   = f"{case_dir}/{run_mode}/{train_name}/time/"
    state_data_dir  = f"{case_dir}/{run_mode}/{train_name}/state/"
    action_data_dir = f"{case_dir}/{run_mode}/{train_name}/action/"
    time_filepath   = os.path.join(time_data_dir,   f"time_ensemble{ensemble}_step{step}.txt")
    state_filepath  = os.path.join(state_data_dir,  f"state_ensemble{ensemble}_step{step}.txt")
    action_filepath = os.path.join(action_data_dir, f"action_ensemble{ensemble}_step{step}.txt")
    
    time_data   = np.loadtxt(time_filepath)
    state_data  = np.loadtxt(state_filepath)
    action_data = np.loadtxt(action_filepath)
    
    num_time_steps = time_data.size
    state  = state_data.reshape( num_time_steps, rl_n_envs, -1)  # shape [num_time_steps, rl_n_envs, state_dim]
    action = action_data.reshape(num_time_steps, rl_n_envs, -1)  # shape [num_time_steps, rl_n_envs, action_dim]
    
    for n_clusters in np.linspace(2,20,19,dtype='int'):
        logger.info(f"Using {n_clusters} clusters")
        cluster_control_analysis(state, action, n_clusters=n_clusters, agent_id=None)