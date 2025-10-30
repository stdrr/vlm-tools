import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_poincare(z, ax=None, title="Poincaré Embeddings", savepath=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.set_title(title)
    ax.set_xlim(-0.30, 0.30)
    ax.set_ylim(-0.30, 0.30)
    ax.set_aspect('equal')
    circle = plt.Circle((0, 0), 0.30, color='lightgray', alpha=0.5, fill=True)
    ax.add_artist(circle)
    colors = ['blue', 'red', 'orange', 'green']
    for i, group in enumerate(z):
        sns.scatterplot(x=z[group][:, 0], y=z[group][:, 1], ax=ax, s=5, label=group, color=colors[i])
    ax.scatter(x=0, y=0, s=50, marker='x', color='black')
    plt.text(0.02, 0.02, 'O', fontsize=9)
    ax.set_axis_off()
    if savepath:
        plt.savefig(savepath)
    return fig, ax


if __name__ == "__main__":
    # Example usage
    embeddings_path = "data/embeddings/grit_horopca_2d.pkl"
    import pickle
    with open(embeddings_path, "rb") as f:
        z = pickle.load(f)
    plot_poincare(z, title="Poincaré Embeddings", savepath="poincare_plot.png")
    plt.show()

# import numpy as np
# from sklearn.decomposition import PCA
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle


# def lorentz_inner(x, y):
#     """Lorentzian inner product in R^(d+1) with signature (-,+,+,...)"""
#     return -x[0] * y[0] + np.dot(x[1:], y[1:])


# def horo_projection(X, basepoint=None):
#     """
#     Project Lorentz model embeddings into horospherical space.

#     Args:
#         X: np.ndarray of shape (n_samples, d+1), embeddings on Lorentz model
#         basepoint: np.ndarray of shape (d+1,), reference point at infinity
#                    If None, use the standard choice (1, 1, 0, ..., 0)

#     Returns:
#         np.ndarray of shape (n_samples, d), Euclideanized embeddings
#     """
#     n, d_plus1 = X.shape

#     if basepoint is None:
#         basepoint = np.zeros(d_plus1)
#         basepoint[0] = 1.0
#         basepoint[1] = 1.0

#     # Check normalization
#     if not np.isclose(lorentz_inner(basepoint, basepoint), 0.0):
#         raise ValueError("Basepoint must be lightlike (null vector)")

#     # Project to horosphere: <x, u> = -1 condition
#     proj = np.zeros((n, d_plus1 - 1))
#     for i in range(n):
#         xi = X[i]
#         scale = -1.0 / lorentz_inner(xi, basepoint)
#         proj[i] = scale * xi[1:]  # discard time coordinate
#     return proj


# def horo_pca(X, n_components=2):
#     """
#     Perform HoroPCA dimensionality reduction.
    
#     Args:
#         X: np.ndarray of shape (n_samples, d+1), Lorentz embeddings
#         n_components: target dimension

#     Returns:
#         np.ndarray of shape (n_samples, n_components)
#     """
#     X_horo = horo_projection(X)
#     pca = PCA(n_components=n_components)
#     return pca.fit_transform(X_horo)


# # Example usage
# if __name__ == "__main__":
#     # Load Lorentz hyperbolic embeddings
#     X = np.load("data/embeddings/grit_lorentz.npy")

#     n_samples = X.shape[0] // 4

#     # Reduce with HoroPCA
#     X_2d = horo_pca(X, n_components=2)

#     # Plot with seaborn
#     sns.set_theme(style="white", context="notebook")
#     fig, ax = plt.subplots(figsize=(6, 6))
#     # ax.set_title("Hyperbolic embeddings reduced with HoroPCA")
#     # ax.set_xlim(-1, 1)
#     # ax.set_ylim(-1, 1)
#     ax.set_aspect('equal')
#     # circle = plt.Circle((0, 0), 1, color='black', fill=False)
#     # ax.add_artist(circle)
#     sns.scatterplot(x=X_2d[:n_samples, 0], y=X_2d[:n_samples, 1], s=10, alpha=0.7, edgecolor=None, label="box_text_feats", color='red')
#     sns.scatterplot(x=X_2d[n_samples:2*n_samples, 0], y=X_2d[n_samples:2*n_samples, 1], s=10, alpha=0.7, edgecolor=None, label="text_feats", color='blue')
#     sns.scatterplot(x=X_2d[2*n_samples:3*n_samples, 0], y=X_2d[2*n_samples:3*n_samples, 1], s=10, alpha=0.7, edgecolor=None, label="box_image_feats", color='green')
#     sns.scatterplot(x=X_2d[3*n_samples:4*n_samples, 0], y=X_2d[3*n_samples:4*n_samples, 1], s=10, alpha=0.7, edgecolor=None, label="image_feats", color='purple')
#     plt.legend()
#     plt.title("Hyperbolic embeddings reduced with HoroPCA")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.savefig("horo_pca_plot.png")