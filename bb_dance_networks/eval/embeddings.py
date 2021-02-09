import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import tqdm.auto
import openTSNE


def plot_embeddings(embeddings, Y):

    # Sample only a subset of all data points (include the positive labels, though).
    sample_indices = np.zeros(shape=Y.shape[0], dtype=np.bool)
    sample_indices[np.random.choice(np.arange(Y.shape[0]), size=10000)] = 1
    sample_indices[Y[:, 0] > 0] = 1

    proj = openTSNE.TSNE().fit(embeddings[sample_indices])
    x, y = proj[:, 0], proj[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))
    has_label = Y[sample_indices, 0] != -1
    sns.scatterplot(
        x[~has_label], y[~has_label], hue=0, palette="Greys", markers="x", alpha=0.1
    )
    sns.scatterplot(
        x[has_label],
        y[has_label],
        hue=Y[sample_indices, 0][has_label],
        palette="tab10",
        alpha=0.5,
    )
    ax.legend(
        ax.get_legend_handles_labels()[0],
        ["unlabelled", "other", "following", "dancing"],
        loc="best",
    )
    plt.show()

    return proj[:, :2]
