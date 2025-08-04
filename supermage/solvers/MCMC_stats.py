import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.patches import Ellipse
from scipy.stats import norm

def corner_plot(
    chain,
    labels=None,
    figsize=(10, 10),
    true_values=None,
    ellipse_colors="g",
):
    num_params = chain.shape[1]
    cov_matrix = np.cov(chain, rowvar=False)
    mean = np.mean(chain, axis=0)
    fig, axes = plt.subplots(num_params, num_params, figsize=figsize)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.style.use('ggplot')

    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i, j]

            if i == j:
                x = np.linspace(
                    mean[i] - 3 * np.sqrt(cov_matrix[i, i]),
                    mean[i] + 3 * np.sqrt(cov_matrix[i, i]),
                    100,
                )
                y = norm.pdf(x, mean[i], np.sqrt(cov_matrix[i, i]))
                ax.plot(x, y, color="g")
                ax.set_xlim(
                    mean[i] - 3 * np.sqrt(cov_matrix[i, i]),
                    mean[i] + 3 * np.sqrt(cov_matrix[i, i]),
                )
                if true_values is not None:
                    ax.axvline(true_values[i], color="red", linestyle="-", lw=1)
            elif j < i:
                ax.scatter(chain[:, j], chain[:, i], color="c", s=0.1, zorder=0)
                cov = cov_matrix[np.ix_([j, i], [j, i])]
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                angle = np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))
                for k in [1, 2]:
                    ellipse = Ellipse(
                        xy=(mean[j], mean[i]),
                        width=lambda_[0] * k * 2,
                        height=lambda_[1] * k * 2,
                        angle=angle,
                        edgecolor=ellipse_colors,
                        facecolor="none",
                    )
                    ax.add_artist(ellipse)

                # Set axis limits
                margin = 3
                ax.set_xlim(
                    mean[j] - margin * np.sqrt(cov_matrix[j, j]),
                    mean[j] + margin * np.sqrt(cov_matrix[j, j]),
                )
                ax.set_ylim(
                    mean[i] - margin * np.sqrt(cov_matrix[i, i]),
                    mean[i] + margin * np.sqrt(cov_matrix[i, i]),
                )

                if true_values is not None:
                    ax.axvline(true_values[j], color="red", linestyle="-", lw=1)
                    ax.axhline(true_values[i], color="red", linestyle="-", lw=1)

            if j > i:
                ax.axis("off")

            if i < num_params - 1:
                ax.set_xticklabels([])
            else:
                if labels is not None:
                    ax.set_xlabel(labels[j])
            ax.yaxis.set_major_locator(plt.NullLocator())

            if j > 0:
                ax.set_yticklabels([])
            else:
                if labels is not None:
                    ax.set_ylabel(labels[i])
            ax.xaxis.set_major_locator(plt.NullLocator())

    plt.show()