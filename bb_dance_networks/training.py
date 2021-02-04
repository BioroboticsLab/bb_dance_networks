import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm.auto


def run_train_loop(
    model,
    datareader_embeddings,
    datareader_full_trajectories,
    full_trajectories_labels,
    n_batches=100000,
    statistics=None,
    optimizer=None,
):

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    if statistics is None:
        from collections import defaultdict

        statistics = defaultdict(list)

    trange = tqdm.auto.tqdm(total=n_batches)
    model.train()

    try:
        for batch_index in range(n_batches):

            model.zero_grad()
            L = model.embedding.train_batch(
                datareader_embeddings, batch_index, batch_statistics_dict=statistics
            )
            L += model.train_batch(
                datareader_full_trajectories,
                full_trajectories_labels,
                batch_index,
                batch_statistics_dict=statistics,
            )
            L.backward()
            optimizer.step()

            postfix_dict = dict()
            for loss_title, values in statistics.items():
                fun = np.mean
                if loss_title == "support_loss":
                    fun = np.nanmean
                postfix_dict[loss_title] = fun(values[-1000:])

            trange.set_postfix(postfix_dict)
            trange.update()
    except KeyboardInterrupt:
        pass
    finally:
        torch.cuda.empty_cache()

    return statistics, optimizer


def plot_training_losses(statistics):
    name_mapping = dict(
        reg="Regularization",
        prediction_loss="Embedding pred. loss",
        comparison_loss="Embedding comp. loss",
        support_loss="Embedding support loss",
        traj_prediction_loss="Trajectory pred. loss",
    )

    fig, ax = plt.subplots(figsize=(20, 5))

    def moving_average(a, n=1000):
        ret = np.nancumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    for key, values in statistics.items():
        if key in name_mapping:
            key = name_mapping[key]
        ax.plot(moving_average(values), label=key)
    ax.legend()
    plt.show()
