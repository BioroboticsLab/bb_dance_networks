import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm.auto
import sklearn.metrics


def run_train_loop(
    model,
    datareader_embeddings,
    datareader_full_trajectories,
    full_trajectories_labels_train,
    full_trajectories_labels_test,
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
                datareader_embeddings.train_X,
                datareader_embeddings.train_Y,
                batch_index,
                batch_statistics_dict=statistics,
            )
            L += model.train_batch(
                datareader_full_trajectories.train_X,
                full_trajectories_labels_train,
                batch_index,
                batch_statistics_dict=statistics,
            )
            L.backward()
            optimizer.step()

            if batch_index % 200 == 0:
                model.eval()
                test_loss = torch.nn.CrossEntropyLoss()
                pred = model.predict_trajectories(datareader_full_trajectories.test_X)
                margin = (full_trajectories_labels_test.shape[1] - pred.shape[2]) // 2
                ground_truth = full_trajectories_labels_test[:, margin:-margin]

                ce_loss = np.nan
                if test_loss is not None:
                    prediction = torch.from_numpy(pred).cuda().permute(0, 2, 1)
                    prediction = prediction.reshape(
                        ground_truth.shape[0] * prediction.shape[1], 3
                    )
                    labels = (
                        torch.from_numpy(ground_truth.astype(np.int)).cuda().reshape(-1)
                    )
                    ce_loss = test_loss(prediction, labels).data.cpu().numpy()

                acc = sklearn.metrics.balanced_accuracy_score(
                    ground_truth.flatten(), np.argmax(pred, axis=1).flatten()
                )
                statistics["test_accuracy"].append((batch_index, acc))
                statistics["test_crossentropy"].append((batch_index, ce_loss))
                model.train()

            postfix_dict = dict()
            for loss_title, values in statistics.items():
                fun = np.mean
                if loss_title == "support_loss":
                    fun = np.nanmean
                if type(values[0]) is tuple:
                    _, values = zip(*values)
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
        if type(values[0]) is tuple:
            x, y = zip(*values)
            ax.plot(x, y, label=key)
        else:
            ax.plot(moving_average(values), label=key)
    ax.legend()
    plt.show()
