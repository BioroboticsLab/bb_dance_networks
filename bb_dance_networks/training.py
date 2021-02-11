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
    full_trajectory_lr_factor=0.1,
    statistics=None,
    optimizer=None,
):

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    if statistics is None:
        from collections import defaultdict

        statistics = defaultdict(list)

    n_pretrained_batches = 0
    if len(statistics) > 0:
        n_pretrained_batches = max((len(v) for k, v in statistics.items()))

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

            L += full_trajectory_lr_factor * model.train_batch(
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
                pred = model.predict_trajectories(
                    datareader_full_trajectories.test_X, return_logits=True
                )
                margin = (full_trajectories_labels_test.shape[1] - pred.shape[2]) // 2
                ground_truth = full_trajectories_labels_test[:, margin:-(margin)]

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

                mid_idx = ground_truth.shape[1] // 2
                gt_classes = ground_truth[:, mid_idx]
                pred_classes = pred[:, :, mid_idx]

                middle_idx_acc = sklearn.metrics.balanced_accuracy_score(
                    gt_classes.flatten(),
                    np.argmax(pred_classes, axis=1).flatten(),
                    adjusted=True,
                )

                middle_idx_acc2 = sklearn.metrics.balanced_accuracy_score(
                    datareader_full_trajectories.test_Y.flatten(),
                    np.argmax(pred_classes, axis=1).flatten(),
                    adjusted=True,
                )

                all_timesteps_acc = sklearn.metrics.balanced_accuracy_score(
                    ground_truth.flatten(),
                    np.argmax(pred, axis=1).flatten(),
                    adjusted=True,
                )
                statistics["test_accuracy_all_timesteps"].append(
                    (n_pretrained_batches + batch_index, all_timesteps_acc)
                )

                statistics["test_accuracy_center_timestep"].append(
                    (n_pretrained_batches + batch_index, middle_idx_acc)
                )

                statistics["test_accuracy_gt_label"].append(
                    (n_pretrained_batches + batch_index, middle_idx_acc2)
                )

                statistics["test_crossentropy"].append(
                    (n_pretrained_batches + batch_index, ce_loss)
                )
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

    fig, ax = plt.subplots(figsize=(10, 4))

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
