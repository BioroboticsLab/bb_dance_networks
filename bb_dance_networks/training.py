import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm.auto
import sklearn.metrics

from .preprocessing import FeatureNormalizer

import bb_behavior.trajectory.features


def load_all_training_data(root_path, verbose=False):

    with open(root_path + "/BB2019_train_groups.npz", "rb") as f:
        precalculated_train_groups = np.load(f)["arr_0"]

    datareader_full_trajectories = bb_behavior.trajectory.features.DataReader.load(
        root_path + "BB2019_longtraj_datareader.h5"
    )
    datareader_full_trajectories._groups = (
        datareader_full_trajectories.samples.timestamp.apply(lambda t: t // (60 * 60))
        .astype(np.int)
        .values[datareader_full_trajectories._valid_sample_indices]
    )

    datareader_full_trajectories.create_train_test_split(
        predefined_train_groups=precalculated_train_groups
    )
    if verbose:
        print(
            "Trajectory shape: {}, train set: {}, test set: {}".format(
                datareader_full_trajectories.X.shape,
                datareader_full_trajectories.train_X.shape,
                datareader_full_trajectories.test_X.shape,
            )
        )

    datareader_embeddings = bb_behavior.trajectory.features.DataReader.load(
        root_path + "BB2019_embd_datareader.h5"
    )
    datareader_embeddings._groups = (
        datareader_embeddings.samples.timestamp.apply(lambda t: t // (60 * 60))
        .astype(np.int)
        .values[datareader_embeddings._valid_sample_indices]
    )
    datareader_embeddings.create_train_test_split(
        predefined_train_groups=precalculated_train_groups
    )
    if verbose:
        print(
            "Small embeddings shape: {}, train set: {}, test set: {}".format(
                datareader_embeddings.X.shape,
                datareader_embeddings.train_X.shape,
                datareader_embeddings.test_X.shape,
            )
        )

    with open(root_path + "BB2019_longtraj_labels.npz", "rb") as f:
        full_trajectories_labels = np.load(f)["arr_0"]
    train_idx = np.isin(
        datareader_full_trajectories._groups, precalculated_train_groups
    )
    full_trajectories_labels = full_trajectories_labels[:, :-1]
    full_trajectories_labels_train = full_trajectories_labels[train_idx]
    full_trajectories_labels_test = full_trajectories_labels[~train_idx]
    if verbose:
        print(
            "Trajectory labels shape: {}, train set: {}, test set: {}".format(
                full_trajectories_labels.shape,
                full_trajectories_labels_train.shape,
                full_trajectories_labels_test.shape,
            )
        )

    feature_normalizer = FeatureNormalizer().fit(datareader_embeddings.train_X)
    datareader_embeddings.X[:] = feature_normalizer.transform(datareader_embeddings.X)[
        :
    ]
    datareader_embeddings.train_X[:] = feature_normalizer.transform(
        datareader_embeddings.train_X
    )[:]
    datareader_embeddings.test_X[:] = feature_normalizer.transform(
        datareader_embeddings.test_X
    )[:]

    datareader_full_trajectories.X[:] = feature_normalizer.transform(
        datareader_full_trajectories.X
    )[:]
    datareader_full_trajectories.train_X[:] = feature_normalizer.transform(
        datareader_full_trajectories.train_X
    )[:]
    datareader_full_trajectories.test_X[:] = feature_normalizer.transform(
        datareader_full_trajectories.test_X
    )[:]

    return (
        datareader_embeddings,
        datareader_full_trajectories,
        full_trajectories_labels,
        full_trajectories_labels_train,
        full_trajectories_labels_test,
        feature_normalizer,
    )


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
    lr_scheduler=None,
    next_warm_restart=100,
):

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if statistics is None:
        from collections import defaultdict

        statistics = defaultdict(list)

    n_pretrained_batches = 0
    if len(statistics) > 0:
        n_pretrained_batches = max((len(v) for k, v in statistics.items()))

    trange = tqdm.auto.tqdm(total=n_batches)
    model.train()

    try:
        for batch_index in range(
            n_pretrained_batches, n_pretrained_batches + (n_batches * 2)
        ):

            if (batch_index == next_warm_restart) or (lr_scheduler is None):

                # Break at lowest LR point.
                if batch_index >= (n_pretrained_batches + n_batches):
                    break

                next_phaseout = 100
                if lr_scheduler is not None:
                    next_phaseout = 2 * lr_scheduler.T_max
                next_warm_restart = batch_index + next_phaseout
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, next_phaseout
                )
            statistics["learning_rate"].append(
                (batch_index, lr_scheduler.get_last_lr())
            )

            model.zero_grad()
            L = model.embedding.train_batch(
                datareader_embeddings.train_X,
                datareader_embeddings.train_Y,
                batch_index,
                batch_statistics_dict=statistics,
            )

            if batch_index % 20 == 0:
                L += full_trajectory_lr_factor * model.train_batch(
                    datareader_full_trajectories.train_X,
                    full_trajectories_labels_train,
                    batch_index,
                    batch_statistics_dict=statistics,
                )
            L.backward()
            optimizer.step()
            lr_scheduler.step()

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

                all_timesteps_f1 = sklearn.metrics.f1_score(
                    ground_truth.flatten(),
                    np.argmax(pred, axis=1).flatten(),
                    average="macro",
                )

                all_timesteps_acc = sklearn.metrics.balanced_accuracy_score(
                    ground_truth.flatten(),
                    np.argmax(pred, axis=1).flatten(),
                    adjusted=True,
                )

                statistics["test_accuracy_all_timesteps"].append(
                    (batch_index, all_timesteps_acc)
                )

                statistics["test_accuracy_center_timestep"].append(
                    (batch_index, middle_idx_acc)
                )

                statistics["test_f1_score"].append((batch_index, all_timesteps_f1))

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

    return statistics, optimizer, lr_scheduler, next_warm_restart


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
        if len(values) == 0:
            continue
        if key in name_mapping:
            key = name_mapping[key]

        def fix_value(v):
            if type(v) is list:
                v = v[0]
            return float(v)

        if type(values[0]) is tuple:
            x, y = zip(*values)
            y = [fix_value(_y) for _y in y]
            ax.plot(x, y, label=key)
        else:
            try:
                values = [fix_value(v) for v in values]
                ax.plot(moving_average(values), label=key)
            except Exception as e:
                print((key, str(e)))
    ax.legend()
    plt.show()
