import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import sklearn.metrics
import torch
from collections import Counter


def calculate_track_statistics(model, X, Y):
    model.eval()
    trajectory_predictions = model.predict_trajectories(X, medianfilter=True)

    statistics = []

    def idx_to_class(i):
        return ["unlabelleled", "nothing", "following", "waggle"][int(i) + 1]

    label_middle_index = Y.shape[1] // 2
    for idx in range(X.shape[0]):
        gt_index = int(Y[idx, label_middle_index])
        gt_class = idx_to_class(gt_index)
        pred = trajectory_predictions[idx]
        assert pred.shape[0] == 3
        pred = np.argmax(pred, axis=0)
        assert pred.shape[0] > 3

        prediction_middle_index = pred.shape[0] // 2
        predicted_index = int(pred[prediction_middle_index])
        predicted_class = idx_to_class(predicted_index)

        statistics.append(
            dict(
                gt_class=gt_class,
                predicted_class=predicted_class,
                gt_index=gt_index,
                predicted_index=predicted_index,
            )
        )

    statistics = pandas.DataFrame(statistics)

    return statistics


def evaluate_statistics(statistics, heatmap_kws=dict()):

    print("N samples:               {}.".format(statistics.shape[0]))
    print(
        "GT classes:              {}".format(
            sorted(Counter(statistics.gt_class).items())
        )
    )
    print(
        "Predicted classes:       {}".format(
            sorted(Counter(statistics.predicted_class).items())
        )
    )

    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(
        statistics.gt_class.values, statistics.predicted_class.values, adjusted=True
    )
    print("Balanced accuracy score: {:3.2%}".format(balanced_accuracy))

    confusion_matrix = sklearn.metrics.confusion_matrix(
        statistics.gt_class.values, statistics.predicted_class.values, normalize="true"
    )
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.heatmap(
        confusion_matrix,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=["nothing", "following", "waggle"],
        yticklabels=["nothing", "following", "waggle"],
        **heatmap_kws
    )
    ax.set_aspect("equal")
    plt.show()
