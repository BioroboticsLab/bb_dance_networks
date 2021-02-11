import torch
import torch.nn
import torch.optim
import numpy as np

import bb_behavior.utils.model_selection
import scipy.ndimage


class TrajectorySeq2Seq(torch.nn.Module):
    def __init__(self, embedding_size, sequence_size, n_classes):
        super().__init__()

        self.embedding_size = embedding_size
        self.sequence_size = sequence_size
        self.n_classes = n_classes

        layers = [
            torch.nn.Conv1d(embedding_size, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(16),
            torch.nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, n_classes, kernel_size=3, stride=1, padding=1),
        ]

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TrajectoryClassificationModel(torch.nn.Module):
    def __init__(
        self,
        feature_channels,
        embedding_input_length,
        n_classes,
        trajectory_length,
        embedding_model_class,
    ):
        super().__init__()

        self.loss_function = None

        self.embedding = embedding_model_class(
            channels=feature_channels,
            n_classes=n_classes,
            temporal_dimension=embedding_input_length,
        )
        self.traj2traj = TrajectorySeq2Seq(
            embedding_size=self.embedding.embedding_size,
            sequence_size=trajectory_length,
            n_classes=n_classes,
        )

    def forward(self, x):
        e = self.embedding.full_embedding(x)
        e = self.traj2traj(e)
        return e

    def embed(self, x):
        return self.embedding.embed(x)

    def get_support(self, x):
        return self.embedding.get_support(x)

    def compare(self, a, b):
        return self.embedding.compare(a, b)

    def predict_embedding(self, e):
        return self.embedding.predict_embedding(e)

    def predict_class_logits(self, e):
        return self.embedding.predict_class_logits(e)

    def train_batch(
        self,
        X,
        Y,
        batch_number,
        batch_statistics_dict,
        initial_batch_size=64,
        device="cuda",
    ):
        if self.loss_function is None:
            self.loss_function = torch.nn.CrossEntropyLoss(
                weight=torch.Tensor([0.01, 1.0, 1.0]).to(torch.device(device))
            )

        sample_indices = np.arange(0, X.shape[0], dtype=np.int)
        batch_size = initial_batch_size + batch_number // 500

        drawn_sample_indices = np.random.choice(
            sample_indices, size=batch_size, replace=False
        )
        samples = X[drawn_sample_indices]
        batch_X = torch.from_numpy(samples).cuda()

        prediction = self(batch_X)

        cut_off_margin = (Y.shape[1] - prediction.shape[2]) // 2
        labels = Y[drawn_sample_indices, cut_off_margin:-(cut_off_margin)]
        assert labels.shape[1] == prediction.shape[2]
        labels = torch.from_numpy(labels.astype(np.int)).cuda()

        prediction = prediction.permute(0, 2, 1)
        prediction = prediction.reshape(batch_size * prediction.shape[1], 3)

        labels = labels.view(batch_size * labels.shape[1])
        loss = self.loss_function(prediction, labels)

        batch_statistics_dict["traj_prediction_loss"].append(loss.data.cpu().numpy())
        return loss

    def predict_trajectories(self, X, return_logits=False, medianfilter=False):
        self.eval()
        trajectory_predictions = []
        for batch in bb_behavior.utils.model_selection.iterate_minibatches(
            X, None, 2048, include_small_last_batch=True
        ):
            pred = self(torch.from_numpy(batch).cuda())

            if not return_logits:
                pred = torch.nn.functional.softmax(pred, dim=1)
            pred = pred.detach().cpu().numpy()

            if medianfilter:
                pred = scipy.ndimage.median_filter(pred, size=(1, 1, 5))

            trajectory_predictions.append(pred)

        trajectory_predictions = np.concatenate(trajectory_predictions)

        return trajectory_predictions
