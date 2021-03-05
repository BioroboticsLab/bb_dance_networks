import torch
import torch.nn
import torch.optim
import numpy as np

import bb_behavior.utils.model_selection
import scipy.ndimage


class ActionEmbeddingModel(torch.nn.Module):
    def __init__(self, channels=5, temporal_dimension=6 * 2, n_classes=3):
        super().__init__()

        self.loss = None
        self.temporal_dimension = temporal_dimension
        self.embedding_size = 12

        layers = [
            # 12
            torch.nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
            # 12
            torch.nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(32),
            # 12
            torch.nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(32),
            # 8
            torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(64),
            # 6
            torch.nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(64),
            # 2
            torch.nn.Conv1d(
                64, self.embedding_size, kernel_size=2, stride=1, padding=0
            ),
            torch.nn.ELU(),
            # 1
            torch.nn.ReLU(),
        ]
        self.embedding = torch.nn.Sequential(*layers)

        self.prediction = torch.nn.Sequential(
            *[
                torch.nn.Linear(self.embedding_size, 64),
                torch.nn.ELU(),
                torch.nn.BatchNorm1d(64),
                torch.nn.Linear(64, 16),
                torch.nn.ELU(),
                torch.nn.BatchNorm1d(16),
                torch.nn.Linear(16, n_classes),
            ]
        )

        self.n_classes = n_classes

    def full_embedding(self, x):
        e = self.embedding(x)
        return e

    def embed(self, x):
        a, b = (
            x[:, :, : self.temporal_dimension],
            x[:, :, (x.shape[-1] - self.temporal_dimension) :],
        )
        e1, e2 = self.embedding(a), self.embedding(b)
        e1 = e1.view(x.shape[0], -1)
        e2 = e2.view(x.shape[0], -1)
        assert e1.shape[1] == self.embedding_size
        return e1, e2

    def forward(self, x):
        return self.full_embedding(x)

    def predict_class_logits(self, e):
        e = self.prediction(e)
        return e

    def train_batch(
        self,
        X,
        Y,
        batch_number,
        batch_statistics_dict,
        initial_batch_size=1024,
        device="cuda",
    ):
        sample_indices = np.arange(0, X.shape[0], dtype=np.int)
        batch_size = initial_batch_size

        if self.loss is None:
            self.loss = torch.nn.CrossEntropyLoss(
                weight=torch.Tensor([0.01, 1.0, 1.0]).to(torch.device(device))
            )

        samples = np.random.choice(sample_indices, size=batch_size, replace=False)
        labels = Y[samples, 0]
        samples = X[samples]

        batch_X = torch.from_numpy(samples).cuda()

        e = self.embed(batch_X)
        idx = torch.randperm(e[0].shape[0])

        e1 = torch.cat((e[0], e[0]), dim=0)
        e2 = torch.cat((e[1], e[1][idx]), dim=0)

        predicted_embedding = e1

        prediction_loss = torch.bmm(
            e2.view(2 * batch_size, 1, -1),
            predicted_embedding.view(2 * batch_size, -1, 1),
        ).view(2 * batch_size)
        prediction_loss = (
            -torch.sigmoid(prediction_loss[:batch_size])
            + (torch.sigmoid(prediction_loss[batch_size:]))
        ).mean()

        support_loss = None
        e_labelled = torch.cat((e[0][labels != -1], e[1][labels != -1]))
        labels = labels[labels != -1]
        labels = torch.from_numpy(labels.astype(np.int)).cuda()
        labels = torch.cat((labels, labels))
        if e_labelled.shape[0] > 4 and True:
            class_predictions = self.predict_class_logits(e_labelled)
            support_loss = self.loss(class_predictions, labels)

        reg = 0.05 * (torch.mean(torch.abs(e[0])) + torch.mean(torch.abs(e[1])))

        L = prediction_loss + reg
        if not support_loss is None:
            L += 1 * support_loss
            support_loss = float(support_loss.data.cpu().numpy())
        else:
            support_loss = np.nan

        prediction_loss = float(prediction_loss.data.cpu().numpy())

        reg = float(reg.data.cpu().numpy())

        batch_statistics_dict["prediction_loss"].append(prediction_loss)
        batch_statistics_dict["support_loss"].append(support_loss)
        batch_statistics_dict["reg"].append(reg)

        return L


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
        initial_batch_size=512,
        device="cuda",
    ):
        if self.loss_function is None:
            self.loss_function = torch.nn.CrossEntropyLoss(
                weight=torch.Tensor([0.01, 1.0, 1.0]).to(torch.device(device))
            )

        sample_indices = np.arange(0, X.shape[0], dtype=np.int)
        batch_size = initial_batch_size

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

        batch_statistics_dict["traj_prediction_loss"].append(
            (batch_number, loss.data.cpu().numpy())
        )
        return loss

    def predict_trajectories(
        self, X, return_logits=False, medianfilter=False, cuda=True
    ):
        self.eval()
        trajectory_predictions = []
        for batch in bb_behavior.utils.model_selection.iterate_minibatches(
            X, None, 2048, include_small_last_batch=True
        ):
            input_X = torch.from_numpy(batch)
            if cuda:
                input_X = input_X.cuda()
            pred = self(input_X)

            if not return_logits:
                pred = torch.nn.functional.softmax(pred, dim=1)
            pred = pred.detach().cpu().numpy()

            if medianfilter:
                pred = scipy.ndimage.median_filter(pred, size=(1, 1, 5))

            trajectory_predictions.append(pred)

        trajectory_predictions = np.concatenate(trajectory_predictions)

        return trajectory_predictions
