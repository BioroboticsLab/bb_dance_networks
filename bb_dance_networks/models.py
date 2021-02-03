import torch
import torch.nn
import torch.optim
import numpy as np


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

    loss_function = None

    def __init__(
        self,
        feature_channels,
        embedding_input_length,
        n_classes,
        trajectory_length,
        embedding_model_class,
    ):
        super().__init__()

        self.embedding = embedding_model_class(
            channels=feature_channels, n_classes=n_classes
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
        datareader,
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

        sample_indices = np.arange(0, datareader.X.shape[0], dtype=np.int)
        batch_size = initial_batch_size + batch_number // 500

        drawn_sample_indices = np.random.choice(
            sample_indices, size=batch_size, replace=False
        )
        samples = datareader.X[drawn_sample_indices]
        X = torch.from_numpy(samples).cuda()

        labels = Y[drawn_sample_indices, 31:-31]
        labels = torch.from_numpy(labels.astype(np.int)).cuda()

        prediction = self(X)
        prediction = prediction.view(batch_size * prediction.shape[2], 3)
        labels = labels.view(batch_size * labels.shape[1])

        loss = self.loss_function(prediction, labels)

        batch_statistics_dict["traj_prediction_loss"].append(loss.data.cpu().numpy())

        return loss
