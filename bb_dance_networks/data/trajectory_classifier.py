import bisect
import numpy as np
import pandas
import tqdm.auto

import bb_behavior.trajectory.features


def subsample_df(df, unlabelled_ratio=0.05, verbose=True):
    random_subsampling = np.random.uniform(size=df.shape[0], low=0.0, high=1.0) > (
        1.0 - unlabelled_ratio
    )
    random_subsampling = random_subsampling | (df.label.values > 0)
    reduced_samples_df = df.loc[random_subsampling, :].reset_index(drop=True)
    random_subsampling.sum(), reduced_samples_df.shape
    if verbose:
        print(
            "Subsampled {} samples out of a total of {}; retaining labelled samples.".format(
                reduced_samples_df.shape[0], df.shape[0]
            )
        )
    return reduced_samples_df


def generate_label_sequences(drawn_samples, all_samples_df, frame_margin, fps):
    temporal_margin = frame_margin / fps
    frames_per_trajectory = frame_margin * 2

    temporal_labels = []
    for (bee_id, timestamp) in tqdm.auto.tqdm(
        drawn_samples[["bee_id", "timestamp"]].itertuples(index=False),
        total=drawn_samples.shape[0],
    ):
        begin_ts, end_ts = timestamp - temporal_margin, (timestamp + temporal_margin)
        available_labels = all_samples_df[all_samples_df.bee_id == bee_id]
        available_labels = available_labels[
            available_labels.timestamp.between(begin_ts, end_ts)
        ]

        available_labels = available_labels.sort_values("timestamp")
        available_timestamps = available_labels.timestamp.values
        available_labels = available_labels.label.values

        labels = np.zeros(shape=[1, frames_per_trajectory])
        for idx, ts in enumerate(
            np.linspace(begin_ts, end_ts, num=frames_per_trajectory)
        ):
            label_index = bisect.bisect_left(available_timestamps, ts)
            # Before the start of any label.
            if label_index == 0 and available_timestamps[0] < ts:
                continue
            # Any label ended before that.
            if label_index == len(available_timestamps):
                continue
            if available_timestamps[label_index] > ts:
                labels[0, idx] = available_labels[label_index - 1]
            else:
                assert available_timestamps[label_index] == ts
                labels[0, idx] = available_labels[label_index]
        temporal_labels.append(labels)
    temporal_labels = np.concatenate(temporal_labels, axis=0)

    return temporal_labels


def generate_data_for_ground_truth(
    all_samples_df,
    unlabelled_ratio=0.1,
    frame_margin=6 * 3 * 3,
    fps=6,
    n_subsample_results=None,
    verbose=True,
):
    all_samples_df = all_samples_df.copy()

    def target_to_index(t):
        return ["nothing", "follower", "waggle"].index(t)

    all_samples_df["group"] = all_samples_df.timestamp.apply(
        lambda t: t // (60 * 60)
    ).astype(np.int)
    all_samples_df["label"] = all_samples_df.label.apply(target_to_index)

    # Reduce amount of data for the long trajectories.
    reduced_samples_df = subsample_df(
        all_samples_df, verbose=verbose, unlabelled_ratio=unlabelled_ratio
    )

    drawn_samples = reduced_samples_df

    temporal_labels = generate_label_sequences(
        drawn_samples, all_samples_df=all_samples_df, frame_margin=frame_margin, fps=fps
    )
    if verbose:
        print("Generated labels shape: {}.".format(temporal_labels.shape))

    from bb_behavior.trajectory.features import FeatureTransform

    features = [FeatureTransform.Angle2Geometric(), FeatureTransform.Egomotion()]
    data_reader = bb_behavior.trajectory.features.DataReader(
        dataframe=drawn_samples,
        use_hive_coords=True,
        frame_margin=frame_margin,
        target_column="label",
        feature_procs=features,
        sample_count=n_subsample_results,
        chunk_frame_id_queries=True,
        n_threads=4,
    )

    data_reader.create_features()

    if verbose:
        print("Generated features shape: {}.".format(data_reader.X.shape))

    label_indices = np.array(data_reader._valid_sample_indices, dtype=np.int)
    used_labels = temporal_labels[label_indices, :]

    return data_reader, used_labels
