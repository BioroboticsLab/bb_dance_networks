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
        labels = np.zeros(shape=[1, frames_per_trajectory])
        df_row_idx = 0
        for idx, ts in enumerate(
            np.linspace(begin_ts, end_ts, num=frames_per_trajectory)
        ):
            row = available_labels.iloc[df_row_idx, :]
            if row.timestamp <= ts:
                labels[0, idx] = row.label
                df_row_idx += 1
                if df_row_idx >= available_labels.shape[0]:
                    break
        temporal_labels.append(labels)
    temporal_labels = np.concatenate(temporal_labels, axis=0)

    return temporal_labels


def generate_data_for_ground_truth(
    all_samples_df,
    unlabelled_ratio=0.01,
    frame_margin=6 * 3 * 3,
    fps=6,
    n_subsample_results=None,
    verbose=True,
):
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

    # Further reduce to one sample per X sec per individual.
    reduced_samples_df["sample_interval"] = reduced_samples_df.timestamp.apply(
        lambda t: t // (5)
    ).astype(np.int)

    sample_pivot = reduced_samples_df.pivot_table(
        index=["bee_id", "sample_interval"], values="frame_id", aggfunc="count"
    )
    sample_pivot.columns = ["cnt"]
    sample_pivot = sample_pivot.reset_index(level=(0, 1))
    sample_pivot["sample_index"] = np.arange(0, sample_pivot.shape[0]).astype(np.int)

    drawn_samples = reduced_samples_df.merge(
        sample_pivot, on=["bee_id", "sample_interval"], how="inner"
    )

    drawn_samples = drawn_samples.groupby("sample_index").apply(lambda df: df.sample(1))
    print(
        "Reduced to one sample per behavior ({} samples).".format(
            drawn_samples.shape[0]
        )
    )

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
