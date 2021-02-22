import datetime
import numpy as np
import pandas

import bb_behavior.db
import bb_behavior.utils.misc


def apply_model_to_interval(
    dt_from,
    dt_to,
    bee_ids,
    model_path,
    feature_normalizer_path,
    cam_ids,
    n_threads=8,
    verbose=False,
):

    import bb_behavior.db
    import bb_behavior.db.sampling
    import bb_behavior.trajectory.features
    from bb_dance_networks.preprocessing import FeatureNormalizer
    import torch

    all_samples = []
    use_each_n_frame = 6 * 5

    for cam_id in cam_ids:
        all_frames = bb_behavior.db.sampling.get_frames(cam_id, dt_from, dt_to)
        all_frames = all_frames[::use_each_n_frame]

        for bee_id in bee_ids:
            for (timestamp, frame_id, cam_id) in all_frames:
                all_samples.append((bee_id, cam_id, frame_id, timestamp))

    all_samples_df = pandas.DataFrame(
        all_samples, columns=["bee_id", "cam_id", "frame_id", "timestamp"]
    )

    if verbose:
        print(
            "Getting data for {} samples (from approx. 2 x {} frames and {} bees).".format(
                all_samples_df.shape[0], len(all_frames), len(bee_ids)
            ),
            flush=True,
        )

    from bb_behavior.trajectory.features import FeatureTransform

    features = [FeatureTransform.Angle2Geometric(), FeatureTransform.Egomotion()]
    additional_kwargs = dict()
    if not verbose:
        additional_kwargs["progress"] = None
    datareader = bb_behavior.trajectory.features.DataReader(
        dataframe=all_samples_df,
        use_hive_coords=True,
        frame_margin=6 * 3 * 3,
        target_column=None,
        feature_procs=features,
        chunk_frame_id_queries=True,
        n_threads=n_threads,
        **additional_kwargs
    )
    datareader.create_features()

    if verbose:
        print("Fetched {} samples.".format(datareader.X.shape[0]), flush=True)

    normalizer = FeatureNormalizer.load(feature_normalizer_path)
    datareader.X[:] = normalizer.transform(datareader.X)

    model = torch.load(model_path)
    model.eval()
    pred = model.predict_trajectories(
        datareader.X, return_logits=False, medianfilter=True
    )
    del model

    valid_samples = datareader.samples.iloc[datareader._valid_sample_indices]
    valid_samples = valid_samples.copy().reset_index(drop=True)

    return valid_samples, pred


def merge_predictions_to_events(
    samples, all_predictions, prediction_margin=5, confidence_threshold=0.5
):

    all_events = []
    for bee_id in samples.bee_id.unique():

        bee_idx = samples.bee_id.values == bee_id
        df = samples.iloc[bee_idx]
        sorted_idx = np.argsort(df.timestamp.values)
        df = df.iloc[sorted_idx]
        predictions = all_predictions[bee_idx][sorted_idx]

        fps = 6
        sequence_length = predictions.shape[2]
        mid_timestamp_index = sequence_length // 2
        last_bee_timestamp = None

        for idx in range(prediction_margin, predictions.shape[0] - prediction_margin):
            frame_id, cam_id = df.frame_id.iloc[idx], df.cam_id.iloc[idx]

            labels = predictions[idx] > confidence_threshold
            if labels[1:, :].sum() == 0:
                continue
            labels = np.argmax(labels, axis=0)
            assert labels.shape[0] > 3

            mid_timestamp = df.timestamp.iloc[idx]

            for t in range(sequence_length):
                l = labels[t]
                conf = predictions[idx, l, t]
                ts = mid_timestamp + datetime.timedelta(
                    seconds=(t - mid_timestamp_index) / fps
                )

                if (last_bee_timestamp is not None) and (ts < last_bee_timestamp):
                    continue
                last_bee_timestamp = ts

                all_events.append(
                    dict(
                        bee_id=bee_id,
                        timestamp=ts,
                        mid_frame_id=frame_id,
                        cam_id=cam_id,
                        label=l,
                        confidence=conf,
                    )
                )
    all_events = pandas.DataFrame(all_events)

    return all_events


def merge_consecutive_events(
    gen, min_duration, max_gap_length, class_key=None, timestamp_key=None
):

    events = []

    if timestamp_key is None:
        timestamp_key = lambda d: d
        if class_key is not None:
            raise ValueError(
                "If timestamp_key is None, the generator should just return timestamps."
            )

    last_timestamp = None
    last_class_key = None
    last_event = []

    for item in gen:
        timestamp = timestamp_key(item)
        key = class_key(item) if (class_key is not None) else None

        if last_timestamp is None:
            last_timestamp = timestamp
            last_class_key = key

        delta = timestamp - last_timestamp
        if (delta > max_gap_length) or (
            (class_key is not None) and (last_class_key != key)
        ):
            if last_event:
                if last_event[1] - last_event[0] >= min_duration:
                    events.append(last_event)
                last_event = None

        if not last_event:
            last_event = [timestamp, None, key]
        last_event[1] = timestamp

        last_timestamp = timestamp
        last_class_key = key

    if last_event and (last_event[1] - last_event[0] >= min_duration):
        events.append(last_event)

    return events


def extract_consecutive_dance_events(
    all_events,
    min_dance_duration=datetime.timedelta(seconds=2),
    max_dance_gap_length=datetime.timedelta(seconds=2),
):
    dances_df = []

    for idx, (bee_id, bee_df) in enumerate(all_events.groupby("bee_id")):
        bee_df = bee_df[bee_df.label == 2]

        bee_dances = merge_consecutive_events(
            bee_df[["timestamp", "cam_id"]].itertuples(index=False),
            min_duration=min_dance_duration,
            max_gap_length=max_dance_gap_length,
            timestamp_key=lambda d: d[0],
            class_key=lambda d: d[1],
        )
        for dance in bee_dances:
            dances_df.append(
                [bee_id, bb_behavior.utils.misc.generate_64bit_id()] + dance
            )

    dances_df = pandas.DataFrame(
        dances_df, columns=["bee_id", "dance_id", "from", "to", "cam_id"]
    )
    dances_df["duration"] = (dances_df["to"] - dances_df["from"]).apply(
        lambda d: d.total_seconds()
    )

    return dances_df


def fetch_follower_from_database(
    dances_df,
    all_events,
    min_following_duration=datetime.timedelta(seconds=1),
    max_following_gap_length=datetime.timedelta(seconds=2),
    verbose=False,
    fps=6,
):
    from collections import Counter
    import bb_behavior.db.sampling

    all_follower_events = []

    with bb_behavior.db.DatabaseCursorContext() as cursor:
        for bee_id, bee_df in dances_df.groupby("bee_id"):
            for (dance_id, dt_from, dt_to, cam_id) in bee_df[
                ["dance_id", "from", "to", "cam_id"]
            ].itertuples(index=False):
                frames = bb_behavior.db.get_frames(
                    cam_id, dt_from, dt_to, cursor=cursor, cursor_is_prepared=True
                )
                detections = bb_behavior.db.get_bee_detections(
                    bee_id,
                    frames=frames,
                    use_hive_coords=True,
                    cursor=cursor,
                    cursor_is_prepared=True,
                    confidence_threshold=0.5,
                )

                bee_id_count = Counter()
                bee_candidates = []
                for det in detections:
                    if det is None:
                        continue
                    (ts, frame_id, x, y, orientation, track_id) = det
                    xlim = [x - 14, x + 14]
                    ylim = [y - 14, y + 14]
                    candidates = (
                        bb_behavior.db.sampling.get_detections_for_location_in_frame(
                            frame_id,
                            xlim,
                            ylim,
                            confidence_threshold=0.5,
                            cursor=cursor,
                            cursor_is_prepared=True,
                        )
                    )

                    for candidate in candidates:
                        bee_id = candidate[0]
                        c_o = candidate[8]
                        if bee_id is None:  # Drop unmarked bees.
                            continue
                        if c_o is None:
                            # Bees without orientation are usually in cells / on the glass..
                            continue

                        ts = candidate[1]
                        c_x, c_y = candidate[6:8]

                        dxy = np.array([(x - c_x), (y - c_y)], dtype=np.float32)
                        dxy /= np.linalg.norm(dxy)
                        oxy = np.array([np.cos(c_o), np.sin(c_o)])
                        relative_angle = np.inner(dxy, oxy)
                        if relative_angle < 0.5:
                            continue

                        bee_id_count[bee_id] += 1
                        bee_candidates.append(dict(bee_id=bee_id, timestamp=ts))

                bee_candidates = pandas.DataFrame(
                    bee_candidates, columns=["bee_id", "timestamp"]
                )
                valid_bee_ids = {ID for ID, cnt in bee_id_count.items() if cnt >= fps}
                bee_candidates = bee_candidates[
                    bee_candidates.bee_id.isin(valid_bee_ids)
                ]

                for follower_id, follower_df in bee_candidates.groupby("bee_id"):
                    follower_df = follower_df.sort_values("timestamp")

                    def fetch_label(timestamp):
                        begin = timestamp - datetime.timedelta(seconds=1.0 / fps)
                        end = timestamp + datetime.timedelta(seconds=1.0 / fps)

                        events = all_events[(all_events.bee_id == follower_id)]
                        events = events[
                            (events.timestamp > begin) & (events.timestamp < end)
                        ]
                        label = "unknown"
                        if events.shape[0] > 0:
                            label = list(set(events.label.values))
                            if len(label) > 1 and label[0] != 0 and verbose:
                                print(
                                    "{} has two different labels ({}) between {} and {}.".format(
                                        follower_id,
                                        label,
                                        events.timestamp.min(),
                                        events.timestamp.max(),
                                    )
                                )
                            label = label[-1]
                            label = ["nothing", "follower", "dancing"][label]

                        return label

                    follower_events = merge_consecutive_events(
                        follower_df[["timestamp"]].itertuples(index=False),
                        min_duration=min_following_duration,
                        max_gap_length=max_following_gap_length,
                        timestamp_key=lambda d: d[0],
                        class_key=lambda d: fetch_label(d[0]),
                    )

                    for (ts_from, ts_to, label) in follower_events:
                        all_follower_events.append(
                            (dance_id, follower_id, ts_from, ts_to, label)
                        )

    all_follower_events = pandas.DataFrame(
        all_follower_events,
        columns=["dance_id", "follower_id", "timestamp_from", "timestamp_to", "label"],
    )

    return all_follower_events
