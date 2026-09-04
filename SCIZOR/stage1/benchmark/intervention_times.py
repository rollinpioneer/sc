import numpy as np

def select_intervention_times(actions, num_times=4, min_separation=10):
    actions = np.asarray(actions)
    T = len(actions)
    if T <= 10:
        return [(int(t), 'mid_trajectory') for t in range(min(T, num_times))]
    valid = np.arange(5, T - 5)
    motion = np.linalg.norm(actions[:, :-1], axis=1)
    grip = np.r_[0.0, np.abs(np.diff(actions[:, -1]))]
    chosen = []
    tail = valid[valid >= int(.6 * T)]
    positive_tail = [int(t) for t in tail if motion[int(t)] > 1e-6]
    specs = [
        (valid[np.argsort(grip[valid])[::-1]], 'gripper_event'),
        (valid[np.argsort(motion[valid])[::-1]], 'motion_peak'),
        (sorted(positive_tail, key=lambda t: (motion[t], -t)), 'late_precision'),
        (sorted(valid, key=lambda t: abs(int(t) - round(.5 * T))), 'mid_trajectory'),
    ]
    for values, reason in specs:
        for t in values:
            t = int(t)
            if all(abs(t - old) >= min_separation for old, _ in chosen):
                chosen.append((t, reason)); break
        if len(chosen) >= num_times: break
    if len(chosen) < num_times:
        fallback = sorted(valid, key=lambda t: (-motion[int(t)], abs(int(t) - round(.5 * T))))
        for t in fallback:
            t = int(t)
            if all(abs(t - old) >= min_separation for old, _ in chosen):
                chosen.append((t, 'fallback_motion'))
            if len(chosen) >= num_times:
                break
    return chosen[:num_times]
