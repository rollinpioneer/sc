import numpy as np

def _run(mask, start, length=3):
    for i in range(int(start), len(mask)-length+1):
        if np.all(mask[i:i+length]): return i
    return None

def label_outcome(clean, perturbed, perturb_t):
    cr, pr = np.asarray(clean['rewards'], float), np.asarray(perturbed['rewards'], float)
    n = min(len(cr), len(pr))
    cs, ps = bool(clean['success'][-1]), bool(perturbed['success'][-1])
    out = {
        'failure_onset': None,
        'failure_type': 'ambiguous',
        'responsible_t': int(perturb_t),
        'responsible_start': max(0, int(perturb_t) - 1),
        'responsible_end': min(n - 1, int(perturb_t) + 1),
        'recovery_start': None,
        'recovery_end': None,
        'final_success_clean': cs,
        'final_success_perturbed': ps,
        'label_status': 'ambiguous',
    }
    if n < 4 or not np.isfinite(cr[:n]).all() or not np.isfinite(pr[:n]).all():
        return out

    reward_gap = cr[:n] - pr[:n]
    threshold = max(0.05, 0.20 * np.percentile(np.abs(np.diff(cr[:n])), 90))
    combined_gap = reward_gap.copy()
    if clean.get('staged_rewards') is not None and perturbed.get('staged_rewards') is not None:
        a = np.asarray(clean['staged_rewards'])
        b = np.asarray(perturbed['staged_rewards'])
        if a.ndim == 2 and b.shape == a.shape:
            combined_gap = np.maximum(combined_gap, np.max(a[:n] - b[:n], axis=1))

    mask = combined_gap > threshold
    onset = _run(mask, perturb_t, length=3)
    out.update({'failure_onset': onset, 'gap_threshold': float(threshold), 'label_status': 'ok'})

    if onset is None:
        if cs == ps:
            out['failure_type'] = 'no_effect'
        else:
            out['failure_type'] = 'ambiguous'
            out['label_status'] = 'ambiguous'
        return out

    peak = float(np.max(combined_gap[onset:])) if onset < n else 0.0
    if peak > 0:
        avg = np.convolve(combined_gap, np.ones(5) / 5.0, mode='valid')
        hits = np.flatnonzero(avg[onset:] <= 0.7 * peak)
        out['recovery_start'] = int(onset + hits[0]) if len(hits) else None

    out['recovery_end'] = _run(combined_gap <= 0.5 * threshold, onset, length=3)
    if out['recovery_end'] is not None and ps:
        out['failure_type'] = 'recovery_success'
    elif out['recovery_start'] is not None and not ps:
        out['failure_type'] = 'recovery_failure'
    elif not ps:
        out['failure_type'] = 'direct_failure' if onset <= int(perturb_t) + 3 else 'delayed_failure'
    elif cs == ps:
        out['failure_type'] = 'no_effect'
    else:
        out['label_status'] = 'ambiguous'
    return out
