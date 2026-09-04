import numpy as np

def perturb_action(action, kind, magnitude, rng, action_low, action_high):
    original = np.asarray(action)
    target = original.astype(np.float64, copy=True)
    if kind == 'zero_motion':
        target[:-1] = 0.0
    elif kind == 'reverse_motion':
        target[:-1] *= -1.0
    elif kind == 'flip_gripper':
        target[-1] *= -1.0
    elif kind == 'axis_impulse':
        axis = int(rng.integers(0, min(3, original.size - 1)))
        sign = float(np.sign(target[axis]))
        if sign == 0.0:
            sign = 1.0 if rng.integers(0, 2) else -1.0
        low = np.asarray(action_low, dtype=np.float64)
        high = np.asarray(action_high, dtype=np.float64)
        target[axis] = low[axis] if sign > 0 else high[axis]
    else:
        raise ValueError(kind)
    result = original + float(magnitude) * (target - original)
    return np.clip(result, action_low, action_high).astype(original.dtype, copy=False)
