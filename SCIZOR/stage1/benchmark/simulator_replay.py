import copy
import numpy as np
from .perturbations import perturb_action

def create_shaped_env(dataset_path):
    from robomimic.utils import env_utils, file_utils, obs_utils
    obs_utils.initialize_obs_utils_with_obs_specs({'obs': {'low_dim': ['robot0_eef_pos'], 'rgb': []}})
    meta = copy.deepcopy(file_utils.get_env_metadata_from_dataset(dataset_path))
    meta['env_kwargs']['reward_shaping'] = True
    return env_utils.create_env_from_metadata(meta, render=False, render_offscreen=True), meta

def replay_episode(env, initial_state, actions, perturb_t=None, perturb_kind=None, magnitude=None, rng=None, camera_name='agentview', height=84, width=84, collect_images=True):
    actions = np.asarray(actions)
    executed = actions.copy()
    if perturb_t is not None:
        low, high = getattr(env, 'action_spec', getattr(env.env, 'action_spec'))
        executed[int(perturb_t)] = perturb_action(actions[int(perturb_t)], perturb_kind, magnitude, rng, low, high)
        changed = np.where(np.any(np.abs(executed - actions) > 1e-6, axis=1))[0]
        assert changed.tolist() == [int(perturb_t)]
    env.reset(); env.reset_to(initial_state)
    states, images, rewards, success, staged = [], [], [], [], []
    stage_fn = getattr(env.env, 'staged_rewards', None)
    for action in executed:
        _, reward, _, _ = env.step(action)
        states.append(np.asarray(env.get_state()['states']).copy())
        if collect_images:
            images.append(np.asarray(env.render(mode='rgb_array', height=height, width=width, camera_name=camera_name)).copy())
        rewards.append(float(reward))
        success.append(bool(env.is_success().get('task', False)))
        if stage_fn is not None: staged.append(np.asarray(stage_fn(), dtype=np.float32).reshape(-1))
    return {
        'actions': executed,
        'original_actions': actions.copy(),
        'states': np.asarray(states),
        'images': np.asarray(images, dtype=np.uint8) if collect_images else None,
        'rewards': np.asarray(rewards, dtype=np.float32),
        'success': np.asarray(success, dtype=np.bool_),
        'staged_rewards': np.asarray(staged, dtype=np.float32) if staged else None,
    }
