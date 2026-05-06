# Design: Add Habitat-Lab ProximitySensor to IMU Observation

## Summary

Register the built-in Habitat-Lab `ProximitySensor` in the task config, extract its scalar observation in `HabitatNavEnv`, and append it as the 11th element of the IMU vector. Update the observation space dimension from `(10,)` to `(11,)`.

## Context

The Habitat image-goal navigation environment (`habitat_env.py`) already synthesizes a 10-D IMU vector from agent state deltas. The project wants to augment this with an omnidirectional proximity measurement so the policy can sense nearby obstacles without relying solely on visual features.

## Architecture

### 1. Config Layer (`configs/habitat_config.py`)

Add a toggle so the sensor can be disabled if needed:

```python
@dataclass
class HabitatNavConfig:
    ...
    proximity_sensor: bool = True
```

### 2. Habitat-Lab Config Builder (`habitat_env.py::_build_habitat_config`)

Import `ProximitySensorConfig` from `habitat.config.default_structured_configs` and append it to the task sensors:

```python
from habitat.config.default_structured_configs import ProximitySensorConfig

if cfg.proximity_sensor:
    base_cfg.habitat.task.sensors += [OmegaConf.structured(ProximitySensorConfig())]
```

Habitat-Lab will then expose the observation under the key `"proximity"`.

### 3. Environment Integration (`habitat_env.py`)

- **Dimension bump**: `self._imu_dimension` changes from `10` to `11`.
- **Observation space**: `observation_space["imu"]` shape changes from `(10,)` to `(11,)`.
- **Reset path**: After `self._env._task.reset(...)` and `self._sim.get_sensor_observations()`, read:
  ```python
  self._proximity = float(obs.get("proximity", -1.0))
  ```
- **Step path**: After `self._sim.get_sensor_observations()`, read the same key.
- **`_synthesize_imu`**: Append `self._proximity` as the final element.

The resulting IMU vector becomes:

```
[action[0], action[1], ax, ay, gx, gy,
 mean_resultant_accel_20, mean_throttle_20,
 geo_dist, mask_flag, proximity]
```

### 4. Data Flow (per step)

```
habitat.Env.step()
  → obs["proximity"] = distance to nearest obstacle (float, >= 0)
  → HabitatNavEnv.step(action)
    → self._proximity = obs["proximity"]
    → self._current_imu = _synthesize_imu()
    → _get_obs() returns {"image": RGB, "imu": (11,) float32}
```

## Error Handling

- If `obs["proximity"]` is absent (older habitat-lab, or sensor disabled), default to `-1.0`. This sentinel lets the policy learn that the reading is unavailable rather than silently using `0.0` (which would mean "touching an obstacle").

## Testing Plan

1. **Unit tests** (`tests/test_habitat_env.py`):
   - Assert `obs["imu"].shape == (11,)` after `reset()` and `step()`.
   - Assert `obs["imu"][-1] >= -1.0`.
2. **Smoke test**: Run `python -c "from habitat_env import HabitatNavEnv; ..."` to verify the environment instantiates without shape mismatch.
3. **Wrapper regression**: Confirm `StackingWrapper` and `MobileNetFeatureWrapper` still accept the `(11,)` IMU because they operate on the pre-stacked `imu` key generically.

## Files Modified

| File | Change |
|------|--------|
| `configs/habitat_config.py` | Add `proximity_sensor: bool = True` |
| `habitat_env.py` | Register `ProximitySensorConfig`; update `_imu_dimension`, `observation_space`, `_synthesize_imu`, `reset`, `step` |
| `tests/test_habitat_env.py` | Update shape assertions from `(10,)` / `(6,)` to `(11,)`; add proximity sanity checks |

## Out of Scope

- Reward shaping based on proximity.
- Multi-ray directional proximity.
- Depth-sensor-based proximity.

---
*Approved 2026-05-01*
