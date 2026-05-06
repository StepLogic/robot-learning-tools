# Proximity Sensor in Habitat IMU — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register Habitat-Lab's built-in `ProximitySensor`, extract its scalar reading, and append it to the IMU observation vector (bumping dimension from 10 → 11).

**Architecture:** Add `ProximitySensorConfig` to the Habitat task config, read the `"proximity"` observation key in `HabitatNavEnv.reset()` and `step()`, store it in a new `self._proximity` field, and append it in `_synthesize_imu()`. Update `observation_space` and tests accordingly.

**Tech Stack:** Python, habitat-lab, habitat-sim, gymnasium, numpy, pytest

---

## File Mapping

| File | Responsibility |
|------|----------------|
| `configs/habitat_config.py` | Adds `proximity_sensor: bool` toggle to `HabitatNavConfig` |
| `habitat_env.py` | Registers `ProximitySensorConfig`; updates `_imu_dimension`, `observation_space`, `_synthesize_imu`, `reset`, `step` |
| `tests/test_habitat_env.py` | Fixes stale IMU shape assertion; adds proximity sanity checks |

---

## Task 1: Add `proximity_sensor` toggle to config

**Files:**
- Modify: `configs/habitat_config.py`

- [ ] **Step 1: Add field**

Insert a new boolean field `proximity_sensor: bool = True` in `HabitatNavConfig` after the existing fields (e.g., after `goal_max_distance`).

```python
    goal_max_distance: float = 10.0   # Cap on sampled goal distance (meters)
    proximity_sensor: bool = True      # Enable habitat-lab ProximitySensor
    randomize_scenes: bool = False     # Randomly switch scenes across episodes
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "from configs.habitat_config import HabitatNavConfig; c = HabitatNavConfig(); print(c.proximity_sensor)"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add configs/habitat_config.py
git commit -m "config: add proximity_sensor toggle"
```

---

## Task 2: Register ProximitySensor in Habitat config builder

**Files:**
- Modify: `habitat_env.py:47-49`
- Modify: `habitat_env.py:138-142`

- [ ] **Step 1: Import ProximitySensorConfig**

```python
from habitat.config.default_structured_configs import (
    VelocityControlActionConfig,
    ProximitySensorConfig,
)
```

- [ ] **Step 2: Append sensor to task config**

After the line `del base_cfg.habitat.task.measurements.top_down_map` (around line 141), add:

```python
    # Add proximity sensor
    if cfg.proximity_sensor:
        base_cfg.habitat.task.sensors += [OmegaConf.structured(ProximitySensorConfig())]
```

- [ ] **Step 3: Commit**

```bash
git add habitat_env.py
git commit -m "habitat: register ProximitySensor in task config"
```

---

## Task 3: Bump IMU dimension and observation space

**Files:**
- Modify: `habitat_env.py:203`
- Modify: `habitat_env.py:231-234`

- [ ] **Step 1: Update `_imu_dimension`**

Change `self._imu_dimension: int = 10` → `self._imu_dimension: int = 11`.

- [ ] **Step 2: Update `observation_space`**

The `"imu"` entry already uses `shape=(self._imu_dimension,)`, so it automatically picks up the new value. No further change needed here, but verify the line reads:

```python
            "imu":   spaces.Box(low=-np.inf, high=np.inf, shape=(self._imu_dimension,),
                               dtype=np.float32),
```

- [ ] **Step 3: Add `self._proximity` field in `__init__`**

After the line `self._collision_detected: bool = False` (around line 260), add:

```python
        self._proximity: float = -1.0
```

- [ ] **Step 4: Commit**

```bash
git add habitat_env.py
git commit -m "habitat: bump IMU dimension to 11 and add _proximity field"
```

---

## Task 4: Extract proximity in `reset()`

**Files:**
- Modify: `habitat_env.py:605-619`

In `reset()`, after `sim_obs = self._sim.get_sensor_observations()` and `obs["rgb"] = sim_obs["rgb"]`, add the proximity extraction before `# Reset habitat-lab measures ...`.

- [ ] **Step 1: Insert proximity read**

After:
```python
        sim_obs = self._sim.get_sensor_observations()
        obs["rgb"] = sim_obs["rgb"]
```

Add:
```python
        # Extract proximity sensor reading (Habitat-Lab key is "proximity")
        self._proximity = float(obs.get("proximity", -1.0))
```

- [ ] **Step 2: Commit**

```bash
git add habitat_env.py
git commit -m "habitat: extract proximity observation during reset"
```

---

## Task 5: Extract proximity in `step()`

**Files:**
- Modify: `habitat_env.py:712-714`

In `step()`, after `sim_obs = self._sim.get_sensor_observations()` and `self._current_image = sim_obs["rgb"]`, add the proximity extraction.

- [ ] **Step 1: Insert proximity read**

After:
```python
        sim_obs = self._sim.get_sensor_observations()
        self._current_image = sim_obs["rgb"]
```

Add:
```python
        # Extract proximity sensor reading
        self._proximity = float(obs.get("proximity", -1.0))
```

- [ ] **Step 2: Commit**

```bash
git add habitat_env.py
git commit -m "habitat: extract proximity observation during step"
```

---

## Task 6: Append proximity to IMU vector in `_synthesize_imu()`

**Files:**
- Modify: `habitat_env.py:510`

- [ ] **Step 1: Append to return array**

Change the return line from:

```python
        return np.array([self.action[0],self.action[1], ax,ay,gx,gy,mean_resultant, mean_throttle,gd,float(int(mask_img))], dtype=np.float32)
```

To:

```python
        return np.array([self.action[0],self.action[1], ax,ay,gx,gy,mean_resultant, mean_throttle,gd,float(int(mask_img)), self._proximity], dtype=np.float32)
```

- [ ] **Step 2: Commit**

```bash
git add habitat_env.py
git commit -m "habitat: append proximity to synthesized IMU vector"
```

---

## Task 7: Fix stale test assertions and add proximity checks

**Files:**
- Modify: `tests/test_habitat_env.py:111`

- [ ] **Step 1: Fix existing IMU shape assertion**

The current test asserts `obs["imu"].shape == (6,)`, which is stale (code already returns 10). Update it to `(11,)` to match the new dimension.

Change:
```python
        assert obs["imu"].shape == (6,)
```

To:
```python
        assert obs["imu"].shape == (11,)
```

- [ ] **Step 2: Add proximity sanity checks**

In `test_reset_and_step`, after the existing assertions, add:

```python
        assert obs["imu"][-1] >= -1.0, "proximity should be >= -1.0"
```

In `test_actual_vel_in_info`, after the existing assertions, add:

```python
        assert "actual_vel" in info, "actual_vel key missing from info"
        assert isinstance(info["actual_vel"], float)
        assert obs["imu"].shape == (11,), "IMU shape should be (11,) after step"
        assert obs["imu"][-1] >= -1.0, "proximity should be >= -1.0 after step"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_habitat_env.py::TestHabitatNavEnv -v`
Expected: All 3 tests pass (or skip if habitat-lab is not installed).

- [ ] **Step 4: Commit**

```bash
git add tests/test_habitat_env.py
git commit -m "test: fix IMU shape assertion and add proximity checks"
```

---

## Self-Review Checklist

- [ ] `configs/habitat_config.py` — `proximity_sensor` field added.
- [ ] `habitat_env.py` — `ProximitySensorConfig` imported and registered conditionally.
- [ ] `habitat_env.py` — `_imu_dimension` bumped to 11.
- [ ] `habitat_env.py` — `self._proximity` initialized.
- [ ] `habitat_env.py` — proximity extracted in `reset()`.
- [ ] `habitat_env.py` — proximity extracted in `step()`.
- [ ] `habitat_env.py` — proximity appended in `_synthesize_imu()`.
- [ ] `tests/test_habitat_env.py` — shape assertion fixed to `(11,)`.
- [ ] `tests/test_habitat_env.py` — proximity sanity checks added.
- [ ] No placeholders (TBD/TODO) remain in the plan.

---

*Plan written 2026-05-01*
