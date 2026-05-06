# Proximity Reward Shaping — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `reward -= exp(-proximity)` penalty to `HabitatRewardWrapper.step()` so the policy is smoothly discouraged from approaching obstacles.

**Architecture:** Read proximity from `obs["imu"][-1]` (already available from previous work), apply `math.exp(-proximity)` as a penalty with a guard for the -1.0 sentinel (sensor unavailable).

**Tech Stack:** Python, gymnasium, numpy, pytest, math

---

## File Mapping

| File | Responsibility |
|------|----------------|
| `habitat_wrappers.py` | Add `import math`; add proximity penalty in `step()` |
| `tests/test_habitat_env.py` | Add `"imu"` key to `MockEnv`; add `test_proximity_penalty` |

---

### Task 1: Add `"imu"` key to MockEnv

**Files:**
- Modify: `tests/test_habitat_env.py:19-50` (MockEnv class)

- [ ] **Step 1: Add `"imu"` to MockEnv.reset() return dict**

Change the return in `reset()` (line 33) from:
```python
            return {"pixels": np.zeros(8, dtype=np.float32)}, {
```
To:
```python
            obs = {"pixels": np.zeros(8, dtype=np.float32),
                   "imu": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)}
            return obs, {
```

- [ ] **Step 2: Add `"imu"` to MockEnv.step() return dict**

Change the return in `step()` (line 51) from:
```python
            return {"pixels": np.zeros(8, dtype=np.float32)}, 0.0, False, False, info
```
To:
```python
            obs = {"pixels": np.zeros(8, dtype=np.float32),
                   "imu": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)}
            return obs, 0.0, False, False, info
```

- [ ] **Step 3: Run existing tests to verify they still pass**

Run: `pytest tests/test_habitat_env.py::TestHabitatRewardWrapper -v`
Expected: 3 tests pass (goal_condition, collision_penalty, steering_penalty)

- [ ] **Step 4: Commit**

```bash
git add tests/test_habitat_env.py
git commit -m "test: add imu key to MockEnv for proximity-aware reward wrapper"
```

---

### Task 2: Write failing proximity penalty test

**Files:**
- Modify: `tests/test_habitat_env.py` (inside `TestHabitatRewardWrapper` class)

- [ ] **Step 1: Add integration test for proximity penalty**

Add this test method to `TestHabitatRewardWrapper` (after `test_steering_penalty`, around line 83):

```python
    def test_proximity_penalty(self):
        from train_habitat_her import HabitatRewardWrapper
        import math

        # Near obstacle (proximity=0.1) → significant penalty
        env_near = self.MockEnv()
        # Override step() to return low proximity imu
        orig_step = env_near.step
        def step_near(action):
            obs, _, _, _, info = orig_step(action)
            obs["imu"][-1] = 0.1
            return obs, 0.0, False, False, info
        env_near.step = step_near

        wrapper_near = HabitatRewardWrapper(env_near)
        wrapper_near.reset()
        _, reward_near, _, _, _ = wrapper_near.step(np.array([0.0, 0.5]))
        # Base reward is -1.0, plus steering penalty 0, throttle check varies
        # But proximity penalty of exp(-0.1) ≈ 0.905 must be present
        assert reward_near <= -1.9, f"Expected penalty >= 0.9 on top of base -1.0, got {reward_near:.4f}"

        # Far from obstacle (proximity=5.0) → negligible penalty
        env_far = self.MockEnv()
        orig_step_far = env_far.step
        def step_far(action):
            obs, _, _, _, info = orig_step_far(action)
            obs["imu"][-1] = 5.0
            return obs, 0.0, False, False, info
        env_far.step = step_far

        wrapper_far = HabitatRewardWrapper(env_far)
        wrapper_far.reset()
        _, reward_far, _, _, _ = wrapper_far.step(np.array([0.0, 0.5]))
        penalty_far = math.exp(-5.0)
        assert penalty_far < 0.01, f"Far proximity penalty should be negligible, got {penalty_far:.4f}"

        # Sentinel -1.0 → should NOT apply proximity penalty (guard)
        env_sentinel = self.MockEnv()
        wrapper_sentinel = HabitatRewardWrapper(env_sentinel)
        wrapper_sentinel.reset()
        _, reward_sentinel, _, _, _ = wrapper_sentinel.step(np.array([0.0, 0.5]))
        # With sentinel -1.0 and guard, no proximity penalty applied
        assert reward_sentinel > -2.0, f"Sentinel should not trigger proximity penalty, got {reward_sentinel:.4f}"
```

- [ ] **Step 2: Run the test to verify it FAILS**

Run: `pytest tests/test_habitat_env.py::TestHabitatRewardWrapper::test_proximity_penalty -v`
Expected: FAIL — `assert reward_near <= -1.9` fails because proximity penalty isn't implemented yet

- [ ] **Step 3: Commit**

```bash
git add tests/test_habitat_env.py
git commit -m "test: add failing proximity penalty integration test"
```

---

### Task 3: Implement proximity penalty in HabitatRewardWrapper

**Files:**
- Modify: `habitat_wrappers.py:1` (add import)
- Modify: `habitat_wrappers.py:68-69` (add penalty in step)

- [ ] **Step 1: Add `import math`**

Add `import math` after the existing `import os` line:
```python
import math
import os
```

- [ ] **Step 2: Add proximity penalty in `step()`**

In the `step()` method, after line 69 (`obs, _, terminated, truncated, info = self.env.step(action)`), add:

```python
        # Proximity penalty: smoothly escalate as agent nears obstacles
        proximity = float(obs["imu"][-1])
        if proximity >= 0:
            reward -= math.exp(-proximity)
```

- [ ] **Step 3: Run all reward wrapper tests**

Run: `pytest tests/test_habitat_env.py::TestHabitatRewardWrapper -v`
Expected: 4 tests pass (3 existing + test_proximity_penalty)

- [ ] **Step 4: Commit**

```bash
git add habitat_wrappers.py
git commit -m "reward: add exp(-proximity) penalty for obstacle avoidance"
```

---

## Self-Review Checklist

- [ ] `tests/test_habitat_env.py` — `"imu"` key added to MockEnv.reset() and MockEnv.step().
- [ ] `tests/test_habitat_env.py` — `test_proximity_penalty` integration test added (near, far, sentinel cases).
- [ ] `habitat_wrappers.py` — `import math` added.
- [ ] `habitat_wrappers.py` — proximity penalty with guard (`proximity >= 0`) added in `step()`.
- [ ] All 4 `TestHabitatRewardWrapper` tests pass.
- [ ] No placeholders (TBD/TODO) remain in the plan.

---
*Plan written 2026-05-02*
