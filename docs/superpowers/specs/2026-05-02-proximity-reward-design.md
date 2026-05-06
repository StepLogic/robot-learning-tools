# Design: Proximity-Based Reward Shaping in HabitatRewardWrapper

## Summary

Add a proximity penalty term to `HabitatRewardWrapper.step()` using `exp(-proximity)`, so the policy is smoothly discouraged from approaching obstacles.

## Context

The Habitat image-goal navigation environment now exposes a proximity sensor reading as the 11th element of the IMU vector (`obs["imu"][-1]`). This value is the distance in meters to the nearest obstacle (0 = touching). The reward wrapper currently has no access to this signal before a collision happens.

## Design

### Reward Term

```
reward -= exp(-proximity)
```

- **Proximity far** (≥5m): `exp(-5) ≈ 0.007` — negligible penalty
- **Proximity moderate** (1–2m): `exp(-1) ≈ 0.37` — mild penalty
- **Proximity close** (<0.5m): `exp(-0.5) ≈ 0.61` — significant penalty
- **Proximity zero** (touching): `exp(0) = 1.0` — max penalty, matches existing collision penalty magnitude

### Guard

Only apply when `proximity >= 0`. When the proximity sensor is unavailable, the env produces `-1.0` as a sentinel. `exp(1.0) ≈ 2.72` would be an incorrect heavy penalty, so the guard prevents it.

### Placement

Single line added to `HabitatRewardWrapper.step()` in `habitat_wrappers.py`, after the existing `obs, _, terminated, truncated, info = self.env.step(action)` call:

```python
proximity = float(obs["imu"][-1])
if proximity >= 0:
    reward -= math.exp(-proximity)
```

Requires `import math` at the top of the file.

## Interaction with Existing Terms

The proximity penalty **supplements** the existing collision penalty (`-1.0` when `info["hit"]`). They serve different purposes:
- Collision penalty: binary, punishes actual contact
- Proximity penalty: smooth, discourages approaching obstacles in the first place

The max proximity penalty (1.0 when touching) is the same magnitude as the collision penalty, so at contact the total penalty is ~2.0 — reasonable given that both collision and proximity=0 signal the same event through different channels.

## Files Modified

| File | Change |
|------|--------|
| `habitat_wrappers.py` | Add `import math`; add proximity penalty (3 lines) in `step()` |

## Testing Plan

1. **Unit test**: Extend `TestHabitatRewardWrapper` with a `test_proximity_penalty` that asserts:
   - `reward < 0` when proximity is low (e.g., 0.1 → penalty ≈ 0.90)
   - `reward ≈ 0` when proximity is high (e.g., 5.0 → penalty ≈ 0.007)
   - No penalty applied when proximity = -1.0 (sentinel guard)
2. **Smoke test**: Run existing `TestHabitatRewardWrapper` tests to confirm no regressions.

## Out of Scope

- Removing or modifying the existing collision penalty
- Making `k_proximity` configurable (fixed at 1.0)
- Applying proximity penalty to non-Habitat environments

---
*Approved 2026-05-02*
