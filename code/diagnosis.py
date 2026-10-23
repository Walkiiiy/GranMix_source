"""Contribution estimation and online interval calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from data import ReferenceTree


class BlockHistory:
    """Normal updates advance the block clock; auxiliary probes only add moments."""

    def __init__(self, parents, root, *, block_steps, history_blocks, scale_floor):
        self.parents = np.asarray(parents, dtype=np.int64)
        self.root = int(root)
        self.block_steps = int(block_steps)
        self.history_blocks = int(history_blocks)
        self.scale_floor = float(scale_floor)
        self.pending = np.zeros((len(self.parents), 4), dtype=np.float64)
        self.blocks = []
        self.next_index = 0
        self.paths = []
        for node in range(len(self.parents)):
            route = []
            while node != -1:
                route.append(node)
                node = int(self.parents[node])
            self.paths.append(np.asarray(route, dtype=np.int64))

    def accumulate(self, leaf_ids, scores, importance, *, audit_sum=None):
        for leaf in np.unique(leaf_ids):
            selected = leaf_ids == leaf
            w, y = importance[selected], scores[selected]
            moments = np.array([w.sum(), np.square(w).sum(), np.dot(w, y),
                                np.dot(w, y * y)], dtype=np.float64)
            route = self.paths[int(leaf)]
            self.pending[route] += moments
            if audit_sum is not None:
                audit_sum[route] += moments[2]

    def finish_normal_step(self, window_step):
        if window_step % self.block_steps:
            return
        root = self.pending[self.root]
        if root[0] <= 0:
            raise ArithmeticError("a completed ordinary block cannot have zero root weight")
        rms = max(self.scale_floor, float(np.sqrt(root[3] / root[0])))
        self.blocks.append({"index": self.next_index, "stats": self.pending.copy(), "scale": rms})
        self.next_index += 1
        if len(self.blocks) > self.history_blocks:
            self.blocks = self.blocks[-self.history_blocks:]
        self.pending.fill(0)


def readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ForecastRecord:
    window_index: int
    calibration: float
    normal_draw_count: int
    mean: np.ndarray
    base: np.ndarray
    radius: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    valid: np.ndarray
    effective_count: np.ndarray
    mixer_signal: np.ndarray
    standard_deviation: np.ndarray
    block_dispersion: np.ndarray
    drift: np.ndarray
    root_scale: float


def forecast_from_history(history, config, *, window_index, calibration, normal_draw_count):
    n = len(history.parents)
    mean, base = np.full(n, np.nan), np.full(n, np.nan)
    effective = np.zeros(n)
    sd, between, drift, mixer = (np.zeros(n) for _ in range(4))
    current_scale = config.root_scale_floor
    if history.blocks:
        moments = np.asarray([block["stats"] for block in history.blocks])
        scales = np.asarray([block["scale"] for block in history.blocks])
        indices = np.asarray([block["index"] for block in history.blocks])
        decay = config.history_decay ** (indices[-1] - indices)[:, None]
        weight = (decay * moments[:, :, 0]).sum(axis=0)
        squared_weight = (decay**2 * moments[:, :, 1]).sum(axis=0)
        first = (decay * moments[:, :, 2] / scales[:, None]).sum(axis=0)
        second = (decay * moments[:, :, 3] / scales[:, None]**2).sum(axis=0)
        present = weight > 0
        average = np.divide(first, weight, out=np.zeros(n), where=present)
        second_average = np.divide(second, weight, out=np.zeros(n), where=present)
        sd = np.sqrt(np.maximum(0, second_average - average**2))
        effective = np.divide(weight**2, squared_weight, out=np.zeros(n), where=squared_weight > 0)
        block_average = np.divide(moments[:, :, 2], scales[:, None] * moments[:, :, 0],
                                  out=np.zeros_like(moments[:, :, 0]), where=moments[:, :, 0] > 0)
        fractions = np.divide(decay * moments[:, :, 0], weight[None, :],
                              out=np.zeros_like(moments[:, :, 0]), where=weight[None, :] > 0)
        between = np.sqrt(np.maximum(0, (
            fractions * (block_average - average[None, :])**2).sum(axis=0)))
        recent = slice(-config.recent_blocks, None)
        recent_weight = (decay[recent] * moments[recent, :, 0]).sum(axis=0)
        recent_first = (decay[recent] * moments[recent, :, 2] / scales[recent, None]).sum(axis=0)
        recent_mean = np.divide(recent_first, recent_weight, out=average.copy(), where=recent_weight > 0)
        drift = np.abs(recent_mean - average)
        for index in range(len(history.blocks)):
            observed = moments[index, :, 0] > 0
            mixer[observed] = block_average[index, observed]
        current_scale = float(scales[-1])
        mean[present] = np.clip(current_scale * average[present], -config.clip, config.clip)
        standard_error = np.divide(sd, np.sqrt(effective), out=np.zeros(n), where=effective > 0)
        base[present] = config.base_floor + current_scale * (
            config.mean_coefficient * standard_error[present]
            + config.block_coefficient * between[present]
            + config.drift_coefficient * drift[present])
    valid = ((effective >= config.min_effective_count) & (effective > 0)
             & np.isfinite(mean) & np.isfinite(base))
    radius = np.full(n, np.inf)
    radius[valid] = calibration * base[valid]
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    lower[valid], upper[valid] = mean[valid] - radius[valid], mean[valid] + radius[valid]
    return ForecastRecord(
        window_index=int(window_index), calibration=float(calibration),
        normal_draw_count=int(normal_draw_count),
        mean=readonly(mean), base=readonly(base), radius=readonly(radius),
        lower=readonly(lower), upper=readonly(upper), valid=readonly(valid),
        effective_count=readonly(effective), mixer_signal=readonly(mixer),
        standard_deviation=readonly(sd), block_dispersion=readonly(between),
        drift=readonly(drift), root_scale=current_scale)


def forecast_to_state(record):
    result = {}
    for item in fields(record):
        value = getattr(record, item.name)
        if isinstance(value, np.ndarray):
            result[item.name] = [None if isinstance(x, float) and not np.isfinite(x) else x
                                 for x in value.tolist()]
        else:
            result[item.name] = value
    return result


def forecast_from_state(state):
    values = dict(state)
    scalar_fields = {"window_index", "calibration", "normal_draw_count", "root_scale"}
    for name in values.keys() - scalar_fields:
        if name == "valid":
            values[name] = readonly(values[name], dtype=bool)
        else:
            missing = -np.inf if name == "lower" else np.inf if name in {"upper", "radius"} else np.nan
            values[name] = readonly([missing if x is None else x for x in values[name]], dtype=np.float64)
    return ForecastRecord(**values)


@dataclass(frozen=True)
class AuditResult:
    window_index: int
    normal_draw_count: int
    normal_optimizer_updates: int
    excluded_probe_optimizer_updates: int
    excluded_probe_draw_count: int
    estimate: np.ndarray
    radius: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    saved_valid: np.ndarray
    failed_nodes: np.ndarray
    joint_error: bool
    calibration_before: float
    calibration_after: float
    target: str = "mean_over_ordinary_full_support_pre_update_states"


def audit_completed_window(*, config, saved, reference_mass, weighted_sum, draws,
                           normal_steps, excluded_steps, excluded_draws, calibration):
    """Test containment against the forecast frozen before these draws."""
    if draws <= 0:
        raise ValueError("the audit requires ordinary observations")
    mass = np.asarray(reference_mass, dtype=np.float64)
    t = saved.window_index
    estimate = np.clip(weighted_sum / (draws * mass), -config.clip, config.clip)
    log_failure = np.log(2 * len(mass) * t * (t + 1) / config.audit_failure_budget)
    x = log_failure / (draws * config.coverage_floor * mass)
    radius = config.clip * np.minimum(2, 2 * x / 3 + np.sqrt(2 * x + 4 * x**2 / 9))
    lower = np.maximum(-config.clip, estimate - radius)
    upper = np.minimum(config.clip, estimate + radius)
    failed = saved.valid & ((lower < saved.lower) | (upper > saved.upper))
    error = bool(np.any(failed))
    updated = max(0.0, calibration + config.calibration_step * (int(error) - config.target_joint_error))
    return AuditResult(
        window_index=t, normal_draw_count=int(draws), normal_optimizer_updates=int(normal_steps),
        excluded_probe_optimizer_updates=int(excluded_steps), excluded_probe_draw_count=int(excluded_draws),
        estimate=readonly(estimate), radius=readonly(radius), lower=readonly(lower),
        upper=readonly(upper), saved_valid=readonly(saved.valid), failed_nodes=readonly(failed),
        joint_error=error, calibration_before=float(calibration), calibration_after=updated)


@dataclass(frozen=True)
class DiagnosisConfig:
    window_steps: int = 100
    block_steps: int = 10
    history_blocks: int = 20
    recent_blocks: int = 5
    history_decay: float = .95
    clip: float = 1.0
    probe_norm_floor: float = 1e-8
    root_scale_floor: float = 1e-6
    coverage_floor: float = .01
    min_effective_count: float = 32.0
    base_floor: float = 1e-3
    mean_coefficient: float = 1.0
    block_coefficient: float = .5
    drift_coefficient: float = .5
    initial_calibration: float = 1.0
    calibration_step: float = .1
    target_joint_error: float = .1
    audit_failure_budget: float = .05

    def __post_init__(self):
        for name in ("window_steps", "block_steps", "history_blocks", "recent_blocks"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.window_steps % self.block_steps:
            raise ValueError("ordinary window length must be divisible by block length")
        if self.history_blocks < self.window_steps // self.block_steps:
            raise ValueError("history must retain all blocks of the current ordinary window")
        if self.recent_blocks > self.history_blocks:
            raise ValueError("recent history cannot exceed retained history")
        for name in ("clip", "probe_norm_floor", "root_scale_floor", "base_floor", "calibration_step"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_effective_count", "mean_coefficient", "block_coefficient",
                     "drift_coefficient", "initial_calibration"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("coverage_floor", "target_joint_error", "audit_failure_budget"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between zero and one")
        if not np.isfinite(self.history_decay) or not 0 < self.history_decay <= 1:
            raise ValueError("history_decay must be in (0, 1]")


@dataclass(frozen=True)
class BoundaryResult:
    audit: AuditResult
    forecast: ForecastRecord


def normalized_contributions(inner_products, own_domain_probe_norms, *, clip=1.0, norm_floor=1e-8):
    dots, norms = np.broadcast_arrays(np.asarray(inner_products, dtype=np.float64),
                                      np.asarray(own_domain_probe_norms, dtype=np.float64))
    if not np.isfinite([clip, norm_floor]).all() or min(clip, norm_floor) <= 0:
        raise ValueError("positive clipping bound and gradient-norm floor are required")
    if not np.isfinite(dots).all() or not np.isfinite(norms).all() or np.any(norms < 0):
        raise ValueError("contribution inputs must be finite, with nonnegative norms")
    return np.clip(dots / np.maximum(norms, norm_floor), -clip, clip)


class PartitionDiagnoser:
    """All-node history with an immutable forecast for each future ordinary window."""

    STATE_VERSION = 1

    def __init__(self, parents, reference_mass, config=None):
        self.config = config if config is not None else DiagnosisConfig()
        parent_array = np.asarray(parents)
        mass = np.asarray(reference_mass, dtype=np.float64)
        if (parent_array.ndim != 1 or mass.shape != parent_array.shape or parent_array.size == 0
                or not np.issubdtype(parent_array.dtype, np.integer)):
            raise ValueError("parents and reference_mass must be matching nonempty vectors")
        if np.any(parent_array < -1) or np.any(parent_array >= len(parent_array)):
            raise ValueError("parent ID outside the tree")
        children = [np.flatnonzero(parent_array == node).tolist() for node in range(len(parent_array))]
        tree = ReferenceTree(children, mass)
        if not np.array_equal(tree.parents, parent_array):
            raise ValueError("invalid parent structure")
        self.parents = parent_array.astype(np.int64, copy=True)
        self.mass = mass.copy()
        self.parents.flags.writeable = self.mass.flags.writeable = False
        self.n_nodes, self.root = tree.n, tree.root
        self.children = [np.array(row, dtype=np.int64) for row in tree.children]
        self.is_leaf = np.array([not row for row in tree.children], dtype=bool)
        cfg = self.config
        self.history = BlockHistory(self.parents, self.root, block_steps=cfg.block_steps,
                                    history_blocks=cfg.history_blocks, scale_floor=cfg.root_scale_floor)
        self.paths = self.history.paths
        self.calibration = float(cfg.initial_calibration)
        self.window_index = 1
        self.total_normal_updates = self.total_normal_draws = 0
        self.total_excluded_probe_updates = self.total_excluded_probe_draws = 0
        self.batch_size = None
        self._window_steps = self._window_draws = 0
        self._excluded_probe_updates = self._excluded_probe_draws = 0
        self._window_sum = np.zeros(self.n_nodes)
        self._saved_forecast = self._forecast()

    @property
    def saved_forecast(self):
        return self._saved_forecast

    def _forecast(self):
        return forecast_from_history(
            self.history, self.config, window_index=self.window_index,
            calibration=self.calibration, normal_draw_count=self.total_normal_draws)

    def _validate_observations(self, leaves, scores, importance):
        if self._window_steps >= self.config.window_steps:
            raise RuntimeError("complete_window must precede further observations")
        leaf = np.asarray(leaves)
        score = np.asarray(scores, dtype=np.float64)
        weight = np.asarray(importance, dtype=np.float64)
        if leaf.ndim != 1 or not leaf.size or score.shape != leaf.shape or weight.shape != leaf.shape:
            raise ValueError("leaf IDs, scores and importance weights must be matching nonempty vectors")
        if (not np.issubdtype(leaf.dtype, np.integer) or np.any(leaf < 0)
                or np.any(leaf >= self.n_nodes)):
            raise ValueError("invalid leaf IDs")
        if not np.all(self.is_leaf[leaf]):
            raise ValueError("observations must identify leaves")
        if not np.isfinite(score).all() or not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ValueError("scores and strictly positive importance weights must be finite")
        return leaf, np.clip(score, -self.config.clip, self.config.clip), weight

    def observe_normal_batch(self, leaves, scores, importance):
        leaf, score, weight = self._validate_observations(leaves, scores, importance)
        if np.any(weight > (1.0 / self.config.coverage_floor) * (1 + 1e-12)):
            raise ValueError("ordinary importance weight exceeds the coverage bound")
        if self.batch_size is not None and len(leaf) != self.batch_size:
            raise ValueError("ordinary optimizer batches must keep a constant size")
        self.batch_size = len(leaf)
        self.history.accumulate(leaf, score, weight, audit_sum=self._window_sum)
        self._window_steps += 1
        self._window_draws += self.batch_size
        self.total_normal_updates += 1
        self.total_normal_draws += self.batch_size
        self.history.finish_normal_step(self._window_steps)

    def observe_probe_history(self, leaves, scores, marginal_importance_weights):
        """Forecast-only auxiliary data; does not advance the ordinary clock."""
        leaf, score, weight = self._validate_observations(leaves, scores, marginal_importance_weights)
        self.history.accumulate(leaf, score, weight)

    def note_excluded_probe_updates(self, count, draws=0):
        for value in (count, draws):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("excluded probe counts must be nonnegative integers")
        self._excluded_probe_updates += int(count)
        self._excluded_probe_draws += int(draws)
        self.total_excluded_probe_updates += int(count)
        self.total_excluded_probe_draws += int(draws)

    def set_next_window_calibration(self, value):
        if self._window_steps or self._window_draws:
            raise RuntimeError("a saved forecast cannot be changed after its first ordinary draw")
        if not np.isfinite(value) or value < 0:
            raise ValueError("calibration must be finite and nonnegative")
        self.calibration = float(value)
        self._saved_forecast = self._forecast()
        return self._saved_forecast

    def complete_window(self):
        if self._window_steps != self.config.window_steps:
            raise RuntimeError("exactly window_steps ordinary updates are required for an audit")
        audit = audit_completed_window(
            config=self.config, saved=self._saved_forecast, reference_mass=self.mass,
            weighted_sum=self._window_sum, draws=self._window_draws, normal_steps=self._window_steps,
            excluded_steps=self._excluded_probe_updates, excluded_draws=self._excluded_probe_draws,
            calibration=self.calibration)
        self.calibration = audit.calibration_after
        self.window_index += 1
        self._saved_forecast = self._forecast()
        self._window_steps = self._window_draws = 0
        self._excluded_probe_updates = self._excluded_probe_draws = 0
        self._window_sum.fill(0)
        return BoundaryResult(audit=audit, forecast=self._saved_forecast)

    def state_dict(self):
        """Serialize partial blocks, windows, and the saved forecast."""
        state = {"version": self.STATE_VERSION, "config": asdict(self.config),
                 "parents": self.parents.tolist(), "reference_mass": self.mass.tolist()}
        for name in ("calibration", "window_index", "total_normal_updates", "total_normal_draws",
                     "total_excluded_probe_updates", "total_excluded_probe_draws", "batch_size"):
            state[name] = getattr(self, name)
        for name in ("window_steps", "window_draws", "excluded_probe_updates", "excluded_probe_draws"):
            state[name] = getattr(self, "_" + name)
        state.update(
            window_sum=self._window_sum.tolist(), block_stats=self.history.pending.tolist(),
            blocks=[{"index": b["index"], "stats": b["stats"].tolist(), "scale": b["scale"]}
                    for b in self.history.blocks], next_block_index=self.history.next_index,
            saved_forecast=forecast_to_state(self._saved_forecast),
            target="mean_over_ordinary_full_support_pre_update_states")
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != cls.STATE_VERSION:
            raise ValueError("unsupported diagnosis state version")
        restored = cls(state["parents"], state["reference_mass"], DiagnosisConfig(**state["config"]))
        for name in ("calibration", "window_index", "total_normal_updates", "total_normal_draws",
                     "total_excluded_probe_updates", "total_excluded_probe_draws", "batch_size"):
            setattr(restored, name, state[name])
        for name in ("window_steps", "window_draws", "excluded_probe_updates", "excluded_probe_draws"):
            setattr(restored, "_" + name, state[name])
        restored._window_sum = np.array(state["window_sum"], dtype=np.float64)
        restored.history.pending = np.array(state["block_stats"], dtype=np.float64)
        restored.history.blocks = [
            {"index": b["index"], "stats": np.array(b["stats"], dtype=np.float64), "scale": b["scale"]}
            for b in state["blocks"]]
        restored.history.next_index = state["next_block_index"]
        restored._saved_forecast = forecast_from_state(state["saved_forecast"])
        arrays = [restored.history.pending] + [b["stats"] for b in restored.history.blocks]
        if restored._window_sum.shape != (restored.n_nodes,) or any(
                a.shape != (restored.n_nodes, 4) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("checkpoint moment arrays do not match the tree")
        if (not np.isfinite(restored._window_sum).all()
                or restored._saved_forecast.window_index != restored.window_index):
            raise ValueError("invalid checkpoint audit state or forecast chronology")
        if any(getattr(restored._saved_forecast, name).shape != (restored.n_nodes,)
               for name in ("mean", "base", "radius", "lower", "upper", "valid", "effective_count",
                            "mixer_signal", "standard_deviation", "block_dispersion", "drift")):
            raise ValueError("checkpoint forecast shape does not match the tree")
        return restored
