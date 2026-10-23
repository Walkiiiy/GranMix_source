"""Mixture projection under coverage and KL constraints."""
from __future__ import annotations

import math

import numpy as np


def logsumexp(values):
    data = np.asarray(values, dtype=np.float64)
    largest = float(data.max())
    if largest == -math.inf:
        return largest
    return largest + math.log(float(np.exp(data - largest).sum()))


def probability_vector(values, name="weights"):
    data = np.array(values, dtype=np.float64, copy=True)
    if (data.ndim != 1 or not data.size or not np.isfinite(data).all()
            or np.any(data <= 0) or not np.isclose(data.sum(), 1., atol=1e-10, rtol=0)):
        raise ValueError(f"{name} must be a positive probability vector")
    return data / data.sum()


def _vector(values, shape, name):
    data = np.array(values, dtype=np.float64, copy=True)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"{name} must be a finite vector of shape {shape}")
    return data


def floor_normalize(log_base, floor):
    """Solve w_i=max(floor_i, c*exp(log_base_i)) in log space."""
    logits = np.asarray(log_base, dtype=np.float64)
    minimum = _vector(floor, logits.shape, "floor")
    if (logits.ndim != 1 or not logits.size or not np.isfinite(logits).all()
            or np.any(minimum < 0) or minimum.sum() >= 1):
        raise ValueError("positive free mass and finite one-dimensional logits are required")
    threshold = np.full_like(minimum, -math.inf)
    positive = minimum > 0
    threshold[positive] = np.log(minimum[positive]) - logits[positive]
    order = np.argsort(-threshold, kind="stable")
    used_mass = np.r_[0., np.cumsum(minimum[order])]
    remaining_log_sums = np.logaddexp.accumulate(logits[order[::-1]])[::-1]
    fixed_count = 0
    for candidate in range(len(order)):
        free_mass = 1.0 - float(used_mass[candidate])
        log_multiplier = math.log(free_mass) - float(remaining_log_sums[candidate])
        fixed_count = candidate
        if log_multiplier >= threshold[order[candidate]]:
            break
    free = order[fixed_count:]
    answer = minimum.copy()
    answer[free] = free_mass * np.exp(logits[free] - remaining_log_sums[fixed_count])
    anchor = int(free[np.argmax(answer[free])])
    answer[anchor] += 1.0 - math.fsum(answer.tolist())
    if np.any(answer < minimum - 1e-14):
        raise ArithmeticError("normalization failed its floor constraint")
    if np.any(answer <= 0):
        raise FloatingPointError("mixture underflowed to zero")
    return answer


def _relative_entropy(weights, log_reference):
    value = math.fsum((weights * (np.log(weights) - log_reference)).tolist())
    if value < -1e-9:
        raise ArithmeticError("KL is negative for the supplied reference")
    return max(value, 0.)


def solve_handoff(p_group, kappa, lower, old_upper, eta, *,
                  mixer_signal=None, mixer_eta=0., epsilon=.1,
                  delta=.001, tolerance=1e-10, max_iterations=100):
    p = probability_vector(p_group, "group reference masses")
    inherited = _vector(kappa, p.shape, "inherited log preference")
    lo = _vector(lower, p.shape, "lower contribution bounds")
    signal = np.zeros_like(p) if mixer_signal is None else _vector(mixer_signal, p.shape, "mixer signal")
    rate, mixer_rate = float(eta), float(mixer_eta)
    floor_fraction, tol, previous_upper = float(epsilon), float(tolerance), float(old_upper)
    kl_limit = math.inf if delta is None else float(delta)
    if (not np.isfinite([rate, mixer_rate, floor_fraction, tol, previous_upper]).all()
            or rate <= 0 or mixer_rate < 0 or not 0 <= floor_fraction < 1 or tol <= 0
            or math.isnan(kl_limit) or kl_limit < 0
            or not isinstance(max_iterations, int) or max_iterations < 16):
        raise ValueError("invalid handoff settings")
    inherited_logits = np.log(p) + inherited
    log_Z = logsumexp(inherited_logits)
    if log_Z > 1e-10:
        raise ValueError("inherited preference implies Z > 1")
    log_Z = min(log_Z, 0.)
    log_reference = inherited_logits - logsumexp(inherited_logits)
    shift = float(lo.max())
    centered_scores = inherited_logits + rate * (lo - shift)
    if not np.isfinite(centered_scores).all():
        raise FloatingPointError("target contribution tilt overflowed")
    score_sum = logsumexp(centered_scores)
    log_optimum = centered_scores - score_sum
    mixed = log_optimum + mixer_rate * (signal - float(signal.max()))
    if not np.isfinite(mixed).all():
        raise FloatingPointError("mixer tilt overflowed")
    log_target = mixed - logsumexp(mixed)
    floors = floor_fraction * p
    optimum_gain = math.fsum([shift, score_sum / rate, -previous_upper])
    if not math.isfinite(optimum_gain):
        raise FloatingPointError("optimal conservative gain overflowed")

    def evaluate(alpha):
        weights = floor_normalize((1 - alpha) * log_reference + alpha * log_target, floors)
        reference_kl = _relative_entropy(weights, log_reference)
        full_kl = reference_kl - log_Z
        regret = _relative_entropy(weights, log_optimum) / rate
        gain = math.fsum([float(weights @ lo), -previous_upper, -full_kl / rate])
        return {"weights": weights, "alpha": float(alpha), "full_kl": float(full_kl),
                "kl_to_r0": float(reference_kl), "execution_regret": float(regret),
                "Cactual": float(gain), "score_identity_error": abs(gain - (optimum_gain - regret))}

    least_change = evaluate(0.)
    report = {
        "feasible": False, "accepted": False, "reason": "infeasible_kl_budget",
        "epsilon": floor_fraction, "delta": None if math.isinf(kl_limit) else kl_limit,
        "tolerance": tol, "structural_kl": float(-log_Z), "log_Z": float(log_Z),
        "r0": np.exp(log_reference).tolist(), "floor": floors.tolist(),
        "w_star": np.exp(log_optimum).tolist(), "w_target": np.exp(log_target).tolist(),
        "log_w_star": log_optimum.tolist(), "log_target": log_target.tolist(),
        "Cstar": float(optimum_gain), "min_feasible_kl": least_change["full_kl"],
        "alpha": None, "w_exec": None, "Cactual": None, "execution_regret": None,
        "full_kl": None, "score_identity_error": None, "iterations": 0,
    }
    if least_change["full_kl"] > kl_limit + tol:
        return report
    target = evaluate(1.)
    chosen = target
    if target["full_kl"] > kl_limit + tol:
        feasible_alpha, infeasible_alpha = 0., 1.
        chosen = least_change
        for iteration in range(max_iterations):
            middle = (feasible_alpha + infeasible_alpha) / 2
            point = evaluate(middle)
            if point["full_kl"] <= kl_limit:
                feasible_alpha, chosen = middle, point
            else:
                infeasible_alpha = middle
            report["iterations"] = iteration + 1
            if infeasible_alpha - feasible_alpha <= 2e-14:
                break
    weights = chosen.pop("weights")
    if np.any(weights < floors - 1e-13) or chosen["full_kl"] > kl_limit + tol:
        raise ArithmeticError("handoff result violates its execution constraints")
    if chosen["score_identity_error"] > 1e-7 * max(1., abs(optimum_gain)):
        raise ArithmeticError("executed gain differs from optimum minus execution regret")
    report.update(chosen)
    accepted = chosen["Cactual"] > tol
    report.update(
        feasible=True, accepted=accepted,
        reason="positive_actual_score" if accepted else "nonpositive_actual_score",
        w_exec=weights.tolist(),
        floor_active=int(np.count_nonzero(np.isclose(weights, floors, atol=1e-12, rtol=0))),
        kl_active=bool(math.isfinite(kl_limit)
                       and abs(chosen["full_kl"] - kl_limit) <= max(tol, 1e-12)))
    return report
