"""Budgeted tree-cut search and partition updates."""
from __future__ import annotations

import copy
import math

import numpy as np

from data import ReferenceTree
from handoff import probability_vector, solve_handoff


def exact_tree_dp(tree, log_A, valid, budget):
    """Find the best budgeted cut, adding node scores in log space."""
    scores = np.asarray(log_A, dtype=np.float64)
    eligible = np.asarray(valid, dtype=bool)
    if scores.shape != (tree.n,) or eligible.shape != (tree.n,):
        raise ValueError("scores and eligibility must cover the entire fixed tree")
    if isinstance(budget, bool) or int(budget) != budget or budget < 1:
        raise ValueError("group budget must be a positive integer")
    eligible = eligible & np.isfinite(scores)

    def subtree(node):
        alternatives = {}
        if eligible[node]:
            alternatives[1] = (float(scores[node]), node)
        children = tree.children[node]
        if children:
            joined = {0: (-math.inf, None)}
            for child in children:
                child_options = subtree(child)
                next_join = {}
                for used, (left_score, left_trace) in joined.items():
                    for extra, (right_score, right_trace) in child_options.items():
                        total = used + extra
                        if total > budget:
                            continue
                        merged = float(np.logaddexp(left_score, right_score))
                        previous = next_join.get(total)
                        if previous is None or merged > previous[0]:
                            next_join[total] = (merged, (left_trace, right_trace))
                joined = next_join
            for count, option in joined.items():
                if count not in alternatives or option[0] > alternatives[count][0]:
                    alternatives[count] = option
        return alternatives

    options = subtree(tree.root)
    if not options:
        return None
    count = max(options, key=lambda k: (options[k][0], -k))
    value, trace = options[count]
    cut = []

    def recover(pointer):
        if pointer is None:
            return
        if isinstance(pointer, int):
            cut.append(pointer)
        else:
            recover(pointer[0])
            recover(pointer[1])

    recover(trace)
    return {"cut": cut, "log_total": float(value), "groups": count}


class GranMixController:
    def __init__(self, children, reference_mass, initial_cut, initial_weights=None, *,
                 eta=.01, mixer_eta=0., budget=32, epsilon=.1, delta=.001, tolerance=1e-10):
        self.tree = ReferenceTree(children, reference_mass)
        if (not np.isfinite([eta, mixer_eta, epsilon, tolerance]).all()
                or eta <= 0 or mixer_eta < 0 or not 0 <= epsilon < 1 or tolerance <= 0
                or isinstance(budget, bool) or int(budget) != budget or budget < 1
                or (delta is not None and (math.isnan(delta) or delta < 0))):
            raise ValueError("invalid controller settings")
        self.config = {"eta": float(eta), "mixer_eta": float(mixer_eta), "budget": int(budget),
                       "epsilon": float(epsilon),
                       "delta": None if delta is None or math.isinf(delta) else float(delta),
                       "tolerance": float(tolerance)}
        self.cut = self.tree.validate_cut(initial_cut)
        initial = self.tree.mass[self.cut] if initial_weights is None else initial_weights
        self.weights = probability_vector(initial)
        self._check_sampler(self.cut, self.weights)
        self.decisions = 0
        self.last_report = None

    @property
    def q(self):
        return self.weights.copy()

    def _check_sampler(self, cut, weights):
        if len(cut) > self.config["budget"]:
            raise ValueError("cut exceeds the configured group budget")
        self.tree.sample_distribution(cut, weights)
        if np.any(weights < self.config["epsilon"] * self.tree.mass[cut] - 1e-12):
            raise ValueError("sampler violates the reference coverage floor")

    def sample_distribution(self):
        return self.tree.sample_distribution(self.cut, self.weights)

    def update(self, forecast, *, allow_structure=True):
        def field(name, dtype):
            value = forecast[name] if isinstance(forecast, dict) else getattr(forecast, name)
            array = np.asarray(value, dtype=dtype)
            if array.shape != (self.tree.n,):
                raise ValueError("forecast fields must contain every fixed-tree node")
            return array

        mean, radius = field("mean", float), field("radius", float)
        valid, signal = field("valid", bool), field("mixer_signal", float)
        valid = valid & np.isfinite(mean) & np.isfinite(radius) & (radius >= 0) & np.isfinite(signal)
        old_cut, old_weights = list(self.cut), self.weights.copy()
        previous_sampler = self.tree.sample_distribution(old_cut, old_weights)
        report = {
            "decision_index": self.decisions + 1, "decision": "keep",
            "changed": False, "updated": False, "reason": "invalid_current_node",
            "cut_before": old_cut, "weights_before": old_weights.tolist(),
            "cut_after": old_cut.copy(), "weights_after": old_weights.tolist(),
            "proposals": [], "selected": None,
            "invalid_current": [node for node in old_cut if not valid[node]],
            "current_upper": None, "dp": None,
        }
        if not report["invalid_current"]:
            lower, upper = mean - radius, mean + radius
            inherited = self.tree.kappa(old_cut, old_weights)
            rate = self.config["eta"]
            previous_upper = float(old_weights @ upper[old_cut])
            log_scores = np.full(self.tree.n, -math.inf)
            log_scores[valid] = (np.log(self.tree.mass[valid]) + inherited[valid] + rate * lower[valid])
            searched = (exact_tree_dp(self.tree, log_scores, valid, self.config["budget"])
                        if allow_structure else None)
            report.update(current_upper=previous_upper, dp=searched)
            candidates = [("current", old_cut)]
            if searched is not None and set(searched["cut"]) != set(old_cut):
                candidates.append(("tree_dp", searched["cut"]))
            winner = None
            for name, cut in candidates:
                execution = solve_handoff(
                    self.tree.mass[cut], inherited[cut], lower[cut], previous_upper, rate,
                    mixer_signal=signal[cut], mixer_eta=self.config["mixer_eta"],
                    epsilon=self.config["epsilon"], delta=self.config["delta"],
                    tolerance=self.config["tolerance"])
                proposal = {"name": name, "cut": list(cut), **execution}
                report["proposals"].append(proposal)
                if execution["feasible"]:
                    trial = self.tree.sample_distribution(cut, execution["w_exec"])
                    sample_kl = float(trial @ np.log(trial / previous_sampler))
                    if not math.isclose(sample_kl, execution["full_kl"], abs_tol=1e-9, rel_tol=1e-8):
                        raise ArithmeticError("projected KL disagrees with the induced sample law")
                    proposal["sample_kl_verified"] = sample_kl
                if execution["accepted"] and (
                        winner is None or execution["Cactual"] > winner["Cactual"] + self.config["tolerance"]):
                    winner = proposal
            if winner is None:
                report["reason"] = "no_positive_feasible_proposal"
            else:
                installed_cut = list(winner["cut"])
                installed_weights = np.asarray(winner["w_exec"], dtype=float)
                self._check_sampler(installed_cut, installed_weights)
                self.cut, self.weights = installed_cut, installed_weights.copy()
                structure_changed = set(installed_cut) != set(old_cut)
                report.update(
                    decision="repartition" if structure_changed else "reweight",
                    reason="positive_actual_score", changed=structure_changed,
                    updated=structure_changed or not np.array_equal(installed_weights, old_weights),
                    cut_after=installed_cut, weights_after=installed_weights.tolist(),
                    selected=winner["name"], Cactual=winner["Cactual"],
                    execution_regret=winner["execution_regret"])
        self.decisions += 1
        self.last_report = copy.deepcopy(report)
        return report

    def state_dict(self):
        return {
            "version": 1, "children": [list(row) for row in self.tree.children],
            "reference_mass": self.tree.mass.tolist(), "config": copy.deepcopy(self.config),
            "cut": list(self.cut), "weights": self.weights.tolist(),
            "decisions": self.decisions, "last_report": copy.deepcopy(self.last_report),
        }

    def load_state_dict(self, state):
        if (state.get("version") != 1
                or state.get("children") != [list(row) for row in self.tree.children]
                or not np.array_equal(state.get("reference_mass"), self.tree.mass)
                or state.get("config") != self.config):
            raise ValueError("controller checkpoint belongs to a different tree or configuration")
        cut = self.tree.validate_cut(state["cut"])
        weights = probability_vector(state["weights"])
        self._check_sampler(cut, weights)
        decisions = state["decisions"]
        if isinstance(decisions, bool) or not isinstance(decisions, int) or decisions < 0:
            raise ValueError("invalid decision counter")
        self.cut, self.weights, self.decisions = cut, weights, decisions
        self.last_report = copy.deepcopy(state.get("last_report"))
