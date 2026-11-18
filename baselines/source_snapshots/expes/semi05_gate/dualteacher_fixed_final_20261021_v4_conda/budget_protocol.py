"""Prospective primary-sample budget; no training, evaluation or dispatch.

This matches primary 3D labeled/unlabeled inputs during the SSL stage only.
Initialization, auxiliary donors, teacher/SAM forwards and GPU time are separate
accounting dimensions. Matching this budget does NOT establish equal compute.
"""
from dataclasses import asdict, dataclass
import json


PROTOCOL = "PRIMARY_3D_SSL_EXPOSURE_V1_20261019"


@dataclass(frozen=True)
class Budget:
    dataset: str
    ratio: int
    labeled_cases: int
    unlabeled_cases: int
    cycles: int = 36
    reference_accumulation: int = 5

    def __post_init__(self):
        for value in (self.ratio, self.labeled_cases, self.unlabeled_cases,
                      self.cycles, self.reference_accumulation):
            if type(value) is not int or value <= 0:
                raise ValueError("Positive integer budget fields required")

    @property
    def primary_pairs_per_cycle(self):
        n, k = self.unlabeled_cases, self.reference_accumulation
        return ((n + k - 1) // k) * k

    @property
    def replay_per_cycle(self):
        return self.primary_pairs_per_cycle - self.unlabeled_cases

    @property
    def primary_pairs(self):
        return self.cycles * self.primary_pairs_per_cycle

    @property
    def hpq_reference_updates(self):
        return self.primary_pairs // self.reference_accumulation

    def iteration_plan(self, labeled_per_iteration, unlabeled_per_iteration):
        """Plan native iterations without silently dropping/overdrawing inputs.

        Returns a remainder explicitly. Each method must validate its own tail
        handling, stage mapping and optimizer schedule before dispatch.
        """
        for value in (labeled_per_iteration, unlabeled_per_iteration):
            if type(value) is not int or value <= 0:
                raise ValueError("Actual primary input counts must be positive ints")
        if labeled_per_iteration != unlabeled_per_iteration:
            raise ValueError("Unequal stream sizes require an audited adapter")
        q, r = divmod(self.primary_pairs, unlabeled_per_iteration)
        return {"full_iterations": q, "remaining_labeled_inputs": r,
                "remaining_unlabeled_inputs": r,
                "optimizer_updates": None,
                "runtime_verified": False}

    def record(self):
        return dict(asdict(self),
                    primary_labeled_inputs=self.primary_pairs,
                    primary_unlabeled_inputs=self.primary_pairs,
                    replay_per_cycle=self.replay_per_cycle,
                    hpq_reference_updates=self.hpq_reference_updates,
                    scope="SSL primary 3D inputs only; not total compute",
                    historical_hpq_exists=not(self.dataset == "Public433" and self.ratio == 10),
                    new_baseline_dispatched=False)


BUDGETS = (
    Budget("BY", 5, 5, 95),
    Budget("BY", 10, 10, 90),
    Budget("Public433", 5, 17, 329),
    Budget("Public433", 10, 35, 311),
)


if __name__ == "__main__":
    print(json.dumps({"protocol": PROTOCOL,
                      "selection_uses_test_scores": False,
                      "early_stop_for_performance": False,
                      "budgets": [b.record() for b in BUDGETS]},
                     indent=2, ensure_ascii=False))
