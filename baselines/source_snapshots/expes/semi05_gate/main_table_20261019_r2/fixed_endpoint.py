"""A single-test terminal-checkpoint controller, independent of training backend.

Training adapters must provide measured receipts. A failed/partial attempt is
never silently resumed or evaluated. Resource-locked save/reload and actual
torch/data construction belong to the separately qualified backend.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exclusive_json(path, value):
    # Never overwrite evidence, including an incomplete test attempt.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class Endpoint:
    method: str
    labeled_inputs: int
    unlabeled_inputs: int
    optimizer_updates_per_iteration: int
    test_case_ids: tuple
    protocol: str = "PRIMARY_3D_SSL_EXPOSURE_V1_20261019"

    def validate(self):
        for n in (self.labeled_inputs, self.unlabeled_inputs, self.optimizer_updates_per_iteration):
            if type(n) is not int or n <= 0:
                raise ValueError("Invalid endpoint count")
        if not self.test_case_ids or len(set(self.test_case_ids)) != len(self.test_case_ids):
            raise ValueError("Missing/duplicate actual test cases")
        if not all(isinstance(x, str) and x for x in self.test_case_ids):
            raise ValueError("Invalid test case identity")


class FixedFinalRun:
    def __init__(self, endpoint, output):
        endpoint.validate()
        self.endpoint, self.output = endpoint, Path(output)
        self.counts = dict(labeled_inputs=0, unlabeled_inputs=0,
                           iterations=0, successful_optimizer_updates=0)

    def run(self, backend):
        """Backend evaluation is called exactly once after terminal reload.

        backend.prepare() must verify pinned source/input/init identities and
        create only the training data. backend.test() gets its exclusive permit
        after all budgets and strict reload verification have completed.
        """
        self.output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        intent = dict(endpoint=asdict(self.endpoint), status="INTENT", exact_resume_supported=False)
        exclusive_json(self.output / "run_intent.json", intent)
        try:
            preparation = backend.prepare(self.endpoint)
            for required in ("source_pins_verified", "input_pins_verified", "initialization_verified",
                             "resource_gate_passed", "test_unopened"):
                if preparation.get(required) is not True:
                    raise ValueError("Backend preparation missing: " + required)
            exclusive_json(self.output / "preparation_verified.json", preparation)
            while (self.counts["labeled_inputs"] < self.endpoint.labeled_inputs or
                   self.counts["unlabeled_inputs"] < self.endpoint.unlabeled_inputs):
                remaining = {k: getattr(self.endpoint, k) - self.counts[k]
                             for k in ("labeled_inputs", "unlabeled_inputs")}
                step = backend.train_iteration(remaining)
                self.accept_step(step)
                with (self.output / "steps.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(step=step, cumulative=dict(self.counts)),
                                            allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            coverage = backend.verify_training_coverage()
            if coverage.get("actual_case_coverage_verified") is not True:
                raise ValueError("Actual training case coverage not verified")
            exclusive_json(self.output / "coverage_verified.json", coverage)
            artifact = backend.save_terminal(dict(self.counts))
            if artifact.get("resource_lock_verified") is not True:
                raise ValueError("Terminal save did not verify shared resource lock")
            if sha_file(artifact["path"]) != artifact["sha256"]:
                raise ValueError("Terminal artifact SHA differs")
            reload = backend.strict_reload(artifact)
            if (reload.get("strict") is not True or reload.get("sha256") != artifact["sha256"]
                    or reload.get("native_student_identity_verified") is not True):
                raise ValueError("Terminal strict reload/identity not verified")
            permit = dict(selection="terminal_after_fixed_exposure", final_sha256=artifact["sha256"],
                          test_case_ids=list(self.endpoint.test_case_ids), test_evaluation=1)
            permit_path = self.output / "final_test_intent.json"
            exclusive_json(permit_path, permit)
            result = backend.test(artifact, permit_path)
            if result.get("case_ids") != list(self.endpoint.test_case_ids):
                raise ValueError("Test order/unique case coverage differs")
            if result.get("final_sha256") != artifact["sha256"]:
                raise ValueError("Test result bound to wrong final")
            values = result.get("case_dice")
            if (not isinstance(values, list) or len(values) != len(self.endpoint.test_case_ids)
                    or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
                           for v in values)):
                raise ValueError("Invalid per-case Dice")
            mean = sum(values) / len(values)
            if not math.isclose(float(result["dice"]), mean, abs_tol=1e-12, rel_tol=0):
                raise ValueError("Case aggregation differs")
            if sha_file(artifact["path"]) != artifact["sha256"]:
                raise ValueError("Artifact changed during test")
            record = dict(status="COMPLETE_VERIFIED", counts=dict(self.counts), artifact=artifact,
                          final_metrics=result, test_evaluations=1,
                          wall_seconds=time.perf_counter() - started,
                          comparison_scope="SSL primary inputs; initialization/aux compute reported separately")
            exclusive_json(self.output / "final_metrics.json", record)
            return record
        except BaseException as error:
            exclusive_json(self.output / "failure.json", dict(status="FAILED_PRESERVED",
                           counts=dict(self.counts), error_type=type(error).__name__, error=str(error)))
            raise

    def accept_step(self, step):
        if (step.get("finite_loss") is not True or step.get("finite_nonzero_gradients") is not True
                or step.get("teacher_no_grad") is not True or step.get("skip_count") != 0
                or step.get("actual_case_ids_verified") is not True
                or step.get("unlabeled_gt_in_loss") is not False):
            raise ValueError("Invalid actual training receipt")
        for k in ("labeled_inputs", "unlabeled_inputs"):
            if type(step.get(k)) is not int or step[k] <= 0:
                raise ValueError("Invalid actual sample count")
            if self.counts[k] + step[k] > getattr(self.endpoint, k):
                raise ValueError("Budget overrun: " + k)
        if (type(step.get("successful_optimizer_updates")) is not int
                or step["successful_optimizer_updates"] != self.endpoint.optimizer_updates_per_iteration):
            raise ValueError("Optimizer skip/extra update")
        self.counts["iterations"] += 1
        self.counts["successful_optimizer_updates"] += step["successful_optimizer_updates"]
        for k in ("labeled_inputs", "unlabeled_inputs"):
            self.counts[k] += step[k]
