"""Per-method preparation entry. Inspect is read-only and cannot launch training.

The endpoint controller and extracted scientific kernels are independently
callable. A runtime may be enabled only after its preparation receipt and
backend are built and qualified; this CLI makes pending status explicit.
"""
import argparse
import json
from method_registry import inspect, METHODS


def main(default_method=None):
    parser = argparse.ArgumentParser()
    if default_method is None:
        parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", required=True, choices=("BY", "Public433"))
    parser.add_argument("--ratio", required=True, type=int, choices=(5, 10))
    parser.add_argument("--action", choices=("inspect",), default="inspect")
    args = parser.parse_args()
    print(json.dumps(inspect(default_method or args.method, args.dataset, args.ratio),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
