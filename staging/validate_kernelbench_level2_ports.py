#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path


DEFAULT_SLUGS = [
    "gemm-relu-divide",
    "conv2d-divide-leaky-relu",
    "conv2d-hardswish-relu",
    "matmul-mish-mish",
]


def convert_slug_to_module_name(slug: str) -> str:
    return slug.replace("-", "_")


def load_problem(slug: str):
    repo_root = Path(__file__).resolve().parents[1]
    tensara_engine = repo_root.parents[0] / "tensara" / "engine"
    if str(tensara_engine) not in sys.path:
        sys.path.insert(0, str(tensara_engine))

    problem_path = repo_root / "problems" / slug / "def.py"
    if not problem_path.exists():
        raise FileNotFoundError(f"Problem definition not found: {problem_path}")

    module_name = convert_slug_to_module_name(slug)
    spec = importlib.util.spec_from_file_location(module_name, problem_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {problem_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problem_class = getattr(module, module_name)
    return problem_class()


def perturb_tensor(tensor):
    bad = tensor.clone()
    flat = bad.reshape(-1)
    delta = 1.0 if tensor.dtype.is_floating_point else 1
    flat[0] = flat[0] + delta
    return bad


def validate_case(problem, case_name: str, case: dict, reject_wrong: bool) -> None:
    inputs = case["create_inputs"]()
    expected = problem.reference_solution(*inputs)

    correct_ok, correct_info = problem.verify_result(expected, expected.clone())
    if not correct_ok:
        raise AssertionError(
            f"{problem.name} {case_name}: verifier rejected reference output: {correct_info}"
        )

    if reject_wrong:
        wrong = perturb_tensor(expected)
        wrong_ok, wrong_info = problem.verify_result(expected, wrong)
        if wrong_ok:
            raise AssertionError(
                f"{problem.name} {case_name}: verifier accepted intentionally wrong output: {wrong_info}"
            )

    flops = problem.get_flops(case)
    if flops is not None and flops <= 0:
        raise AssertionError(f"{problem.name} {case_name}: non-positive FLOPs: {flops}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local KernelBench Level 2 Tensara ports")
    parser.add_argument("slugs", nargs="*", default=DEFAULT_SLUGS)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all generated test cases instead of only the first one plus sample",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available in this Python environment.", file=sys.stderr)
        return 2

    for slug in args.slugs:
        problem = load_problem(slug)
        print(f"[validate] {slug}")

        sample = problem.generate_sample()
        validate_case(problem, "sample", sample, reject_wrong=True)
        print("  sample: ok")

        test_cases = problem.generate_test_cases()
        selected_cases = test_cases if args.all else test_cases[:1]
        for index, case in enumerate(selected_cases, start=1):
            validate_case(problem, f"test#{index}", case, reject_wrong=(index == 1))
            print(f"  test#{index}: ok ({case['name']})")

        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
