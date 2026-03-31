#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_FRONTMATTER = ("slug", "title", "difficulty", "author")
VALID_DIFFICULTIES = {"EASY", "MEDIUM", "HARD"}


@dataclass
class Diagnostic:
    level: str
    code: str
    message: str
    path: str | None = None


@dataclass
class ProblemResult:
    slug: str
    ok: bool = True
    diagnostics: list[Diagnostic] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str, path: Path | None = None) -> None:
        if level == "error":
            self.ok = False
        self.diagnostics.append(
            Diagnostic(
                level=level,
                code=code,
                message=message,
                path=str(path) if path else None,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Tensara problem definitions")
    parser.add_argument("targets", nargs="*", help="Problem slugs or paths under problems/")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Path to the problems repo root",
    )
    parser.add_argument(
        "--problems-dir",
        type=Path,
        help="Path to the problems directory (defaults to <repo-root>/problems)",
    )
    parser.add_argument(
        "--runtime",
        choices=("none", "sample", "first", "all"),
        default="none",
        help="Runtime validation mode",
    )
    parser.add_argument(
        "--engine-path",
        type=Path,
        help="Path to tensara/engine for runtime imports",
    )
    parser.add_argument(
        "--enforce-wrong-answer-rejection",
        action="store_true",
        help="Treat verifier acceptance of perturbed outputs as an error",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Treat warnings as errors in the final exit code",
    )
    return parser.parse_args()


def parse_frontmatter(markdown_text: str) -> tuple[dict[str, str], str]:
    if not markdown_text.startswith("---\n"):
        return {}, markdown_text

    lines = markdown_text.splitlines()
    end_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break

    if end_index is None:
        return {}, markdown_text

    frontmatter_lines = lines[1:end_index]
    content = "\n".join(lines[end_index + 1 :]).strip()
    parsed: dict[str, str] = {}

    for raw_line in frontmatter_lines:
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("\t") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")

    return parsed, content


def normalize_slug(value: str) -> str:
    return value.replace("_", "-")


def discover_problem_dirs(problems_dir: Path, targets: list[str]) -> list[Path]:
    if not targets:
        return sorted(
            path
            for path in problems_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"
        )

    resolved: list[Path] = []
    for target in targets:
        target_path = Path(target)
        if target_path.exists():
            if target_path.is_dir():
                resolved.append(target_path)
            else:
                resolved.append(target_path.parent)
            continue

        candidate = problems_dir / target
        if candidate.exists():
            resolved.append(candidate)
            continue

        raise FileNotFoundError(f"Could not resolve target: {target}")

    seen = set()
    unique: list[Path] = []
    for path in resolved:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def required_positional_after_self(fn: ast.FunctionDef) -> int:
    positional = list(fn.args.posonlyargs) + list(fn.args.args)
    if positional and positional[0].arg == "self":
        positional = positional[1:]
    required_count = max(0, len(positional) - len(fn.args.defaults))
    return required_count


def total_positional_after_self(fn: ast.FunctionDef) -> int:
    positional = list(fn.args.posonlyargs) + list(fn.args.args)
    if positional and positional[0].arg == "self":
        positional = positional[1:]
    return len(positional)


def find_problem_class(module: ast.Module) -> ast.ClassDef | None:
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Problem":
                    return node
                if isinstance(base, ast.Attribute) and base.attr == "Problem":
                    return node
    return None


def has_parameters_assignment(problem_class: ast.ClassDef) -> bool:
    for node in problem_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "parameters":
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "parameters":
                return True
    return False


def get_method_map(problem_class: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in problem_class.body
        if isinstance(node, ast.FunctionDef)
    }


def analyze_structure(problem_dir: Path) -> ProblemResult:
    slug = problem_dir.name
    result = ProblemResult(slug=slug)

    def_path = problem_dir / "def.py"
    md_path = problem_dir / "problem.md"

    if not def_path.exists():
        result.add("error", "missing_def", "Missing def.py", def_path)
        return result
    if not md_path.exists():
        result.add("error", "missing_markdown", "Missing problem.md", md_path)
        return result

    frontmatter, markdown_body = parse_frontmatter(md_path.read_text())
    if not frontmatter:
        result.add("error", "missing_frontmatter", "problem.md is missing YAML frontmatter", md_path)
    for field_name in REQUIRED_FRONTMATTER:
        if not frontmatter.get(field_name):
            result.add(
                "error",
                "missing_frontmatter_field",
                f"Missing required frontmatter field: {field_name}",
                md_path,
            )

    if markdown_body == "":
        result.add("warning", "empty_markdown_body", "problem.md body is empty", md_path)

    if frontmatter.get("slug") and frontmatter["slug"] != slug:
        if normalize_slug(frontmatter["slug"]) == slug:
            result.add(
                "warning",
                "legacy_slug_style",
                f"Frontmatter slug '{frontmatter['slug']}' should migrate to '{slug}'",
                md_path,
            )
        else:
            result.add(
                "error",
                "slug_mismatch",
                f"Frontmatter slug '{frontmatter['slug']}' does not match directory '{slug}'",
                md_path,
            )

    difficulty = frontmatter.get("difficulty")
    if difficulty and difficulty not in VALID_DIFFICULTIES:
        result.add(
            "error",
            "invalid_difficulty",
            f"Difficulty must be one of {sorted(VALID_DIFFICULTIES)}",
            md_path,
        )

    python_source = def_path.read_text()
    try:
        module = ast.parse(python_source, filename=str(def_path))
    except SyntaxError as exc:
        result.add("error", "python_syntax_error", str(exc), def_path)
        return result

    problem_class = find_problem_class(module)
    if problem_class is None:
        result.add("error", "missing_problem_class", "No class inheriting from Problem found", def_path)
        return result

    expected_class_name = slug.replace("-", "_")
    if problem_class.name != expected_class_name:
        result.add(
            "warning",
            "class_name_mismatch",
            f"Expected class name '{expected_class_name}', found '{problem_class.name}'",
            def_path,
        )

    methods = get_method_map(problem_class)
    for method_name in ("reference_solution", "generate_test_cases", "generate_sample", "verify_result"):
        if method_name not in methods:
            result.add("error", "missing_method", f"Missing required method: {method_name}", def_path)

    if "generate_test_cases" in methods and required_positional_after_self(methods["generate_test_cases"]) != 0:
        result.add(
            "error",
            "bad_generate_test_cases_signature",
            "generate_test_cases() must not require arguments beyond self",
            def_path,
        )

    if "generate_sample" in methods and required_positional_after_self(methods["generate_sample"]) != 0:
        result.add(
            "error",
            "bad_generate_sample_signature",
            "generate_sample() must not require arguments beyond self",
            def_path,
        )

    if "verify_result" in methods:
        verify_fn = methods["verify_result"]
        if required_positional_after_self(verify_fn) != 2 or total_positional_after_self(verify_fn) < 2:
            result.add(
                "error",
                "bad_verify_result_signature",
                "verify_result() must accept expected_output and actual_output after self",
                def_path,
            )

    has_parameters = has_parameters_assignment(problem_class)
    has_signature_override = "get_function_signature" in methods
    if not has_parameters and not has_signature_override:
        result.add(
            "error",
            "missing_parameters",
            "Problem must define parameters or override get_function_signature()",
            def_path,
        )

    if "tags" not in frontmatter:
        result.add(
            "info",
            "missing_tags",
            "problem.md is missing tags; current sync tolerates this, but agents benefit from tags",
            md_path,
        )

    if "source" not in frontmatter:
        result.add(
            "info",
            "missing_source_metadata",
            "Recommended source metadata is missing from problem.md frontmatter",
            md_path,
        )

    return result


def resolve_engine_path(repo_root: Path, explicit_engine_path: Path | None) -> Path | None:
    if explicit_engine_path:
        return explicit_engine_path

    sibling_engine = repo_root.parent / "tensara" / "engine"
    if sibling_engine.exists():
        return sibling_engine

    env_path = Path.cwd() / "engine"
    if env_path.exists():
        return env_path

    return None


def load_problem_instance(problem_dir: Path, engine_path: Path):
    if str(engine_path) not in sys.path:
        sys.path.insert(0, str(engine_path))

    module_name = problem_dir.name.replace("-", "_")
    def_path = problem_dir / "def.py"
    spec = importlib.util.spec_from_file_location(module_name, def_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {def_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problem_class = getattr(module, module_name)
    return problem_class()


def clone_output(value, torch_module):
    if isinstance(value, torch_module.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_output(item, torch_module) for item in value)
    if isinstance(value, list):
        return [clone_output(item, torch_module) for item in value]
    if isinstance(value, (int, float, bool)):
        return value
    raise TypeError(f"Unsupported output type for cloning: {type(value)!r}")


def perturb_output(value, torch_module):
    if isinstance(value, torch_module.Tensor):
        mutated = value.clone()
        if mutated.numel() == 0:
            raise ValueError("Cannot perturb an empty tensor output")
        flat = mutated.reshape(-1)
        delta = 1.0 if mutated.dtype.is_floating_point else 1
        flat[0] = flat[0] + delta
        return mutated
    if isinstance(value, tuple) and value:
        first = perturb_output(value[0], torch_module)
        return (first, *value[1:])
    if isinstance(value, list) and value:
        mutated = list(value)
        mutated[0] = perturb_output(mutated[0], torch_module)
        return mutated
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    raise TypeError(f"Unsupported output type for perturbation: {type(value)!r}")


def normalize_case_collection(value, *, allow_single_dict: bool) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value] if allow_single_dict else []
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return []


def validate_runtime(
    repo_root: Path,
    problem_dir: Path,
    result: ProblemResult,
    runtime_mode: str,
    engine_path: Path | None,
    enforce_wrong_answer_rejection: bool,
) -> None:
    if runtime_mode == "none":
        return

    if engine_path is None:
        result.add(
            "error",
            "missing_engine_path",
            "Runtime validation needs tensara/engine. Pass --engine-path or place a sibling tensara clone next to this repo.",
        )
        return

    import torch

    if not torch.cuda.is_available():
        result.add("error", "cuda_unavailable", "Runtime validation requires a CUDA-enabled torch environment")
        return

    try:
        problem = load_problem_instance(problem_dir, engine_path)
    except Exception as exc:  # noqa: BLE001
        result.add("error", "import_failed", f"Failed to import problem: {exc}", problem_dir / "def.py")
        return

    if getattr(problem, "name", None) != problem_dir.name:
        result.add(
            "warning",
            "name_mismatch",
            f"Problem instance name '{getattr(problem, 'name', None)}' does not match slug '{problem_dir.name}'",
            problem_dir / "def.py",
        )

    try:
        sample_cases = normalize_case_collection(problem.generate_sample(), allow_single_dict=True)
    except Exception as exc:  # noqa: BLE001
        result.add("error", "sample_generation_failed", f"generate_sample() failed: {exc}", problem_dir / "def.py")
        return

    if len(sample_cases) != 1:
        result.add(
            "error",
            "bad_sample_shape",
            "generate_sample() must return one dict or a one-element list of dicts",
            problem_dir / "def.py",
        )
        return

    try:
        test_cases = normalize_case_collection(problem.generate_test_cases(), allow_single_dict=False)
    except Exception as exc:  # noqa: BLE001
        result.add("error", "test_case_generation_failed", f"generate_test_cases() failed: {exc}", problem_dir / "def.py")
        return

    if not test_cases:
        result.add("error", "empty_test_cases", "generate_test_cases() returned no usable test cases", problem_dir / "def.py")
        return

    selected_cases = [("sample", sample_cases[0])]
    if runtime_mode == "sample":
        pass
    elif runtime_mode == "first":
        selected_cases.append(("test#1", test_cases[0]))
    else:
        selected_cases.extend((f"test#{index}", case) for index, case in enumerate(test_cases, start=1))

    executed_cases: list[str] = []
    for case_name, case in selected_cases:
        if "name" not in case or not isinstance(case["name"], str):
            result.add("error", "case_missing_name", f"{case_name} is missing a string 'name' field", problem_dir / "def.py")
            return
        if "create_inputs" not in case or not callable(case["create_inputs"]):
            result.add("error", "case_missing_factory", f"{case_name} is missing callable create_inputs", problem_dir / "def.py")
            return

        try:
            raw_inputs = case["create_inputs"]()
            if isinstance(raw_inputs, tuple):
                inputs = raw_inputs
            elif isinstance(raw_inputs, list):
                inputs = tuple(raw_inputs)
            else:
                inputs = (raw_inputs,)
            expected = problem.reference_solution(*inputs)
            correct = clone_output(expected, torch)
            correct_ok, correct_info = problem.verify_result(expected, correct)
        except Exception as exc:  # noqa: BLE001
            result.add("error", "runtime_case_failed", f"{case_name} failed during runtime validation: {exc}", problem_dir / "def.py")
            return

        if not correct_ok:
            result.add(
                "error",
                "verifier_rejected_reference",
                f"{case_name} verifier rejected the reference output: {correct_info}",
                problem_dir / "def.py",
            )
            return

        try:
            wrong = perturb_output(expected, torch)
            wrong_ok, wrong_info = problem.verify_result(expected, wrong)
            if wrong_ok:
                level = "error" if enforce_wrong_answer_rejection else "warning"
                result.add(
                    level,
                    "verifier_accepted_perturbed_output",
                    f"{case_name} verifier accepted an intentionally perturbed output: {wrong_info}",
                    problem_dir / "def.py",
                )
        except Exception as exc:  # noqa: BLE001
            result.add(
                "warning",
                "wrong_answer_check_failed",
                f"{case_name} wrong-answer rejection check could not run: {exc}",
                problem_dir / "def.py",
            )

        try:
            flops = problem.get_flops(case)
            if flops is not None and flops <= 0:
                result.add("error", "non_positive_flops", f"{case_name} reported non-positive FLOPs: {flops}", problem_dir / "def.py")
                return
        except Exception as exc:  # noqa: BLE001
            result.add("warning", "flops_check_failed", f"{case_name} FLOPs check failed: {exc}", problem_dir / "def.py")

        executed_cases.append(case["name"])

    result.runtime = {
        "mode": runtime_mode,
        "engine_path": str(engine_path),
        "executed_cases": executed_cases,
    }


def summarize(results: list[ProblemResult], warnings_as_errors: bool) -> tuple[dict[str, int], bool]:
    summary = {
        "problems_checked": len(results),
        "errors": sum(1 for result in results for diag in result.diagnostics if diag.level == "error"),
        "warnings": sum(1 for result in results for diag in result.diagnostics if diag.level == "warning"),
        "infos": sum(1 for result in results for diag in result.diagnostics if diag.level == "info"),
    }
    ok = summary["errors"] == 0 and (not warnings_as_errors or summary["warnings"] == 0)
    return summary, ok


def print_text(results: list[ProblemResult], summary: dict[str, int]) -> None:
    for result in results:
        status = "ok" if result.ok else "failed"
        print(f"[{status}] {result.slug}")
        for diag in result.diagnostics:
            if diag.level == "info":
                continue
            path_suffix = f" ({diag.path})" if diag.path else ""
            print(f"  - {diag.level.upper()} {diag.code}: {diag.message}{path_suffix}")
        if result.runtime:
            print(f"  - runtime: mode={result.runtime['mode']} cases={', '.join(result.runtime['executed_cases'])}")
    print()
    print(
        "Summary: "
        f"{summary['problems_checked']} problems, "
        f"{summary['errors']} errors, "
        f"{summary['warnings']} warnings, "
        f"{summary['infos']} infos"
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    problems_dir = (args.problems_dir or repo_root / "problems").resolve()

    results: list[ProblemResult] = []
    for problem_dir in discover_problem_dirs(problems_dir, args.targets):
        result = analyze_structure(problem_dir)
        validate_runtime(
            repo_root=repo_root,
            problem_dir=problem_dir,
            result=result,
            runtime_mode=args.runtime,
            engine_path=resolve_engine_path(repo_root, args.engine_path.resolve() if args.engine_path else None),
            enforce_wrong_answer_rejection=args.enforce_wrong_answer_rejection,
        )
        results.append(result)

    summary, ok = summarize(results, warnings_as_errors=args.warnings_as_errors)
    payload = {
        "ok": ok,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print_text(results, summary)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
