# Problem Validation Contract

This document defines the validation ladder for `tensara/problems`.

## Validation Tiers

### Tier 1: Structural Validation

Runs in ordinary CI without GPUs.

Checks:

- required files exist
- frontmatter has required fields
- slug matches directory
- `def.py` has a `Problem` subclass
- required methods exist
- method signatures match the stable contract
- parameters are present in `def.py` or `get_function_signature(...)` is overridden

This tier should run on every PR.

### Tier 2: Local CUDA Validation

Runs on a real GPU such as Together H100.

Checks:

- `generate_sample()` executes
- `generate_test_cases()` executes
- `reference_solution(...)` runs on CUDA
- `verify_result(...)` accepts correct outputs
- verifier rejects perturbed outputs when enabled
- `get_flops(...)` is positive when provided

This tier is the fast author-side correctness gate.

### Tier 3: Product Runtime Validation

Runs through the same runtime path as the real Tensara product.

Authoritative target:

- Modal-backed sample/checker endpoints from `tensara/tensara`

Why this is authoritative:

- it exercises the same engine loading path used by the product
- it catches signature, allocation, and runner mismatches that local file execution can miss

This is the final acceptance gate for automation.

## Source of Truth

Validation truth should be ordered as:

1. structural CI
2. local CUDA runtime
3. Modal/product runtime

Local H100 success is necessary but not sufficient. Modal runtime is the product-truth layer.

## Standard Validator Output

Validators should emit machine-readable results:

```json
{
  "ok": true,
  "summary": {
    "problems_checked": 1,
    "errors": 0,
    "warnings": 1
  },
  "results": [
    {
      "slug": "relu",
      "ok": true,
      "diagnostics": [
        {"level": "warning", "code": "missing_source_metadata", "message": "Recommended metadata not present"}
      ]
    }
  ]
}
```

That output shape is chosen so agents can repair failures automatically.

## Required Checks For New Problems

New problems should pass:

- structural validation
- sample execution
- at least one generated test case
- wrong-answer rejection
- Modal sample or checker validation before merge

## Migration Policy

Backward compatibility matters, so the validator distinguishes:

- `error`: merge blocker
- `warning`: migration or quality issue
- `info`: useful metadata only

Existing published problems should initially be brought under structural validation first. Stronger runtime requirements can then tighten in phases.
