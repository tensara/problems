# Problem Authoring Contract

This document defines the backward-compatible authoring format for `tensara/problems`.

## Goals

- Make problem authoring deterministic for agents.
- Preserve compatibility with existing published problems.
- Separate authoring truth from runtime truth.
- Keep `sync-problems.ts` compatible while stricter validation is rolled out.

## Stable Problem Layout

Each problem lives in:

```text
problems/<slug>/
├── def.py
└── problem.md
```

Both files remain required.

## `problem.md` Contract

### Required Frontmatter

These fields are already expected by sync and remain required:

```yaml
slug: "relu"
title: "ReLU"
difficulty: "EASY"
author: "sarthak"
```

### Recommended Frontmatter

These are now the recommended fields for new problems. They are backward-compatible because unknown frontmatter is ignored by the current sync path.

```yaml
tags: ["activation-function"]
source:
  kind: "kernelbench"
  repo: "ScalingIntelligence/KernelBench"
  path: "KernelBench/level2/63_Gemm_ReLU_Divide.py"
authoring:
  mode: "exact-port"   # exact-port | normalized
validation:
  deterministic: true
  sample_path: true
  wrong_answer_rejection: true
  runtime_targets: ["local-cuda", "modal-sample"]
```

### Content Rules

- `slug` must match the directory name.
- `difficulty` must be `EASY`, `MEDIUM`, or `HARD`.
- Markdown body should describe the mathematical contract, not implementation trivia.
- If the problem is adapted from an external source, include attribution in the body or `source` block.

## `def.py` Contract

### Required Class Shape

`def.py` must define one primary problem class that subclasses `Problem`.

Canonical class naming:

- directory slug: `gemm-relu-divide`
- class name: `gemm_relu_divide`

### Required Methods

New problems should implement:

- `reference_solution(self, *args)`
- `generate_test_cases(self)`
- `generate_sample(self)`
- `verify_result(self, expected_output, actual_output)`

Backward-compatible rule:

- existing problems are allowed to keep current behavior if they already work
- new validation treats extra required args on `generate_test_cases`, `generate_sample`, or `verify_result` as contract errors

### Parameters

Preferred:

- define `parameters = [...]` in `def.py`

Fallback still supported:

- override `get_function_signature(...)`
- include legacy `parameters` frontmatter in `problem.md`

New problems should define parameters in `def.py`, because that is the most machine-readable source for agents and validators.

### Test Case Shape

`generate_test_cases()` should return a list of dicts.

Each dict should contain:

- `name`: stable string label
- `create_inputs`: zero-arg callable returning the reference inputs

Additional keys such as dimensions, seed, or descriptive metadata are encouraged.

### Sample Shape

Canonical form:

- `generate_sample()` returns a single dict with the same shape as one test case

Backward-compatible form still accepted by the validator:

- a list containing exactly one dict

### Verifier Rules

`verify_result(...)` should:

- accept the exact reference output
- reject intentionally perturbed outputs
- return `(bool, debug_info)`
- provide debug info that is useful for automated repair

## Canonical Authoring Pattern

1. Define the mathematical contract in `problem.md`.
2. Define parameters in `def.py`.
3. Make `reference_solution(...)` deterministic.
4. Use `Problem.get_seed(...)` for seeded test generation.
5. Add one small sample case.
6. Add several larger generated cases.
7. Make verifier failures explicit and debuggable.

## Compatibility Policy

This contract is intentionally additive:

- existing frontmatter fields keep working
- old problems are not forced to adopt new optional metadata immediately
- validation distinguishes structural errors from migration warnings

The long-term direction is to move all new problems toward this contract and then tighten sync around it.
