# Problems
This repository contains problems hosted on [Tensara](https://tensara.org/). Our immediate goal is to port over all [KernelBench Level 1 and 2 challenges](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench). 

## Local Setup
Use the same database URL from the initial setup of the [original repository](https://github.com/tensara/tensara). Create an `.env` file and add the following:
```
DATABASE_URL="<your database url>"
```

Then, you can run:
```
pnpm i
pnpm prisma generate
pnpm sync-problems
```

This will take contents from the `problems/` folder and sync it with the database. You should be able to see the changes in your local instance of Tensara if you're running one. 

## Adding a Problem
A problem is defined by two files `def.py` and `problem.md`:

The `def.py` file is extended from the [problems class](https://github.com/tensara/tensara/blob/main/engine/problem.py) and requires:
- `reference_solution`: this is treated as the correct implementation of the problem, and each submission is checked against this function. We recommend using pre-defined PyTorch functions when possible (with autocasting disabled), but CUDA reference solutions are also possible.
- `generate_test_cases`: returns a set of test cases that will be used to validate submissions.
- `verify_result`: implement logic to check whether the output of a submission matches the expected result. This is flexible -- you can include comparisons for numerical values or verify algorithmically. 
- `get_function_signature`: return argtypes based on [ctypes](https://docs.python.org/3/library/ctypes.html).
- `get_flops`: get the number of FLOPs as a function of the testcase size. Relevant for [benchmarking submissions](https://tensara.org/blog/benchmarking-solutions).
TODO: Get generalized FLOP counting?
- `get_extra_params`: (soon to be phased out) returns function parameters not used by `reference_solution`.

The `problem.md` file should contain a description of the problem written in Markdown (LaTeX supported!). The YAML Front Matter should contain:
- `slug`
- `title`
- `difficulty`: EASY, MEDIUM, or HARD
- `author`
- `tags` (soon to be cleaned up!)
- `parameters`
  - `name`
  - `type`: `[VAR]` if it's dependent on what `dtype` the problem is configured for, otherwise the C++ type
  - `pointer`: boolean
  - `const`: boolean
 
Once you add a problem, make sure to test both correct (slow/fast) and incorrect submissions. Let us know if you encounter any issues/bugs!
