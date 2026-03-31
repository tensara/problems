# Problem Automation Roadmap

This roadmap defines how `tensara/problems` becomes agent-friendly without breaking current workflows.

## Immediate Direction

The first priority is not contests. It is deterministic ingestion.

That means:

1. agents can author to a stable contract
2. validators can reject weak or broken problems automatically
3. accepted problems are safe to sync and publish

## Phase 1: Contract First

Ship:

- a stable authoring contract
- a stable validation contract
- a machine-readable validator
- structural CI on every PR
- templates for new problems

This PR implements Phase 1.

## Phase 2: Stronger Runtime Validation

Add:

- routine H100 validation for cheap local CUDA truth
- a Modal-backed product-runtime check as the final acceptance gate
- persisted validation artifacts that agents can inspect

## Phase 3: Verifier Strength

Add:

- adversarial wrong-answer checks
- mutation tests for common failure modes
- problem-family-specific negative cases

This is how testcase quality becomes measurable instead of subjective.

## Phase 4: Sync Hardening

Tighten `sync-problems.ts` so sync fails on structural contract violations instead of only warning.

Examples:

- missing parameters
- bad method signatures
- slug/frontmatter mismatch

## Phase 5: Automated Growth

Once the contract and validators are stable:

- agents can open daily problem PRs
- CI can auto-classify failures
- maintainers can review mostly by exception
- accepted problems can auto-sync to production

## Phase 6: Contest Automation

Only after validation is trusted:

- assemble contest sets automatically
- require hidden-test quality and difficulty spread
- publish and schedule through the same validated pipeline

## Non-Goals For This Phase

- rewriting all existing problems
- forcing immediate frontmatter migration
- making Modal validation mandatory in this repo before the surrounding auth/runtime glue is ready

The immediate goal is a reliable contract and validator surface that later Modal automation can plug into.
