---
name: "lamet_asymptotic_expansion"
description: "Use when relating quasi-observables at finite hadron momentum to light-cone quantities using leading-power LaMET asymptotic expansion."
---

# LaMET Asymptotic Expansion

Apply this skill when the task involves quasi-distributions, quasi-TMD matrix elements, or other quasi-observables computed at finite hadron momentum and you need a leading-power relation to the corresponding light-cone quantity.

## Goal

Relate quasi-observables computed at finite hadron momentum to their light-cone counterparts using the large-momentum effective theory (LaMET) asymptotic expansion.

## Scope

- Hadron momentum `P_z >> Lambda_QCD`
- Leading-power LaMET only
- Neglect `O(1 / P_z^2)` power corrections unless the user explicitly asks to analyze them

## Inputs

- `quasi_observable`: Renormalized quasi-distribution or quasi-TMD matrix element defined with spacelike Wilson lines
- `hadron_momentum`: Large hadron momentum component, usually `P_z`
- `renormalization_scale`: Renormalization scale `mu`

## Outputs

- `factorized_form`: Leading-power factorized expression relating the quasi-observable to the corresponding light-cone quantity
- `power_counting_statement`: Explicit statement of neglected power corrections and their parametric scaling

## Workflow

1. Identify the large-momentum variable.
   Fix the reference frame and specify which hadron momentum component is taken to be asymptotically large.

2. Perform the power expansion.
   Expand the quasi-observable in powers of `1 / P_z` and keep the leading-power term.

3. Match operator structures.
   Identify the light-cone operator corresponding to the leading term in the large-momentum expansion.

4. State the factorization formula.
   Write the leading-power LaMET relation, including perturbative matching if required by the task.

## Quality Checks

- The extracted leading contribution should be `P_z` independent up to higher-order corrections.
- Subleading terms should be explicitly described as parametrically suppressed, typically `O(1 / P_z^2)` or smaller.

## Constraints

- Do not claim power-correction control beyond leading power unless it is derived or explicitly provided.
- Do not blur quasi-observables and their matched light-cone counterparts; keep the mapping explicit.
- If perturbative matching kernels are required but not given, state that they must be supplied or computed separately.
