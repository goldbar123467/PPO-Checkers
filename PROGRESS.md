# Progress

## Current Phase

Phase 1 — Rules verification (IN PROGRESS)

Work label: setup validation and primary-source verification.

Falsifiable objective: every R1.1–R7.3 row in `docs/RULES.md` cites an exact primary WCDF/ACF
clause or is explicitly labelled ENGINE VARIANT; R6.7 has a checked finite-resource derivation.

## In Flight

1. Obtain and archive the authoritative WCDF/ACF English Draughts rules text and its provenance.
2. Write a failing completeness/source-clause test for the `docs/RULES.md` traceability matrix.
3. Produce the rule matrix and R6.7 proof, adjudicate discrepancies, and run Gate 1.

## Gate Evidence

- Gate 0: `make check`, exit 0; 19 passed, 0 failed; scaffold coverage 100%;
  `logs/gates/phase-0.txt`.
- Sensitivity: failing-test and Ruff F401 probes both exited nonzero; evidence under
  `logs/test-output/000001-injected-*-red.txt`.
- GPU doctor after lock change: exit 0; `logs/gates/phase-0-gpu-doctor.txt`.

## Last Five Iterations

- 000001: Gate 0 GREEN; injected lint/test failures detected; CUDA/BF16/NF4 doctor passed.

## Open Risks

- Primary rules wording and clause numbering are not yet archived or mapped.
- The R6.7 worst-case termination bound must be derived carefully; `max_plies=512` must remain
  honestly below that bound.
- The repository has no initial commit yet; `git_sha` remains null until the green tree is committed.

## Next Step

Fetch the primary WCDF/ACF rules publication and preserve source metadata before authoring the
traceability matrix.
