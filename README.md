# SCIZOR research-project mirror

This is a GitHub-safe mirror of the local SCIZOR research project. It preserves
the local project directory layout under `SCIZOR/` and `experiments/cr_scizor/`.

Large generated artifacts, datasets, checkpoints, indexes, archives, videos, and
evaluation tables are intentionally not uploaded. Each omitted artifact has a
same-location `<original-name>.placeholder.md` file that records its original
filename, local relative path, byte size, and artifact class. The complete index
is in `LARGE_ARTIFACT_MANIFEST.tsv`.

The original local project remains authoritative for non-versioned large assets.

## HB1 handback feasibility experiment

The HB1 experiment is published under
[`experiments/handback/hb1_v1/`](experiments/handback/hb1_v1/), with its
implementation under [`SCIZOR/recovery_handback/`](SCIZOR/recovery_handback/).

Final decision: **HOLD**.

- Can: `NEED_BASE_POLICY`
- Square: `NEED_STRONGER_REPAIRER`
- Final audit: 39/39 checks passed
- Report: [`HB1_REPORT.md`](experiments/handback/hb1_v1/report/HB1_REPORT.md)
- Lightweight results: [`HB1_results_lightweight.zip`](experiments/handback/hb1_v1/package/HB1_results_lightweight.zip)

The formal paired probe was not run because the protocol's capability stop
conditions were reached. Checkpoints, replay buffers, trajectory payloads,
evaluation tables, and runtime logs remain local; see the HB1 local-only
artifact note in the result directory.

## HB2 visual handback prediction experiment

The HB2 Square experiment is published under
[`experiments/handback/hb2_v1/`](experiments/handback/hb2_v1/), with its
implementation under [`SCIZOR/recovery_handback/hb2/`](SCIZOR/recovery_handback/hb2/).

Final status: **HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN**.

- Test coverage: 40 roots, 118 valid anchors, and 590 unique branch outcomes
- Selected visual model: `M4_paired` (canonical seed 0)
- Canonical visual-vs-nonvisual utility difference: -0.0374
- Root-cluster bootstrap 95% CI: [-0.0866, 0.0031]
- Report: [`HB2_REPORT.md`](experiments/handback/hb2_v1/report/HB2_REPORT.md)
- Test metrics: [`summary.json`](experiments/handback/hb2_v1/metrics/test/summary.json)
- Final decision: [`hb2_decision.json`](experiments/handback/hb2_v1/metrics/hb2_decision.json)
- Lightweight results: [`HB2_results_lightweight.zip`](experiments/handback/hb2_v1/package/HB2_results_lightweight.zip)

## HB3-P episode-level handback experiment

The HB3-P Square experiment is published under
[`experiments/handback/hb3p_v1/`](experiments/handback/hb3p_v1/), with its
implementation under [`SCIZOR/recovery_handback/hb3p/`](SCIZOR/recovery_handback/hb3p/).

Final status: **STATE_ENTRY_SIGNAL_ONLY**.

- Coverage: 80 roots, 480 complete logical records, 320 unique rollouts
- Primary M1 versus S_STAR utility delta: `0.014531`
- Root-cluster bootstrap 95% CI: `[-0.068457, 0.093523]`
- Exit diagnosis: [`HB3P_EXIT_DIAGNOSIS.md`](experiments/handback/hb3p_v1/metrics/diagnosis/HB3P_EXIT_DIAGNOSIS.md)
- Route decision: [`exit_diagnosis_decision.json`](experiments/handback/hb3p_v1/metrics/diagnosis/exit_diagnosis_decision.json)
- Lightweight results: [`HB3P_results_lightweight.zip`](experiments/handback/hb3p_v1/package/HB3P_results_lightweight.zip)

The diagnosis is read-only and does not train a stopping model. Its bounded
follow-up probe uses new roots and confirmed an intermediate-exit opportunity:
[`HB3P_EXIT_PROBE.md`](experiments/handback/hb3p_exit_probe_v1/report/HB3P_EXIT_PROBE.md).

## HB3-P intermediate exit probe

The bounded follow-up uses a fixed entry at t=20 and fixed 40/60/80-step exits
on 40 new Square roots.

- Coverage: 40 roots, 160 complete logical records, 160 unique rollouts
- Total environment steps: `57990` of the `64000` protocol limit
- Autonomous completion: `5/40` at L40, `10/40` at L60, `11/40` at L80
- Prefix verification: `40/40`
- Decision: **INTERMEDIATE_EXIT_OPPORTUNITY_CONFIRMED**
- Report: [`HB3P_EXIT_PROBE.md`](experiments/handback/hb3p_exit_probe_v1/report/HB3P_EXIT_PROBE.md)
- Decision JSON: [`decision.json`](experiments/handback/hb3p_exit_probe_v1/metrics/exit_probe/decision.json)
- Lightweight results: [`HB3P_exit_probe_lightweight.zip`](experiments/handback/hb3p_exit_probe_v1/package/HB3P_exit_probe_lightweight.zip)

No stopping/continuation model was trained; any training requires a new frozen protocol.
