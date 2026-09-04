# Stage 1C Annotation Specification

## Scope and source of truth

Every perturbed pair changes exactly one recorded action. Simulator reward and staged reward gaps define outcome labels; SCIZOR scores are never used to define ground truth.

- `intervention_t`: the known action replacement time step.
- `failure_onset`: the first of three consecutive steps above the reward or staged-reward gap threshold.
- `recovery_start`: first onset-following point where the five-step gap average falls at least 30 percent from its peak.
- `recovery_end`: first of three consecutive steps at or below half the gap threshold.
- `responsible_t`, `responsible_start`, `responsible_end`: the intervention point and its one-step neighborhood, clipped to episode bounds.

## Outcome labels

- `direct_failure`: final perturbed failure whose onset is at most three steps after the intervention.
- `delayed_failure`: final perturbed failure with a later onset.
- `recovery_failure`: recovery evidence occurs but the perturbed rollout is not successful.
- `recovery_success`: a recovery interval ends and the perturbed rollout is successful.
- `no_effect`: no sustained adverse outcome, with no failure onset.
- `ambiguous`: numerical issue, insufficient episode, clean replay failure, or unresolved rule conflict. Ambiguous pairs are excluded from primary localization metrics.

The Stage 1C freeze corrects the historical conflict `failure_type=no_effect` with non-null `failure_onset`: only a successful rollout with both recovery bounds can become `recovery_success`; all other such cases become `ambiguous`. This is a fixed per-record policy and does not modify global thresholds or rerun rollouts.

## Responsibility labels

Effective interventions are only `direct_failure`, `delayed_failure`, `recovery_failure`, and `recovery_success`.

- `is_responsible_point = is_effective_intervention and t == responsible_t`.
- `is_responsibility_region = is_effective_intervention and responsible_start <= t <= responsible_end`.
- `is_no_effect_negative_control` identifies consistent no-effect pairs.
- `no_effect_false_attribution_rate` is the fraction of their intervention steps assigned high model responsibility after model scores exist.

## Transition rows and split

Clean transitions are written exactly once per `(task, base_demo_id)`, with null `pair_id` and `variant=clean`. Every perturbed pair retains one full transition sequence. Groups are split by `(task, base_demo_id)` with seed `20260831`; clean and all variants of a base episode share one split. `is_rare` is derived only from train-split frequencies, regrasp transitions, and recovery-failure behavior.
