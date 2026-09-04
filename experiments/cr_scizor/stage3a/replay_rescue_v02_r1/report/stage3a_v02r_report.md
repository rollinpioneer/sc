# Stage 3A v0.2-R Result Definition and Long-Horizon Counterfactual Evaluation Repair

## Final Decision

`RESUME_STAGE3A_E_ON_V02R`

## Repair Scope

- Fixed the paired-clean normalizer field mismatch.
- Replaced the 40-frame observation with the frozen 100-frame horizon.
- Used the frozen three-frame persistent-improvement score and 0.4/0.5/0.1 weights.
- Kept the normalizer train-only from v0.2 paired-clean data.
- Generated the confirmation set from 8 previously unused successful Can demos and 8 previously unused successful Square demos.

## Development Validation Record Only

- paired-clean AUROC: 0.8810572687224669
- primary feasible AUROC: 0.8286495518760444

## Confirmation Final Gate

- pairs: 256, effective interventions: 28, feasible replacements: 1024
- paired-clean engineering: branch prefix, reference replay, clean replay, and finite-target rates are all 1.0
- feasible engineering: branch prefix, reference replay, and finite-target rates are all 1.0
- engineering pass: true
- paired-clean AUROC: 0.9051535087719298
- primary feasible AUROC: 0.9071115288220551
- best-of-4 feasible AUROC: 0.8981829573934837 (diagnostic only)
- failed rules: none

## Next Action

Resume Stage 3A-E using v0.2 train long-horizon scores. The confirmation set must not enter training.
