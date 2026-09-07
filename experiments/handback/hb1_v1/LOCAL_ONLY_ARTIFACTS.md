# HB1 local-only artifacts

This GitHub mirror contains the HB1 implementation, fixed configuration and
policy-pair metadata, JSON/CSV metrics, figures, final report, and verified
lightweight results package.

The following generated artifact classes remain in the authoritative local
experiment directory and are intentionally not mirrored:

- base-policy and repair-policy checkpoints (`*.pth`, `*.pt`, SAC `*.zip`)
- replay buffers and normalizers (`*.pkl`)
- rollout, payload, history, and policy-memory arrays (`*.npz`)
- source datasets (`*.hdf5`, `*.h5`)
- evaluation and root tables (`*.parquet`)
- videos (`*.mp4`)
- logs, PID files, caches, and machine-specific environment files

`package/HB1_results_lightweight.zip` is the only included ZIP. Its digest is
recorded in `package/HB1_results_lightweight.zip.sha256`.
