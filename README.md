# tag-gar

This repository contains ST-GCN-based code and assets for basketball action recognition and KD experiments.

## Directory overview

- `migrate_bundle_train_kd/`: main training/evaluation project (code, configs, data, work outputs).
- `net/`, `feeder/`, `torchlight/`: core modules mirrored outside the migration bundle.
- `tools_min/`: lightweight analysis and visualization scripts.
- `sga_interact/`: config files for interaction-related runs.
- `docker/`: container build/run helpers.
- `data/`, `outputs/`: project-level data/output placeholders.

## Quick cleanup

Use the organizer script to remove generated clutter safely.

```bash
bash scripts/organize_project.sh --dry-run
bash scripts/organize_project.sh --apply
```

The cleanup targets cached Python files, build artifacts, and Windows Zone Identifier sidecar files.

## TAL migration

`migrate_bundle_train_kd` now includes an initial TAL path that reuses the basketball `tactic.pkl` multi-segment annotations inside each clip:

- feeder: `migrate_bundle_train_kd/feeder/feeder_tal.py`
- model: `migrate_bundle_train_kd/net/tag_tal.py`
- processor: `migrate_bundle_train_kd/processor/tal.py`
- config: `migrate_bundle_train_kd/config/tal/stgcn_tag_tal_basketball.yaml`

This is clip-level temporal localization, not full untrimmed-video TAL.
