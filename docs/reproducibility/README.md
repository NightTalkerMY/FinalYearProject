# Reproducibility Guide

This directory connects the HoloPi manuscript to the retained implementation without turning the manuscript into a software module.

The distinction matters because the project contains three kinds of material:

1. **Live runtime code** used by the integrated prototype.
2. **Experiment code** used to evaluate an individual method, which may not be called by the live runtime.
3. **Reported evidence** retained in the manuscript submission package, which may outlive the original raw data or exact execution log.

## Status Vocabulary

| Status | Meaning |
|---|---|
| `live-integrated` | The retained orchestration path calls the component. |
| `benchmark-only` | The component was evaluated separately but is not invoked by the live system. |
| `configuration-dependent` | Source is present, but execution requires external assets, hardware, or deployment values. |
| `reported-only` | A manuscript result survives as an aggregate or transcription, but the exact original run cannot be independently reconstructed. |
| `legacy-recovered` | A later recovered artifact is genuine but does not correspond to the manuscript experiment. |

These statuses describe evidence availability, not the quality or validity of a method.

## Recommended Reading Order

1. Read [`manuscript-traceability.md`](manuscript-traceability.md) to locate the implementation behind each manuscript item.
2. Read [`runtime-and-experiments.md`](runtime-and-experiments.md) before running an entry point.
3. Inspect [`artifact-manifest.csv`](artifact-manifest.csv) for missing models, datasets, binaries, and frontend assets.
4. Run `python scripts/check_setup.py` from the repository root.
5. Record the commit, operating system, GPU, Python version, dependency environment, configuration, and command for any new run.

## What Is Not Added Here

- The LaTeX manuscript is not copied into this implementation repository.
- Missing DART participant trials are not recreated from aggregate values.
- The LR-ASD adapter or output is not reconstructed without its original checkpoint and execution record.
- LARA is not relabelled as live-integrated merely because its notebook is retained.
- A successful static source check is not presented as an end-to-end experimental reproduction.

## Current Reproduction Levels

| Scope | Current level |
|---|---|
| Source navigation | Reproducible from a fresh clone |
| Static manuscript-to-code audit | Reproducible from this documentation |
| Full backend startup | Requires external assets and Windows configuration |
| Raspberry Pi publisher | Requires Pi hardware, a supported camera, USB microphone, and wake-word model |
| LARA rerun | Partially reproducible from the retained notebook and datasets/download paths |
| DART manuscript tables | Reported-only; the complete original trial files are unavailable |
| CATT/MSSG tables | Reproducible only with the separately retained MSSG evaluation workspace and datasets |
| LR-ASD adapted rows | Reported-only |
| End-to-end latency | Not measured in the retained project |

The manuscript submission package contains machine-readable supplementary records for the reported LARA, DART, CATT-ASD, and system-integration evidence. Those submission artifacts are intentionally maintained with the manuscript rather than duplicated as live code here.
