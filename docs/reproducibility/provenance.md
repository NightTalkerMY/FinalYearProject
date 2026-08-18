# Provenance and Historical Boundaries

## HoloPi Integration Repository

- Repository: `https://github.com/NightTalkerMY/FinalYearProject`
- Manuscript audit baseline: `18de03cdf1c85251b8f9163b3e840ddaedb3826d`
- Audited platform: Windows backend with NVIDIA RTX 4080 SUPER
- Current cleanup: uncommitted until committed by the repository owner

The audit established that DART and CATT-ASD are present in the retained runtime. LARA is present as a benchmark notebook but is not called by the live RAG service.

## CATT-ASD and MSSG

- Repository: `https://github.com/NightTalkerMY/MSSG`
- Retained revision: `1db2924ccd973cb60321125b00f765c10611a322`
- Training program: `proposed/train_student.py`
- Comprehensive evaluator: `proposed/comprehensive_eval.py`
- Selected checkpoint: epoch 42, chosen by `0.4 * AVA mAP + 0.6 * Columbia F1`

Because Columbia contributed to checkpoint selection, its manuscript result is cross-dataset validation rather than an untouched external-test estimate.

## LARA

- Retained notebook: `experiment_metric/reranker_metric/eval.ipynb`
- Provenance revision cited in the manuscript supplement: `95a014e4c5435ae7c540f08ac7678e5daf145faa`
- Reported hardware: NVIDIA RTX 4080 SUPER on Windows

The exact output log behind the manuscript's single-run Table 2 values was not retained. A saved notebook execution is closely matching but should not be represented as the exact original run.

## DART

The original three-participant raw trial files behind manuscript Tables 3--4 were not recovered. The retained manuscript values are therefore reported aggregate evidence.

The checked-in May 2026 `metric_code/reports/` data are a separate legacy single-session run. Their aggregates differ from the manuscript and must remain labelled `legacy-recovered`.

## LR-ASD

The exact streaming adapter, checkpoint, evaluator, and output log used for the manuscript rows were not retained. The manuscript rows are archival transcriptions from the submitted final-year-project report. They are not the native scores from the cited LR-ASD release.

## Hardware and Device Details

All retained backend experiments described in the current manuscript were performed on the RTX 4080 SUPER. The exact camera and USB microphone product models used for the physical demonstration were not recorded in source or surviving notes. Code-level capture configuration is retained, but product-model details must remain unspecified.
