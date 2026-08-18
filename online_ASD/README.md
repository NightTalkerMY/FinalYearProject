# CATT-ASD Deployed Runtime

This directory is the active-speaker inference subset deployed in HoloPi. It is not the complete historical training and evaluation repository.

## Live Entry Point

The orchestrator launches `realtime/pipeline_main.py`. The pipeline consumes audio and video from the MediaMTX `cam1` WHEP endpoint, maintains a causal 50-frame buffer, performs face tracking and CATT-ASD inference, gates the speech interaction, and exchanges wake-word/stop control messages with the Raspberry Pi over HTTP.

Important retained files include:

- `utils/student_model.py`: causal student architecture;
- `pretrain_model/SOTA_studen_model/holopi_student_best.pt`: selected student checkpoint;
- `model/faceDetector/s3fd/sfd_face.pth`: face-detector weight;
- `realtime/pipeline_main.py`: deployed integration logic.

Deployment endpoints and model paths can be overridden with the variables in the repository-level `.env.example`. S3FD resolves its weight in this order: an explicit `HOLOPI_S3FD_WEIGHTS` value, the original FYP desktop path when it still exists, and then the tracked repository-local weight.

## Training and Evaluation Boundary

The exact manuscript training and comprehensive evaluation workspace is retained in `https://github.com/NightTalkerMY/MSSG` at revision `1db2924ccd973cb60321125b00f765c10611a322`. Its relevant paths are:

- `proposed/train_student.py`;
- `proposed/student_model.py`;
- `proposed/comprehensive_eval.py`;
- `realtime/realtime_epoch/training_log.csv`;
- `realtime/realtime_epoch/comprehensive_eval_result.json`.

The checkpoint used in the manuscript was epoch 42, selected using `0.4 * AVA mAP + 0.6 * Columbia F1`. Columbia therefore participated in model selection and is reported as cross-dataset validation.

`realtime/realtime_main.py` and `utils/data_streamer.py` are retained development/legacy paths and are not launched by the current orchestrator.

See `../docs/reproducibility/manuscript-traceability.md` for Tables 5--7.
