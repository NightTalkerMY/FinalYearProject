# Runtime and Experiment Entry Points

## Integrated Runtime

The main backend entry point is:

```bash
python main_orchestrator.py
```

On the retained Windows deployment, `main.bat` starts the same entry point. The orchestrator starts MediaMTX monitoring, the React renderer, and the local AI services after both media streams are considered stable. The Raspberry Pi publisher must be started separately.

| Runtime role | Entry point | Default interface |
|---|---|---|
| Orchestrator | `main_orchestrator.py` | HTTP port 5000 |
| STT | `STT/main.py` | `POST /transcribe`, port 8000 |
| Phi-2 | `Chatbot_Phi2/main.py` | `POST /chat`, port 8001 |
| RAG | `RAG/main.py` | `POST /get_context`, port 8002 |
| ZipVoice | `TTS/ZipVoice/main.py` | `POST /generate_speech`, port 8003 |
| Visemes | `TTS/allosaurus/server.py` | `POST /generate_visemes`, port 8004 |
| DART | `Gesture_System/real-time-HGR-application/main.py` | WHEP input; HTTP gesture output |
| CATT-ASD | `online_ASD/realtime/pipeline_main.py` | WHEP input; ports 5050/5051 for Pi control |
| Pi publisher | `raspi/camera/send_picamera_stream_to_server.py` | WHIP output and HTTP control |
| React renderer | `react_avatar/` plus `react_avatar/launch-hologram.js` | Vite port 5173; WHIP `avatar` output |
| Edge receiver | `react_avatar/public/dome.html` | WHEP `avatar` input |

`online_ASD/realtime/realtime_main.py`, `online_ASD/utils/data_streamer.py`, `Gesture_System/real-time-HGR-application/app.py`, and the older `TTS/main.py` are retained development or legacy paths. They are not launched by the current central orchestration path.

## LARA Experiment

Entry point: `experiment_metric/reranker_metric/eval.ipynb`.

The notebook contains the evaluated length-aware packing and reciprocal rank-anchored reranking logic. Run it in an `experiment_metric` environment on a CUDA-capable machine. Before interpreting output, verify:

- cross-encoder checkpoint `cross-encoder/ms-marco-MiniLM-L-6-v2`;
- batch size 32;
- maximum sequence length 512;
- MS MARCO development query count 5,161;
- the distinction between the fixed MS MARCO reranking pool and the Retail QnA path that includes hybrid retrieval;
- that Table 2 is a single-run transcription and timing may differ.

Do not use `RAG/DatabaseRouting.py` as evidence that LARA is live: it does not execute the evaluated LARA transformations.

## DART Runtime and Evaluation

Live DART uses:

- `Gesture_System/real-time-HGR-application/main.py`;
- `Gesture_System/real-time-HGR-application/hgr_box_gate_v2.py`;
- `Gesture_System/real-time-HGR-application/gestureClassInference.py`.

The evaluation workspace is under `Gesture_System/real-time-HGR-application/metric_code/`. Its `eval.py` can compute trial-level and aggregate metrics when the corresponding capture directories are available.

The checked-in `metric_code/reports/*.csv` files are genuine outputs from an independently recovered legacy single-session run performed in May 2026. Their values, condition names, and sample scope do not match manuscript Tables 3--4. They are useful for testing the evaluator and report format, but not for recomputing the manuscript results.

## CATT-ASD Runtime and Evaluation

Live CATT-ASD uses:

- `online_ASD/realtime/pipeline_main.py`;
- `online_ASD/utils/student_model.py`;
- `online_ASD/pretrain_model/SOTA_studen_model/holopi_student_best.pt`.

The exact training and comprehensive evaluation workspace is in the separate MSSG repository at revision `1db2924ccd973cb60321125b00f765c10611a322`:

- `proposed/train_student.py`;
- `proposed/student_model.py`;
- `proposed/comprehensive_eval.py`;
- `realtime/realtime_epoch/training_log.csv`;
- `realtime/realtime_epoch/comprehensive_eval_result.json`.

The current repository's `online_ASD/` directory is the deployed inference subset. It should not be described as the complete training reproduction.

## Recording a New Run

For every new experiment, retain at minimum:

- `git rev-parse HEAD` for every repository used;
- command or notebook cell sequence;
- operating system, GPU, driver, Python, PyTorch, and CUDA versions;
- package lock or environment export;
- random seeds and repeat count;
- dataset revision and split;
- model/checkpoint hash;
- raw machine-readable output;
- timing warm-up, synchronization, batch size, and measured boundary.

Store new results in a new timestamped directory. Do not overwrite the legacy CSVs or the manuscript supplementary records.
