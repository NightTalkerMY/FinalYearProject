#!/usr/bin/env python3
"""Static source and external-asset checks for the retained HoloPi repository."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "reproducibility" / "artifact-manifest.csv"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    required: bool = True


SOURCE_PATHS = (
    "main_orchestrator.py",
    "mediamtx/mediamtx.yml",
    "raspi/camera/send_picamera_stream_to_server.py",
    "Gesture_System/real-time-HGR-application/main.py",
    "Gesture_System/real-time-HGR-application/hgr_box_gate_v2.py",
    "online_ASD/realtime/pipeline_main.py",
    "online_ASD/utils/student_model.py",
    "STT/main.py",
    "RAG/main.py",
    "RAG/DatabaseRouting.py",
    "Chatbot_Phi2/main.py",
    "TTS/ZipVoice/main.py",
    "TTS/allosaurus/server.py",
    "react_avatar/src/App.jsx",
    "react_avatar/public/dome.html",
    "experiment_metric/reranker_metric/eval.ipynb",
)

BACKEND_ASSETS = (
    ("MediaMTX binary", "mediamtx/mediamtx.exe"),
    ("Phi-2 PEFT adapter", "Chatbot_Phi2/models/phi2_retail_native_bf16_38f4a5"),
    ("DART Windows classifier", "Gesture_System/real-time-HGR-application/.sources/[0c02]-6G-[cm_td_fa].pkl"),
    ("CATT-ASD student checkpoint", "online_ASD/pretrain_model/SOTA_studen_model/holopi_student_best.pt"),
    ("S3FD face-detector weight", "online_ASD/model/faceDetector/s3fd/sfd_face.pth"),
    ("RAG database", "RAG/db"),
    ("RAG router anchors", "RAG/teleoracle_v2_anchors.npz"),
    ("React avatar models", "react_avatar/public/models"),
    ("React product assets", "react_avatar/public/products"),
    ("React Node dependencies", "react_avatar/node_modules"),
    ("ZipVoice prompt audio", "TTS/ZipVoice/short_voice.wav"),
)

PI_ASSETS = (
    ("Pi camera publisher", "raspi/camera/send_picamera_stream_to_server.py"),
    ("Custom wake-word model", "raspi/camera/hey_holo.onnx"),
)


def resolve_config_path(variable: str, default: str) -> Path:
    value = Path(os.getenv(variable, default)).expanduser()
    return value if value.is_absolute() else ROOT / value


def path_check(name: str, path: Path, required: bool = True) -> Check:
    if path.exists():
        try:
            detail = str(path.relative_to(ROOT))
        except ValueError:
            detail = str(path)
        return Check(name, "PASS", detail, required)
    return Check(name, "MISSING", str(path), required)


def source_checks() -> list[Check]:
    checks = [path_check(f"source: {item}", ROOT / item) for item in SOURCE_PATHS]
    checks.append(path_check("artifact manifest", MANIFEST))
    checks.append(
        Check(
            "Python version",
            "PASS" if sys.version_info >= (3, 10) else "WARN",
            platform.python_version(),
            False,
        )
    )
    checks.append(Check("platform", "INFO", platform.platform(), False))
    return checks


def backend_checks() -> list[Check]:
    checks = [path_check(name, ROOT / relative) for name, relative in BACKEND_ASSETS]
    checks.extend(
        [
            Check("node command", "PASS" if shutil.which("node") else "MISSING", shutil.which("node") or "not on PATH"),
            Check("npm command", "PASS" if shutil.which("npm") else "MISSING", shutil.which("npm") or "not on PATH"),
            path_check("configured MediaMTX directory", resolve_config_path("HOLOPI_MEDIAMTX_DIR", "mediamtx")),
            path_check(
                "configured watchdog Python",
                resolve_config_path("HOLOPI_WATCHDOG_PYTHON", "mediamtx/venv/Scripts/python.exe"),
            ),
        ]
    )
    if os.name != "nt":
        checks.append(
            Check(
                "Windows deployment",
                "WARN",
                "The retained orchestrator uses Windows process-management commands.",
                False,
            )
        )
    return checks


def pi_checks() -> list[Check]:
    return [path_check(name, ROOT / relative) for name, relative in PI_ASSETS]


def load_manifest() -> Check:
    try:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"artifact", "expected_location", "git_status", "required_for", "acquisition_or_source", "notes"}
        fields = set(rows[0]) if rows else set()
        if not rows or not required.issubset(fields):
            return Check("manifest format", "MISSING", "Manifest is empty or incomplete")
        return Check("manifest format", "PASS", f"{len(rows)} artifact records", False)
    except (OSError, csv.Error) as exc:
        return Check("manifest format", "MISSING", str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("source", "backend", "pi", "full"),
        default="source",
        help="Check retained source only, or include deployment assets.",
    )
    parser.add_argument("--strict", action="store_true", help="Return a non-zero exit code for missing required items.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = source_checks()
    checks.append(load_manifest())
    if args.profile in {"backend", "full"}:
        checks.extend(backend_checks())
    if args.profile in {"pi", "full"}:
        checks.extend(pi_checks())

    if args.as_json:
        print(json.dumps({"profile": args.profile, "checks": [asdict(item) for item in checks]}, indent=2))
    else:
        width = max(len(item.name) for item in checks)
        print(f"HoloPi setup check ({args.profile})")
        print(f"Repository: {ROOT}")
        for item in checks:
            print(f"[{item.status:7}] {item.name:<{width}}  {item.detail}")
        missing = sum(item.status == "MISSING" and item.required for item in checks)
        print(f"\nRequired items missing: {missing}")
        if missing:
            print("See docs/reproducibility/artifact-manifest.csv for acquisition and provenance notes.")

    has_missing = any(item.status == "MISSING" and item.required for item in checks)
    return 1 if args.strict and has_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
