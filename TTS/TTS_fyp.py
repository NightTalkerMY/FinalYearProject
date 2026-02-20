import time
from pathlib import Path
from datetime import datetime

import torch


def _patch_torch_load_weights_only_false():
    """
    PyTorch 2.6+ defaults weights_only=True, which breaks some XTTS checkpoints.
    We trust the official Coqui model files downloaded locally, so force weights_only=False.
    """
    real_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return real_load(*args, **kwargs)

    torch.load = compat_load


class XTTSEngine:
    def __init__(
        self,
        speaker_wav: str | None = None,   # path to your cloned voice
        speaker: str | None = None,       # built-in Coqui speaker name
        language: str = "en",
        device: str | None = None,
        out_dir: str = "outputs_xtts",
        split_sentences: bool = True,
        voice_mode: str = "auto",         # "auto" | "clone" | "default"
    ):
        """
        Voice selection logic 🗣️

        - voice_mode="clone":
            Always use `speaker_wav` (error if not provided or file missing)
        - voice_mode="default":
            Always use `speaker` (error if not provided)
        - voice_mode="auto":
            If speaker_wav file exists -> use cloned voice
            else if speaker is set -> use built-in speaker
            else error

        Examples:
            # 1) Only cloned voice
            XTTSEngine(
                speaker_wav="Recording.wav",
                voice_mode="clone",
            )

            # 2) Only built-in voice
            XTTSEngine(
                speaker="Ana Florence",
                voice_mode="default",
            )

            # 3) Prefer cloned voice, fall back to built-in if file missing
            XTTSEngine(
                speaker_wav="Recording.wav",
                speaker="Ana Florence",
                voice_mode="auto",
            )
        """
        _patch_torch_load_weights_only_false()

        from TTS.api import TTS  # import after patch

        self.speaker_wav = str(Path(speaker_wav)) if speaker_wav else None
        self.speaker = speaker
        self.language = language
        self.split_sentences = split_sentences
        self.voice_mode = voice_mode  # "auto" | "clone" | "default"

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device = "cpu"
        self.device = device

        print("Loading XTTS model...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.tts.to(self.device)

        print(f"Device: {self.device}")
        if self.device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))

        # warmup (optional but helps stabilize first timing)
        self._warmup()

    # ---------- voice mode helpers (so you can switch anytime) ----------

    def use_cloned_voice(self):
        """Force using cloned voice (speaker_wav)."""
        self.voice_mode = "clone"
        print("Voice mode -> clone (speaker_wav)")

    def use_default_voice(self):
        """Force using built-in Coqui speaker."""
        self.voice_mode = "default"
        print("Voice mode -> default (speaker)")

    def use_auto_voice(self):
        """Auto: prefer cloned if file exists, else default speaker."""
        self.voice_mode = "auto"
        print("Voice mode -> auto (prefer cloned, fallback default)")

    # ---------- internal helpers ----------

    def _resolve_voice(self) -> tuple[str, str]:
        """
        Returns (mode, value):

        - ("clone", path_to_wav)
        - ("default", speaker_name)
        """
        mode = self.voice_mode

        # Normalize file existence
        wav_exists = False
        if self.speaker_wav is not None:
            wav_exists = Path(self.speaker_wav).is_file()

        if mode == "clone":
            if not self.speaker_wav:
                raise RuntimeError("voice_mode='clone' but no speaker_wav was provided.")
            if not wav_exists:
                raise RuntimeError(
                    f"voice_mode='clone' but speaker_wav file not found: {self.speaker_wav}"
                )
            return "clone", self.speaker_wav

        if mode == "default":
            if not self.speaker:
                raise RuntimeError("voice_mode='default' but no speaker name was provided.")
            return "default", self.speaker

        # auto mode
        if mode == "auto":
            if wav_exists:
                return "clone", self.speaker_wav
            if self.speaker:
                return "default", self.speaker
            raise RuntimeError(
                "voice_mode='auto' but no usable voice found "
                "(no existing speaker_wav and no speaker name)."
            )

        raise RuntimeError(f"Unknown voice_mode: {mode}")

    def _tts_kwargs(self, text: str, file_path: str) -> dict:
        """
        Build the kwargs for tts_to_file depending on whether we're using
        a cloned voice (speaker_wav) or a built-in Coqui speaker (speaker).
        """
        mode, value = self._resolve_voice()

        kwargs = dict(
            text=text,
            file_path=file_path,
            language=self.language,
            split_sentences=self.split_sentences,
        )

        if mode == "clone":
            kwargs["speaker_wav"] = value
        else:  # "default"
            kwargs["speaker"] = value

        return kwargs

    def _warmup(self):
        print("Warming up...")
        mode, value = self._resolve_voice()
        print(f"Using voice for warmup -> mode={mode}, value={value}")

        # Warmup without writing to disk (faster, no file I/O)
        if mode == "clone":
            _ = self.tts.tts(
                text="Warmup.",
                speaker_wav=value,
                language=self.language,
                split_sentences=False,
            )
        else:  # "default"
            _ = self.tts.tts(
                text="Warmup.",
                speaker=value,
                language=self.language,
                split_sentences=False,
            )

        if self.device == "cuda":
            torch.cuda.synchronize()
        print("Warmup done.")

    def speak(self, text: str, file_path: str | None = None) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")

        if file_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = str(self.out_dir / f"{ts}.wav")

        t0 = time.perf_counter()

        # Generate + write wav
        self.tts.tts_to_file(**self._tts_kwargs(text=text, file_path=file_path))

        if self.device == "cuda":
            torch.cuda.synchronize()

        dt = time.perf_counter() - t0

        # Estimate audio length by reading the saved file (no extra deps)
        import wave

        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            audio_sec = frames / float(sr)

        rtf = dt / audio_sec if audio_sec > 0 else float("inf")

        print(f"Saved: {file_path}")
        print(f"Time: {dt:.3f}s | Audio: {audio_sec:.3f}s | RTF: {rtf:.3f} (lower is faster)")
        return file_path


if __name__ == "__main__":
    # Example 1: prefer cloned voice, fallback to built-in if file not present
    engine = XTTSEngine(
        speaker_wav="New Recording.wav",   # your cloned voice (if exists)
        speaker="Ana Florence",        # default built-in voice
        language="en",
        out_dir="outputs_xtts",
        split_sentences=True,
        voice_mode="auto",             # "auto" | "clone" | "default"
    )

    print("\nType text and press Enter. Type /quit to exit.\n")
    while True:
        s = input("Text> ").strip()
        if s.lower() in ("/q", "/quit", "quit", "exit"):
            break
        try:
            engine.speak(s)
        except Exception as e:
            print("Error:", e)
