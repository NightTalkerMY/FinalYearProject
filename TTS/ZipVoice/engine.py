import os
import sys
import json
import torch
import torchaudio
import warnings

# ==========================================
# 1. KILL THE NETWORK LAG
# Force HuggingFace offline so it uses local cache instantly
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Suppress annoying PyTorch warnings
warnings.filterwarnings("ignore")

# Import ZipVoice internal modules natively
from huggingface_hub import hf_hub_download
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.bin.infer_zipvoice import get_vocoder, generate_sentence

# ==========================================
#               CONFIGURATION
# ==========================================
ORIGINAL_WAV = "New Recording.wav" 
SHORT_WAV = "short_voice.wav"
PROMPT_TEXT = "Technologies has become an essential part of everyday."
# ==========================================

def slice_audio(input_path, output_path, duration_seconds=3):
    print(f"[*] Loading original audio: {input_path}...")
    try:
        waveform, sample_rate = torchaudio.load(input_path)
        num_frames = int(sample_rate * duration_seconds)
        short_waveform = waveform[:, :num_frames]
        torchaudio.save(output_path, short_waveform, sample_rate)
        print(f"[+] Success! Created perfect {duration_seconds}-second prompt: {output_path}")
        return True
    except Exception as e:
        print(f"[-] Error slicing audio: {e}")
        return False

class ZipVoiceEngine:
    """A persistent VRAM engine that holds the model open for zero-latency inference."""
    def __init__(self, prompt_wav, prompt_text):
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        self.repo = "k2-fsa/ZipVoice"
        self.model_dir = "zipvoice_distill"
        self.sampling_rate = 24000
        
        print("\n[!] Waking up the AI and loading into VRAM (This happens ONLY ONCE)...")
        self._load_engine()
        print("[+] AI Engine is awake and locked in VRAM!\n")

    def _load_engine(self):
        # Locate the local cached files
        model_ckpt = hf_hub_download(self.repo, filename=f"{self.model_dir}/model.pt")
        model_config_path = hf_hub_download(self.repo, filename=f"{self.model_dir}/model.json")
        token_file = hf_hub_download(self.repo, filename=f"{self.model_dir}/tokens.txt")

        # Initialize Tokenizer
        self.tokenizer = EmiliaTokenizer(token_file=token_file)
        tokenizer_config = {"vocab_size": self.tokenizer.vocab_size, "pad_id": self.tokenizer.pad_id}

        # Load Model Config
        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        # Initialize the Distilled Model
        self.model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)
        load_checkpoint(filename=model_ckpt, model=self.model, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize Vocoder & Feature Extractor
        self.vocoder = get_vocoder(None).to(self.device)
        self.vocoder.eval()
        self.feature_extractor = VocosFbank()

    # def speak(self, text, output_path):
    #     """Passes text directly to the active model without rebooting."""
    #     # Capture the metrics dictionary returned by the function
    #     metrics = generate_sentence(
    #         save_path=output_path,
    #         prompt_text=self.prompt_text,
    #         prompt_wav=self.prompt_wav,
    #         text=text,
    #         model=self.model,
    #         vocoder=self.vocoder,
    #         tokenizer=self.tokenizer,
    #         feature_extractor=self.feature_extractor,
    #         device=self.device,
    #         num_step=4,              # Aggressive speed optimization
    #         guidance_scale=3.0,      # Default for distill
    #         speed=1.0,
    #         t_shift=0.5,
    #         target_rms=0.1,
    #         feat_scale=0.1,
    #         sampling_rate=self.sampling_rate,
    #         max_duration=30,
    #         remove_long_sil=False
    #     )
    #     torch.cuda.empty_cache()
        
    #     # Extract the exact values you requested
    #     dt = metrics["t"]
    #     audio_sec = metrics["wav_seconds"]
    #     rtf = metrics["rtf"]
        
    #     # Print the performance depth
    #     print(f"Time: {dt:.3f}s | Audio: {audio_sec:.3f}s | RTF: {rtf:.3f} (lower is faster)")

    @torch.inference_mode()  # <--- THIS IS THE MAGIC LOCK
    def speak(self, text, output_path):
        """Passes text directly to the active model without rebooting."""
        # Capture the metrics dictionary returned by the function
        metrics = generate_sentence(
            save_path=output_path,
            prompt_text=self.prompt_text,
            prompt_wav=self.prompt_wav,
            text=text,
            model=self.model,
            vocoder=self.vocoder,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            device=self.device,
            num_step=4,              # Aggressive speed optimization
            guidance_scale=3.0,      # Default for distill
            speed=1.0,
            t_shift=0.5,
            target_rms=0.1,
            feat_scale=0.1,
            sampling_rate=self.sampling_rate,
            max_duration=30,         # <--- DROPPED TO 15 TO CAP VRAM SPIKES
            remove_long_sil=False
        )
        
        # Free the tiny bit of cache actually used
        torch.cuda.empty_cache()
        
        # Extract the exact values you requested
        dt = metrics["t"]
        audio_sec = metrics["wav_seconds"]
        rtf = metrics["rtf"]
        
        # Print the performance depth
        print(f"Time: {dt:.3f}s | Audio: {audio_sec:.3f}s | RTF: {rtf:.3f} (lower is faster)")

def main():
    print("ZipVoice Persistent Engine (TRUE ZERO-LATENCY)\n")
    
    if not os.path.exists(ORIGINAL_WAV) and not os.path.exists(SHORT_WAV):
        print(f"[-] Error: Could not find '{ORIGINAL_WAV}' or '{SHORT_WAV}'. Check your files!")
        return

    if os.path.exists(ORIGINAL_WAV) and not os.path.exists(SHORT_WAV):
        print("[*] Automatically slicing your long audio file down to 3 seconds...")
        success = slice_audio(ORIGINAL_WAV, SHORT_WAV, duration_seconds=3)
        if not success: return
    else:
        print(f"[*] Found existing optimized prompt: {SHORT_WAV}")

    # ==========================================
    # 2. START THE ENGINE ONCE
    # ==========================================
    engine = ZipVoiceEngine(SHORT_WAV, PROMPT_TEXT)

    print("="*50)
    print("RTX 4080 SUPER IS LISTENING")
    print("Type your text and press Enter for instant generation.")
    print("Type 'exit' or 'quit' to close the program.")
    print("="*50 + "\n")
    
    counter = 1
    while True:
        try:
            user_input = input("Text to synthesize: ")
            
            if user_input.lower().strip() in ['exit', 'quit']:
                print("Exiting...")
                break
                
            if not user_input.strip():
                continue
                
            output_file = f"cloned_output_{counter}.wav"
            
            # ==========================================
            # 3. INSTANT GENERATION LOOP
            # ==========================================
            engine.speak(user_input, output_file)
            print(f"[+] Done! Saved to {output_file}\n")
            
            counter += 1
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
