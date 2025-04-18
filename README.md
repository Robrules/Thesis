# Voice‑Privacy‑VAE

**Protecting gender identity in voice‑controlled digital assistants by disentangling pitch with a Variational Autoencoder**

---

## Why this matters  
Voice‑controlled digital assistants (VCDAs) like Alexa and Siri continuously absorb our speech. Beyond the words themselves, those recordings leak biometric cues—pitch, timbre, rhythm—that let an attacker infer gender, age, mood, even body size. The thesis in `thesis.pdf` asks a simple question:

> *Can we transform a user’s voice so gender cannot be inferred while keeping the command intelligible?*

---

## Thesis contribution (condensed)
* Conducted a literature review on attribute‑inference attacks and existing defences (signal processing vs. disentangled representation learning).
* Built a **sound‑based Variational Autoencoder (VAE)** trained on the Fluent Speech Commands dataset.
* Injected pitch‑shifted duplicates into training to help the model isolate *pitch* in its 16‑D latent space.
* Identified the latent variable most sensitive to pitch, edited it at inference, and reconstructed audio with Griffin‑Lim.
* **Privacy result:** average gender‑classifier confidence ⬇︎ to **48 %** (≈ random guess).  
* **Utility cost:** Siri word‑error rate ↑ to **45 %**—a clear privacy‑utility trade‑off.

---

## How the pipeline works
```
raw .wav
  ↓  (librosa STFT + dB scaling)
log‑spectrogram 256×256
  ↓  Encoder (FC → 1000 → 16)
16‑D latent   ← μ, σ reparameterisation
  ↓  tweak pitch‑sensitive variable(s)
  ↓  Decoder (16 → 1000 → 65536)
reconstructed spectrogram
  ↓  Griffin‑Lim ISTFT
new .wav with hidden gender
```
A simplified diagram lives at `docs/diagram.png`.

---

## Repository map
| Path | Purpose |
|------|---------|
| `notebooks/Main.ipynb` | end‑to‑end experiment: preprocessing → training → evaluation |
| `utils/preprocess.py`  | implements the five‑stage audio pipeline described in § 6.1.2 |
| `utils/generate.py`    | `SoundGenerator` class for spectrogram → waveform (§ 6.1.7) |
| `spectrograms/`        | generated log‑spectrogram PNGs + min‑max stats |
| `thesis.pdf`           | full 40‑page write‑up (methods, results, discussion) |

---

## Key numbers
| Metric | Score |
|--------|-------|
| Gender‑classifier confidence | **48.4 %** |
| Word Error Rate (Siri)       | 45.5 % |
| Character Error Rate         | 35 % |
| Training loss (epoch 1000)   | ≈17 500 |
| Validation loss              | ≈20 800 |

---

## Limitations & next steps
* Pitch is **not fully disentangled**; edits introduce audible artefacts.
* Trained on **922 utterances, 2 speakers**—model overfits; larger corpora (LibriSpeech, VCTK) needed.
* Replace FC VAE with **Conv‑VAE or VQ‑VAE** for sharper reconstructions.
* Swap Griffin‑Lim for a neural vocoder (WaveGlow, HiFi‑GAN) to cut WER.
* Extend disentanglement to other attributes (age, emotion) for configurable privacy.

---

## Citation
```bibtex
@mastersthesis{bannayan2025vae,
  title  = {Voice‑Controlled Digital Assistants: Pitch‑Aware VAE for Gender Privacy},
  author = {Robert Bannayan},
  school = {University of Sydney},
  year   = {2025}
}
```

---

MIT License · 2025 Robert Bannayan
