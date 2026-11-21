# VihAI 2.0

Streamlit assistant with optional multi-model answer fusion and 3D Studio.

## Setup (Windows)
1. Install Python and open Command Prompt in this folder.
2. Run `python -m pip install -r requirements.txt`.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in keys.
4. Run `streamlit run "VihAI maximum.py"`.

## Models
- Qwen3-32B defaults to `qwen/qwen3-32b` on Groq.
- `QWEN27_MODEL` and `NEMOTRON_MODEL` are intentionally blank: **verify exact IDs and availability with Groq and NVIDIA** before setting them. The requested names may not exist as stated.
- All configured models run in parallel; a successful model synthesizes one answer. Failed models are skipped if another succeeds. Each request may use multiple billable API calls.
- Nemotron uses NVIDIA's OpenAI-compatible endpoint. Speech transcription always uses Groq.

## 3D
- Local mode generates only primitive geometry (sphere, cylinder, cone, box).
- Cloud mode uses Tripo text-to-3D and image-to-3D via `TRIPO_API_KEY`; API credits or billing may be required.
- Hybrid tries Tripo first, then falls back to primitive geometry; it does not produce an equivalent AI model when cloud fails.
- Interactive GLB preview uses an externally hosted model-viewer script, so an internet connection is needed.
- The existing Video Studio remains a storyboard/GIF tool, not generative video.

Never share `.streamlit/secrets.toml`.

## Tripo setup
Add `TRIPO_API_KEY` to `.streamlit/secrets.toml` (copy the example). Do not share the real key. Tripo cloud generation uses the Tripo task API and may consume credits. Image-to-3D needs cloud access; local fallback only works for text prompts and generates primitives. API endpoints and supported image upload formats can change; consult the current Tripo API documentation if a request fails.
