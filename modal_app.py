"""Modal deployment of the French Lab helper (app.py).

    modal deploy helper/modal_app.py

Secrets come from the Modal secret "french-lab-helper" (CONNECTOR_KEY, EUDIC_TOKEN,
ELEVENLABS_API_KEY, optional BILI_SESSDATA). The web function scales to zero when idle.
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "curl")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    # Piper: free, open-source text-to-speech with a French voice baked into the image (no credits, no account)
    .pip_install("piper-tts>=1.3")
    .run_commands("mkdir -p /voices && cd /voices"
                  " && curl -sSL -O https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx"
                  " && curl -sSL -O https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json")
    .add_local_file(str(HERE / "app.py"), "/root/app.py")
)

app = modal.App("french-lab-helper", image=image)

# Keys added or replaced later with `deploy_helper.py set-key <name>` live in their own secrets.
# The list must be identical here and inside the container, so it is fixed; deploy_helper.py creates
# an empty placeholder for any of these that has not been set yet.
KEY_SECRETS = ("supadata", "elevenlabs", "eudic", "bilibili", "groq")
SECRETS = [modal.Secret.from_name("french-lab-helper")] + [modal.Secret.from_name(f"french-lab-{n}") for n in KEY_SECRETS]


DATA = modal.Volume.from_name("french-lab-data", create_if_missing=True)


@app.function(secrets=SECRETS, scaledown_window=300, timeout=600, cpu=2.0, memory=2048,
              volumes={"/data": DATA}, env={"FL_DATA_DIR": "/data"})
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root")
    import app as helper

    return helper.app
