import modal

app = modal.App("rlhf-smoke")
image = modal.Image.debian_slim()

@app.function(gpu="A10G", image=image, timeout=120)
def gpu_check():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True)
    return out.stdout.strip() or out.stderr.strip()

@app.local_entrypoint()
def main():
    print("GPU SEEN BY MODAL:", gpu_check.remote())
