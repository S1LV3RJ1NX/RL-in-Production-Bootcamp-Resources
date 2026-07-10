"""One-shot rendering diagnostic for headless VizDoom on Modal.

Tells us exactly why the screen buffer is coming back a flat uniform gray:
  * is Xvfb actually running? what did it log?
  * does the raw VizDoom screen_buffer vary (n_unique > 1) under different
    screen formats / with SOFTWARE_RENDERING?

Run:  .venv/bin/modal run modal_apps/doom_diag.py
"""
import json
import modal
from common import app, IMAGE, VOLUMES, GPU_CHEAP, jsonify


@app.function(image=IMAGE, gpu=GPU_CHEAP, volumes=VOLUMES, timeout=60 * 10)
def diag():
    import os, subprocess, time, shutil, numpy as np
    rep = {}
    rep["DISPLAY_before"] = os.environ.get("DISPLAY")
    rep["which_Xvfb"] = shutil.which("Xvfb")
    rep["LIBGL_ALWAYS_SOFTWARE"] = os.environ.get("LIBGL_ALWAYS_SOFTWARE")

    # start Xvfb ourselves, capturing its log
    os.makedirs("/tmp/.X11-unix", exist_ok=True)
    logf = open("/tmp/xvfb.log", "w")
    p = subprocess.Popen(["Xvfb", ":99", "-screen", "0", "640x480x24", "-nolisten", "tcp"],
                         stdout=logf, stderr=subprocess.STDOUT)
    os.environ["DISPLAY"] = ":99"
    time.sleep(2.5)
    rep["xvfb_alive"] = (p.poll() is None)
    rep["x_socket_exists"] = os.path.exists("/tmp/.X11-unix/X99")
    logf.flush()
    try:
        rep["xvfb_log_tail"] = open("/tmp/xvfb.log").read()[-800:]
    except Exception as e:
        rep["xvfb_log_tail"] = f"<{e}>"

    import vizdoom as vzd

    def try_cfg(fmt, software_render):
        try:
            g = vzd.DoomGame()
            g.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
            g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
            g.set_screen_format(fmt)
            g.set_window_visible(False)
            try:
                g.set_render_all_frames(True)
            except Exception:
                pass
            # try to force VizDoom's own software renderer (no OpenGL) if available
            try:
                g.add_game_args("+vid_renderer 0" if software_render else "+vid_renderer 1")
            except Exception:
                pass
            g.init()
            g.new_episode()
            for _ in range(12):
                if g.is_episode_finished():
                    break
                g.make_action([0, 1], 4)
            s = g.get_state()
            out = {"state_is_none": s is None}
            if s is not None:
                b = np.asarray(s.screen_buffer)
                out.update({"shape": list(b.shape), "dtype": str(b.dtype),
                            "min": int(b.min()), "max": int(b.max()),
                            "mean": round(float(b.mean()), 2),
                            "n_unique": int(np.unique(b).size)})
            g.close()
            return out
        except Exception as e:
            return {"error": repr(e)}

    rep["RGB24_glon"] = try_cfg(vzd.ScreenFormat.RGB24, software_render=False)
    rep["RGB24_swrender"] = try_cfg(vzd.ScreenFormat.RGB24, software_render=True)
    rep["CRCGCB_glon"] = try_cfg(vzd.ScreenFormat.CRCGCB, software_render=False)
    return jsonify(rep)


@app.local_entrypoint()
def main():
    print(json.dumps(diag.remote(), indent=2))
