"""Single deploy target that registers ALL functions on the shared `iris-wm` app.

Deploying an individual file (train.py OR train_controller.py) replaces the
deployment with only that file's functions.  Importing every function-defining
module here means `modal deploy modal_apps/deploy_all.py` registers train +
train_controller + evals + baseline together, so all can be spawned server-side.
"""
from common import app          # the shared modal.App("iris-wm")
import train                    # noqa: F401  (registers collect/tokenizer/wm/policy/iris_loop/k256_build)
import train_controller         # noqa: F401  (registers train_controller)
import evals                    # noqa: F401  (registers evaluate/render_dreams/...)
# NB: don't import modules with a duplicate `main` local_entrypoint (baseline_modelfree,
# doom_* diagnostics) — deploy only needs the @app.function definitions, and those
# functions are already available; the diagnostics are run ad hoc via `modal run`.

# `app` is what `modal deploy` deploys; the imports above attach their functions.
