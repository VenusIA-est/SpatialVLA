import os
import sys
sys.path.append("/work/cvcs2026/VenusIA/openvla/LIBERO")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

# 1. Forza il rendering headless EGL
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 2. Carica la task suite 'libero_spatial'
task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(0)

# 3. Costruisce il percorso usando il percorso assoluto noto dal tuo cluster
libero_bddl_base = "/work/cvcs2026/VenusIA/openvla/LIBERO/libero/libero/bddl_files"
bddl_path = os.path.join(libero_bddl_base, task.problem_folder, task.bddl_file)

# 4. Crea l'ambiente passando esplicitamente i parametri
env = OffScreenRenderEnv(
    bddl_file_name=bddl_path,
    camera_heights=256,
    camera_widths=256
)

obs = env.reset()

# 5. Estrae l'immagine dalla prospettiva dell'agente e la salva
img_array = obs["agentview_image"][::-1, :, :]
Image.fromarray(img_array).save("libero_render_test.png")
print("Render completato! Frame salvato come libero_render_test.png")
