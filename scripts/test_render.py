import os
import sys
sys.path.append("/work/cvcs2026/VenusIA/openvla/LIBERO")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(0)

libero_bddl_base = "/work/cvcs2026/VenusIA/openvla/LIBERO/libero/libero/bddl_files"
bddl_path = os.path.join(libero_bddl_base, task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(
    bddl_file_name=bddl_path,
    camera_heights=256,
    camera_widths=256
)

obs = env.reset()

img_array = obs["agentview_image"][::-1, :, :]
Image.fromarray(img_array).save("libero_render_test.png")
print("Render completato! Frame salvato come libero_render_test.png")
