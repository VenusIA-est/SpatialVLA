import sys
sys.path.append("/work/cvcs2026/VenusIA/openvla/LIBERO")
from libero.libero import benchmark

benchmark_dict = benchmark.get_benchmark_dict()

spatial_suite = benchmark_dict["libero_spatial"]()

task_names = spatial_suite.get_task_names()
for i, name in enumerate(task_names):
    print(f"Task {i}: {name}")