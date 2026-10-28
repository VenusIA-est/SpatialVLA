import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import json

model_path = "openvla/openvla-7b-finetuned-libero-spatial"
print(f"Caricamento modello ufficiale da: {model_path}...")

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# --- BLOCCO INIEZIONE (Scommenta questo blocco SOLO se stai testando il modello VANILLA) ---
# cfg_unnorm_key = "libero_spatial"
# if cfg_unnorm_key not in vla.norm_stats:
#     with open("/work/cvcs2026/VenusIA/openvla/checkpoints/openvla_run2_50k_steps/openvla-7b+libero_spatial_no_noops+b12+lr-0.0005+lora-r32+dropout-0.0--image_aug/config.json", "r") as f:
#         ft_config = json.load(f)
#     stats_key = list(ft_config["norm_stats"].keys())[0]
#     vla.norm_stats[cfg_unnorm_key] = ft_config["norm_stats"][stats_key]
# --------------------------------------------------------------------------------------------

# 2. Configurazione dei Task, dei Ground Truth e delle Varianti Semantiche
experiments = {
    "TASK_0 (Between)": {
        "image": "task_0_step0.jpg",
        "gt": np.array([0.12321, 0.25714, 0.15536, -0.045, 0.01179, -0.0, -1.0]),
        "prompts": {
            "Original": "In: What action should the robot take to pick up the black bowl between the plate and the ramekin and place it on the plate?\nOut:",
            "Swapped":  "In: What action should the robot take to pick up the black bowl next to the ramekin and place it on the plate?\nOut:",
            "Removed":  "In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:"
        }
    },
    "TASK_3 (On Cookie Box)": {
        "image": "task_3_step0.jpg",
        "gt": np.array([0.31607, 0.18482, 0.11518, -0.01179, 0.0, -0.0, -1.0]),
        "prompts": {
            "Original": "In: What action should the robot take to pick up the black bowl on the cookie box and place it on the plate?\nOut:",
            "Swapped":  "In: What action should the robot take to pick up the black bowl next to the cookie box and place it on the plate?\nOut:",
            "Removed":  "In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:"
        }
    },
    "TASK_1 (Next to Ramekin)": {
        "image": "task_1_step0.jpg",
        "gt": np.array([0.27321, 0.36429, -0.0, -0.01179, -0.05143, -0.0, -1.0]),
        "prompts": {
            "Original": "In: What action should the robot take to pick up the black bowl next to the ramekin and place it on the plate?\nOut:",
            "Swapped":  "In: What action should the robot take to pick up the black bowl next to the plate and place it on the plate?\nOut:",
            "Removed":  "In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:"
        }
    }
}

# 3. Inferenza e Calcolo degli Errori
print("\nInizio First-Action Probe...\n" + "="*50)

for task_name, data in experiments.items():
    print(f"\n{task_name}")
    print("-" * len(task_name))
    
    image = Image.open(data["image"]).convert("RGB")
    gt_action = data["gt"]
    
    # Stampiamo solo le coordinate XYZ per una lettura più immediata
    print(f"Ground Truth (X, Y, Z): [{gt_action[0]:.4f}, {gt_action[1]:.4f}, {gt_action[2]:.4f}]")
    
    for variant, prompt in data["prompts"].items():
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        
        # Genera l'azione predetta
        pred_action_tensor = vla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
        pred_action = np.array(pred_action_tensor)
        
        # Calcolo MSE solo sulle coordinate XYZ (Traslazione Spaziale)
        mse_xyz = np.mean((pred_action[:3] - gt_action[:3])**2)
        
        print(f"  {variant.ljust(8)} | Pred: [{pred_action[0]:.4f}, {pred_action[1]:.4f}, {pred_action[2]:.4f}] | MSE(xyz): {mse_xyz:.6f}")
        
print("\n" + "="*50 + "\nTest completato.")