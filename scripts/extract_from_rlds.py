import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

# Percorso della tua cartella RLDS
dataset_dir = "/work/cvcs2026/VenusIA/modified_libero_rlds/libero_spatial_no_noops/1.0.0/"
print(f"Caricamento dataset RLDS da: {dataset_dir} ...\n")

builder = tfds.builder_from_directory(dataset_dir)
dataset = builder.as_dataset(split='train')

# Definiamo i target usando le parole chiave che distinguono i 3 task
# (Nel dataset RLDS, le istruzioni solitamente hanno gli spazi al posto degli underscore)
targets = {
    "task_0": {"keyword": "between the plate and the ramekin", "found": False},
    "task_1": {"keyword": "next to the ramekin", "found": False},
    "task_3": {"keyword": "on the cookie box", "found": False}
}

for episode in dataset:
    # Interrompe il ciclo se abbiamo già trovato un esempio per tutti e 3 i task
    if all(info["found"] for info in targets.values()):
        break
        
    # Estraiamo esclusivamente il primo step (Step 0) dell'episodio corrente
    step_iterator = iter(episode['steps'])
    first_step = next(step_iterator)
    
    # Decodifichiamo l'istruzione in testo semplice (spesso memorizzata come bytes in RLDS)
    instruction = first_step['language_instruction'].numpy()
    instruction_str = instruction.decode('utf-8').lower() if isinstance(instruction, bytes) else str(instruction).lower()
    
    # Controlliamo se l'episodio appartiene a uno dei task che stiamo cercando
    for task_id, info in targets.items():
        if not info["found"] and info["keyword"] in instruction_str:
            print(f"--- Trovato {task_id.upper()} ---")
            print(f"Istruzione originaria: '{instruction_str}'")
            
            # 1. Estrazione e salvataggio dell'immagine
            image_tensor = first_step['observation']['image']
            img = Image.fromarray(image_tensor.numpy())
            filename = f"{task_id}_step0.jpg"
            img.save(filename)
            print(f"Salvato frame visivo in: {filename}")
            
            # 2. Estrazione e formattazione dell'azione Ground Truth
            # Vettore a 7 dimensioni: [X, Y, Z, Roll, Pitch, Yaw, Gripper]
            gt_action = first_step['action'].numpy()
            gt_formatted = [round(float(val), 5) for val in gt_action]
            print(f"Azione Ground Truth: {gt_formatted}\n")
            
            info["found"] = True
            break # Passiamo al prossimo episodio per cercare i task rimanenti

print("Estrazione dei dati completata con successo!")