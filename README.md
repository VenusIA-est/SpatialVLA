# VenusIA: OpenVLA Fine-Tuning & Spatial Grounding Analysis

**VenusIA** is a research project focused on fine-tuning and analyzing the spatial reasoning capabilities of the vision-language-action model **OpenVLA-7B** within the **LIBERO-Spatial** simulated environment. The repository includes the entire architectural framework in the `prismatic/` directory, as well as scripts for training, evaluation, and the custom **First-Action Probe** diagnostic suite.

## 📂 Repository Structure

The codebase architecture is organized into the following main directories and files:

* **`environment.yml`** and **`environment_frozen.yml`**: Exported Conda environment files ensuring software reproducibility.

* **`vla-scripts/`**: Directory containing deployment and training scripts, such as `finetune.py`, `train.py`, and `deploy.py`.

* **`slurm_jobs/`**: Batch submission scripts for execution on GPU clusters, including `train_spatial.sh`, `run_eval.sh`, and `probe_job.sh`.

* **`experiments/robot/libero/`**: Modules for simulated evaluation, including `run_libero_eval.py`, `regenerate_libero_dataset.py`, `libero_utils.py`, and `libero_requirements.txt`.

* **`scripts/`**: A suite of diagnostic scripts and utilities, featuring `extract_from_rlds.py`, `probe_variants.py`, `test_render.py`, and `task_enumerator.py`.

* **`img/`**: Directory storing the extracted frames used for the probing tests, containing visual files like `task_0_step0.jpg`, `task_1_step0.jpg`, and `task_3_step0.jpg`.

* **`results/`**: Dedicated folder for saving text logs and experiment results, containing files such as `eval_50k_out_143512.txt`, `probe_out_162600.txt`, and `vanilla_out_154022.txt`.

## 🛠️ Environment Setup

To ensure strict software reproducibility, the repository provides two specific files to recreate the dependencies:

* **Portable Environment:** Use the `environment.yml` file for standard, flexible installations across different workstations.

* **Exact Environment for HPC (Cluster):** Use the `environment_frozen.yml` file to rigorously replicate the exact versions of the locked libraries used on the compute node.

## 🚀 Training (LoRA Fine-Tuning)

Training procedures on LIBERO-Spatial are handled by the Python modules located in `vla-scripts/`. On an HPC node, you can submit the distributed training via the preconfigured Slurm job by executing `sbatch slurm_jobs/train_spatial.sh`.
For direct execution or command-line debugging, the entry point for configuring weights and the LoRA adapter is `vla-scripts/finetune.py`.

## 📊 Evaluation (Closed-Loop)

Evaluating the Success Rate within the Robosuite/MuJoCo simulator requires submitting the job via the `slurm_jobs/run_eval.sh` file. This job internally calls the execution core `experiments/robot/libero/run_libero_eval.py` to test the generated checkpoints in the 3D environment. Text results and error logs from these evaluations (both zero-shot and post-fine-tuning) are automatically archived in the `results/` directory (e.g., `vanilla_err_154022.txt` and `eval_50k_out_143512.txt`).

## 🎯 First-Action Probe

The diagnostic suite dedicated to *targeted semantic probing* is isolated within the `scripts/` folder.

1. **Data Extraction:**
   Run the `scripts/extract_from_rlds.py` script to extract the initial images (Step 0) and their corresponding Ground Truth action vectors from the TFRecord format. The resulting frames are processed and saved in the `img/` folder.

2. **Inference and MSE Calculation:**
   Launch the experiment by submitting the `slurm_jobs/probe_job.sh` file to the GPU partition. This job runs `scripts/probe_variants.py` to calculate the Mean Squared Error (MSE) caused by the textual prompt permutations. The complete logs of the experiment, including any error traces, are saved in the `results/` folder under `probe_out_162600.txt` and `probe_err_162600.txt`.