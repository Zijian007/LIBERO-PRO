<div align="center">
<img src="https://github.com/Zxy-MLlab/LIBERO-OOD/blob/master/images/libero_ood_logo.png" width="360">


<p align="center">
<a href="https://github.com/Zxy-MLlab/LIBERO-OOD/actions">
<img alt="Tests Passing" src="https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg" />
</a>
<a href="https://github.com/Zxy-MLlab/LIBERO-OOD/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/Lifelong-Robot-Learning/LIBERO" />
</a>
<a href="https://github.com/Zxy-MLlab/LIBERO-OOD/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/Lifelong-Robot-Learning/LIBERO?color=0088ff" />

## **Mind the Repetition Trap: Exposing Libero’s Overestimation of Vision‑Language‑Action Model Competence**

xxx

[[Website]](https://libero-project.github.io)
[[Paper]](https://arxiv.org/pdf/2306.03310.pdf)
[[Docs]](https://lifelong-robot-learning.github.io/LIBERO/)
______________________________________________________________________
![pull_figure](https://github.com/Zxy-MLlab/LIBERO-OOD/blob/master/images//libero_ood.png)
</div>

We propose **LIBERO-OOD**—a plug-and-play benchmark built on the LIBERO—designed to offer a more comprehensive and flexible environment for assessing the generalization capabilities of models.​ LIBERO-OOD enables holistic robotic capability assessment via five core generalization dimensions, with rational combinatorial evaluation rules to ensure meaningful analysis:​

- Object Generalization: A new asset library for LIBERO’s four original tasks, created by modifying object appearance, size, and color, to test adaptation to object variations.​
- Position Generalization: Alternative spatial regions for manipulable objects (aligned with physical constraints/task definitions) to evaluate the model’s ability to handle position changes.​
- Semantic Generalization: Three paraphrased variants per task instruction to verify accuracy in understanding natural language semantic variations.​
- Task Generalization: Redesigned feasible task logics, with new object sets and target states, to examine adaptation to task paradigm changes.​
- Environment Generalization: Random cross-task substitution of LIBERO’s five built-in environments to test robustness across scenarios.

We therefore call on the community to adopt more challenging and fair evaluation standards, in order to drive VLA models toward true generalization and comprehension.

---


# Contents

- [Installation](#Installation)
- [Datasets](#Dataset)
- [Getting Started](#Getting-Started)
  - [Task](#Task)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [Citation](#Citation)
- [License](#License)


# Installtion
Clone the official LIBERO-OOD repository by run:
```
git clone https://github.com/Zxy-MLlab/LIBERO-OOD/
```
LIBERO-OOD is developed based on the original LIBERO benchmark, so it uses the same runtime environment as LIBERO—no separate environment configuration for LIBERO-OOD is needed. You only need to install the environment in accordance with LIBERO’s official requirements, as shown below:

Please run the following commands in the given order to install the dependency for **LIBERO**.
```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Zxy-MLlab/LIBERO-OOD/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the `libero` package:
```
pip install -e .
```

# Datasets
We provide high-quality human teleoperation demonstrations for the four task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.

**NEW!!!**

Alternatively, you can download the dataset from HuggingFace by using:
```python
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

This option can also be combined with the specific dataset selection:
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
```

The datasets hosted on HuggingFace are available at [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).

## Evaluation

To specify single-type or combined-type generalization evaluation, you only need to modify the evaluation_config.yaml configuration file in the project directory. The core configuration parameters and their functions are as follows:

In evaluation_config.yaml, adjust the boolean values ( true/false ) of the following parameters to enable or disable specific generalization evaluation types:

| Parameter | Function |
| ----------------- | -------------------------------------------------------------------------------------- |
| use_environment | Enable (true) or disable (false) environment generalization evaluation |
| use_swap | Enable (true) or disable (false) position generalization evaluation |
| use_object | Enable (true) or disable (false) object generalization evaluation |
| use_language | Enable (true) or disable (false) semantic (language) generalization evaluation |
| use_task | Enable (true) or disable (false) task generalization evaluation |

Note: to avoid meaningless evaluation results, task generalization (use_task: true) cannot be combined with any other generalization types.

Below is a reference code snippet for conducting LIBERO-OOD generalization evaluation.
```
...
with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
  evaluation_cfg = yaml.safe_load(f)

evaluation_cfg["bddl_files_path"] = evaluation_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name
evaluation_cfg["task_suite_name"] = cfg.task_suite_name

if not os.path.exists(evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_temp/"):
  perturbation.create_env(
    configs=evaluation_cfg,
  )

cfg.task_suite_name = cfg.task_suite_name + "_temp"
...
```

# Citation
If you use LIBERO-OOD in your research, please cite both the original LIBERO benchmark (as LIBERO-OOD is fully built upon it) and the LIBERO-OOD paper:

Cite LIBERO
```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```
Cite LIBERO-OOD
```
@article{[your-lastname]202Xliberood,
  title={LIBERO-OOD: [Your LIBERO-OOD Paper Title, e.g., A Plug-and-Play Benchmark for Robotic Out-of-Distribution Generalization]},
  author={[Your Name] and [Co-Author 1] and [Co-Author 2] and [Corresponding Author]},
  journal={[Journal Name / arXiv preprint]},
  volume={[Volume]},
  number={[Number]},
  pages={[Pages]},
  year={202X},
  publisher={[Publisher]} / eprint={[arXiv ID]}
}
```

# License
| Component        | License                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Codebase         | [MIT License](LICENSE)                                                                                                                      |
| Datasets         | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)                 |
