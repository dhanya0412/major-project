# Does Reasoning Reduce Hallucination?

A study of reasoning-based prompting strategies for open-weight vision-language models (VLMs) evaluated on [HallusionBench](https://github.com/tianyi-lab/HallusionBench).

This repository contains the notebooks, model outputs, evaluation scripts, and additional analyses for investigating how prompting strategies affect VLM performance on visual question answering tasks designed to expose language hallucination and visual illusion.

## Research questions

The project investigates:

- How do open-weight VLMs perform on HallusionBench under a zero-shot, Yes/No answering setup?
- Do reasoning-based prompting strategies improve answer accuracy compared with zero-shot prompting?
- Do prompting strategies change the models' tendency to answer **Yes** or **No**, even when overall accuracy does not improve?
- How stable are model errors across repeated samples, and how sensitive are results to answer-format failures?

The central focus is not only whether accuracy changes, but also how the *type and distribution of errors* change.

## Models evaluated

The zero-shot evaluation includes:

| Model | Hugging Face checkpoint |
|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| InternVL3-8B | `OpenGVLab/InternVL3-8B-hf` |
| Phi-3.5-Vision | `microsoft/Phi-3.5-vision-instruct` |

The prompting-strategy comparison currently covers **Qwen2.5-VL-7B, InternVL3-8B, and Phi-3.5-Vision**. LLaVA is included in the zero-shot baseline, not the Phase 3 prompting comparison.

## Dataset

Experiments use [HallusionBench](https://huggingface.co/datasets/lmms-lab/HallusionBench), an evaluation benchmark for hallucination and visual reasoning in large vision-language models.

The notebooks load the dataset with:

```python
from datasets import load_dataset

dataset = load_dataset("lmms-lab/HallusionBench")
```

The evaluation uses the `image` and `non_image` splits. Questions have Yes/No ground-truth labels, and model outputs are parsed into the same binary representation:

- `1`: Yes
- `0`: No
- `-1` / unclear: answer could not be parsed; treated as incorrect in the Phase 3 comparison.

## Experimental design

### Phase 1 / baseline evaluation

The initial notebook explores the dataset and evaluation workflow. The model-specific notebooks run zero-shot inference and save the raw responses, parsed predictions, question metadata, and ground-truth labels.

For the baseline evaluation, models are asked to answer each question with **Yes or No**. The resulting files are stored under `results/`.

### Phase 3 / prompting strategies

The prompting comparison evaluates three strategies against zero-shot outputs on a matched subset of questions:

1. **Chain-of-Thought (CoT):** asks the model to briefly describe relevant visual details and reason before giving a final Yes/No answer.
2. **Chain-of-Verification (CoVe):** generates verification questions, answers those checks, and then produces a final answer to the original question.
3. **Self-Consistency (SC):** samples multiple reasoning responses and uses the resulting Yes/No votes to select an answer.

The Phase 3 notebook samples whole sets, stratified by subcategory, using a fixed seed. This keeps related questions and image variants together. The same sample configuration is intended to be used across the models.

The current notebook configuration uses:

| Setting | Value |
|---|---:|
| Sampling seed | `0` |
| Fraction of sets per subcategory | `0.25` |
| Minimum sets per subcategory | `2` |
| Self-consistency samples (`K`) | `5` |
| Self-consistency temperature | `0.7` |
| CoVe verification questions | `3` |
| CoT maximum new tokens | `200` |

These settings are defined in `notebooks/phase3_prompting (1).ipynb`.

## Evaluation metrics

The project reports three HallusionBench-style accuracy metrics:

- **aAcc (answer accuracy):** percentage of individual questions answered correctly.
- **qAcc (question-pair accuracy):** percentage of question groups for which every image variant of the same question is answered correctly. The grouping uses category, subcategory, set, and question identifiers.
- **fAcc (figure accuracy):** percentage of figures for which all associated questions are answered correctly. The VS no-image entries with `figure_id = 0` are excluded, following the benchmark evaluation convention used by the project.

Additional analyses examine:

- **Yes/No response bias:** predicted Yes share minus ground-truth Yes share, in percentage points.
- **Class-wise accuracy and balanced accuracy:** performance on true-Yes and true-No questions.
- **Answer transitions:** wrong-to-right and right-to-wrong changes relative to zero-shot predictions.
- **Set-level bootstrap intervals:** 95% percentile confidence intervals computed using **1,000 set-level resamples** in `results/phase3_compare.py`.
- **Pair sensitivity:** whether predictions change across control-pair image variants with different ground-truth answers.
- **Self-consistency agreement:** accuracy on unanimous versus split-vote questions.
- **Format robustness:** comparison of parseable-only scores and zero-shot fallback handling for unparseable outputs.

Aggregate Yes/No bias is treated as a diagnostic of response tendencies, not as a direct measure of hallucination or visual grounding.

## Repository structure

```text
major-project/
├── notebooks/
│   ├── 01_week1.ipynb
│   ├── 02_qwen.ipynb
│   ├── 03_InternVL2_5_8B.ipynb
│   ├── 04_phi35.ipynb
│   └── phase3_prompting (1).ipynb
├── results/
│   ├── *_results.json
│   ├── *_metrics*.json
│   ├── *_cot.json
│   ├── *_cove.json
│   ├── *_sc.json
│   ├── phase3_compare.py
│   ├── phase3_summary.csv
│   ├── phase3_flips.csv
│   ├── phase3_by_subcategory.csv
│   ├── extra_A_pair_insensitivity.csv
│   ├── extra_B_sc_agreement.csv
│   └── extra_C_format_robust.csv
├── analysis.py
└── results.py
```

The exact set of generated files may change as experiments are rerun. Large model-output JSON files are retained in `results/` for evaluation and analysis.

## Getting started

### Environment

The experiments were developed in notebook-based GPU environments (including Google Colab). Model inference requires a compatible GPU and sufficient VRAM. The notebooks use Hugging Face Transformers and model-specific processing utilities.

The baseline setup notebook installs:

```bash
pip install datasets transformers accelerate bitsandbytes pillow
```

The Qwen notebook additionally installs:

```bash
pip install qwen-vl-utils
```

The Phase 3 notebook installs:

```bash
pip install "bitsandbytes>=0.46.1" datasets accelerate
```

For reproducible runs, use the model and library versions that are compatible with the checkpoint and notebook being executed. The repository does not currently provide a pinned `requirements.txt`.

### Clone the repository

```bash
git clone https://github.com/dhanya0412/major-project.git
cd major-project
```

### Run the notebooks

Open the notebooks in Google Colab or another Jupyter environment with GPU access. The baseline notebooks load HallusionBench, load the corresponding Hugging Face model, run inference over the dataset splits, and save outputs to JSON.

The repository includes Colab links in the first notebook. For Phase 3, use one model configuration at a time and run the notebook cells in order. The Phase 3 notebook is designed to resume from existing output files after an interrupted run.

**Note:** Model inference can be computationally expensive. The notebooks use 4-bit quantization for selected model runs to reduce memory requirements, but available GPU memory and runtime limits still affect feasibility.

## Re-running the Phase 3 comparison

The comparison script does not load a model or require a GPU. It reads the zero-shot and prompting-strategy JSON files, aligns the runs on their common question keys, calculates metrics, and writes summary tables and figures.

The script expects the following files in its working directory:

```text
qwen2.5vl_results.json
internvl3_8b_results.json
phi35vision_results.json
qwen25vl_cot.json
qwen25vl_cove.json
qwen25vl_sc.json
internvl3_cot.json
internvl3_cove.json
internvl3_sc.json
phi35_cot.json
phi35_cove.json
phi35_sc.json
```

Place the required files in the same directory as `phase3_compare.py`, or update the `DIR` and file paths in the script to match your layout.

Install the analysis dependencies:

```bash
pip install matplotlib pandas
```

Run:

```bash
python phase3_compare.py
```

The script uses 1,000 set-level bootstrap resamples for the 95% percentile confidence intervals and a fixed random seed. It generates:

- `phase3_summary.csv`
- `phase3_flips.csv`
- `phase3_by_subcategory.csv`
- `phase3_metrics.png` — accuracy metrics with bootstrap confidence intervals
- `phase3_yes_rate.png` — predicted Yes share compared with the true Yes share

The script also reports sign-test p-values for answer changes. These are exploratory question-level tests; the script does not apply a cluster-aware inferential test.

## Additional analyses

Run the repository-level extra analyses with:

```bash
python analysis.py
```

This script uses the Phase 2 and Phase 3 result JSON files in `results/` and produces:

- `extra_A_pair_insensitivity.csv`
- `extra_B_sc_agreement.csv`
- `extra_C_format_robust.csv`
- `traces_to_label.csv` — a sample of reasoning traces for manual inspection

The trace export can be labelled manually using the label codes described in the script. The optional `export_trace_images()` helper loads HallusionBench from Hugging Face to save the corresponding images; this step requires the dataset and internet access.

## Interpreting the results

The study separates overall accuracy from response tendency and consistency diagnostics. A prompting strategy may reduce Yes/No bias without improving accuracy, or may improve some questions while making others worse. Likewise, agreement across repeated samples does not guarantee that an answer is correct.

For this reason, results should be interpreted across aAcc, qAcc, fAcc, class-wise accuracy, answer transitions, and format robustness rather than using a single metric as evidence that a model hallucinates less.

## References

- Guan et al. **HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models.** CVPR 2024. [Paper](https://arxiv.org/abs/2310.14566)
- [HallusionBench GitHub repository](https://github.com/tianyi-lab/HallusionBench)
- [HallusionBench dataset on Hugging Face](https://huggingface.co/datasets/lmms-lab/HallusionBench)

## Citation

If you use this repository or its experimental setup, please cite the HallusionBench benchmark and refer to this repository:

```bibtex
@inproceedings{guan2024hallusionbench,
  title     = {HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models},
  author    = {Guan, Tianrui and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```

## License

No license file is currently present in the repository. Unless a license is added, the repository's reuse and redistribution terms are not explicitly specified.
