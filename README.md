# PlantBGC: Transformer for Plant BGC Discovery via Label-Free Domain Adaptation and Weak Supervision

PlantBGC detects candidate biosynthetic gene clusters (BGCs) in **plant genomes** by modeling genomes as **ordered Pfam-domain sequences** and scoring BGC-likeness with an **encoder-only Transformer**.
The framework supports (i) microbial supervised pretraining, (ii) **label-free plant domain adaptation** via masked language modeling (MLM), and (iii) optional weak-supervision strategies to reduce primary-metabolism false positives while preserving secondary-metabolism signals.

> This repository contains the training/evaluation scripts used in our paper: **“PlantBGC”** (manuscript in progress).

---

## Key ideas (paper-level summary)

* **Representation**: genome → ordered Pfam tokens (domain sequence)
* **Stage 1 (microbial supervision)**: train a Transformer detector on curated microbial BGC positives vs negatives
* **Stage 2 (plant adaptation, no plant labels)**: continue pretraining on unlabeled plant Pfam sequences with **MLM** to align plant Pfam co-occurrence statistics
* **Stage 3 (weak supervision, optional)**: inject soft negatives (e.g., GO/KEGG-based) to reduce “primary-like” false positives

---

## What you can do with this repo

* Train a BGC-likeness detector on microbial Pfam sequences (Stage 1)
* Adapt the detector to plants using unlabeled plant Pfam sequences (Stage 2)
* Run inference to produce candidate plant loci and scores
* Evaluate recovery/coverage on curated plant BGCs and compare with plantiSMASH (Stage 2/3 experiments)

---

## Installation

### Option A: conda (recommended)

```bash
conda create -n plantbgc python=3.10 -y
conda activate plantbgc
pip install -r requirements.txt
```

### Option B: pip (if you enjoy pain)

```bash
pip install -r requirements.txt
```

> If you rely on external gene calling / Pfam annotation tools, install them separately and ensure they are on PATH.

---

## Data format

### Pfam-domain TSV (required)

We assume a tokenized TSV where each row is a Pfam hit for one protein, and proteins are ordered by genomic position within a contig/chromosome.

**Minimum recommended columns**

* `sequence_id`: ID for a continuous ordered region (e.g., contig/chromosome segment)
* `pfam_id`: Pfam accession (token)
* `protein_id` (or equivalent): optional but helpful for debugging
* `start`, `end` (optional): genomic/protein coordinates if available

> If you already have a pipeline that outputs `*.pfam.tsv`, keep it. PlantBGC mainly cares about **order + Pfam IDs + sequence grouping**.

---

## Quickstart

### 1) Prepare / check your TSV

Put TSVs under something like:

```
data/
  train/
    positives.pfam.tsv
    negatives.pfam.tsv
  plant_unlabeled/
    plant_unlabeled.pfam.tsv
  eval/
    curated_plant_bgcs.pfam.tsv
```

### 2) Train Stage 1 (microbial base)

```bash
python train_stage1.py \
  --pos data/train/positives.pfam.tsv \
  --neg data/train/negatives.pfam.tsv \
  --out checkpoints/stage1.pt
```

### 3) Stage 2 adaptation (MLM on unlabeled plant Pfams)

```bash
python adapt_stage2_mlm.py \
  --init checkpoints/stage1.pt \
  --unlabeled data/plant_unlabeled/plant_unlabeled.pfam.tsv \
  --out checkpoints/stage2.pt
```

### 4) Inference on plant genomes (Pfam TSV)

```bash
python predict.py \
  --model checkpoints/stage2.pt \
  --input data/eval/plant_genome.pfam.tsv \
  --out outputs/plant_candidates.tsv
```

---

## Reproducing paper experiments (recommended layout)

### Stage 1 (microbial)

* 10-fold CV and leave-class-out evaluation
* Compare: Transformer vs DeepBGC-style BiLSTM baseline vs shallow baseline (e.g., RF)

### Stage 2 (plants, no labels for training)

* Evaluate recovery of curated plant BGCs under increasing locus-coverage thresholds
* Compare Stage1-only vs Stage2-adapted
* Compare with plantiSMASH overlap/IoU + compactness (length ratio)

### Stage 3 (weak supervision, optional)

* Add GO/KEGG soft negatives (or other priors) to reduce primary-metabolism false positives
* Compare Stage2 vs Stage3 on false positive rate and curated BGC coverage

> Scripts for each stage should be documented under `scripts/` (or update this README once finalized).

---

## Output

Typical outputs include:

* per-token BGC-likeness scores
* predicted candidate loci with boundaries (start/end in token space)
* summary metrics (ROC-AUC, coverage thresholds, compactness vs plantiSMASH)

---

## Repository structure (suggested)

```
PlantBGC/
  models/                # model definitions
  data_utils/            # TSV parsing, tokenization, masking, batching
  scripts/               # training/adaptation/eval entrypoints
  outputs/               # predictions + evaluation results
  README.md
  requirements.txt
```

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@article{plantbgc2026,
  title   = {PlantBGC: Transformer-based Detection of Plant Biosynthetic Gene Clusters from Pfam-domain Sequences},
  author  = {Zhao, Yuhan and Guo, Zhishan and Sui, Ning and others},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

TBD (choose one: MIT / Apache-2.0 / BSD-3-Clause)

---

## Acknowledgements

This project builds on ideas from genome mining and domain-sequence modeling, and is inspired by prior BGC detection pipelines in microbes.
