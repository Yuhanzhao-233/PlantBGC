# PlantBGC: Transformer for Plant BGC Discovery via Label-Free Domain Adaptation and Weak Supervision

PlantBGC detects candidate biosynthetic gene clusters (BGCs) in **plant genomes** by modeling genomes as **ordered Pfam-domain sequences** and scoring BGC-likeness with an **encoder-only Transformer**.
The framework supports (i) microbial supervised pretraining, (ii) **label-free plant domain adaptation** via masked language modeling (MLM), and (iii) optional weak-supervision strategies to reduce primary-metabolism false positives while preserving secondary-metabolism signals.

> This repository contains the training/evaluation scripts used in our paper: **“PlantBGC”** .

---

## Key ideas (paper-level summary)

* **Representation**: genome → ordered Pfam tokens (domain sequence)
* **Stage 1 (microbial supervision)**: train a Transformer detector on curated microbial BGC positives vs negatives
* **Stage 2 (plant adaptation, no plant labels)**: continue pretraining on unlabeled plant Pfam sequences with **MLM** to align plant Pfam co-occurrence statistics
* **Stage 3 (weak supervision, optional)**: inject soft negatives (e.g., GO/KEGG-based) to reduce “primary-like” false positives

---

## What you get

- **Candidate CDS and BGC loci** 
- **BGC-likeness scores** (Transformer-based) for ranking and triage
- *(Optional)* **GO/KEGG proxy labels** for analysis / weak supervision: (Details see [GO/KEGG Proxy Labels](#gokegg-proxy-labels-optional).)
  - KEGG proxy is reported as **primary / secondary / mixed / review**
  - GO proxy follows the project label scheme

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

> If you already have a Predict that outputs `*.pfam.tsv`, keep it. PlantBGC mainly cares about **order + Pfam IDs + sequence grouping**.

---

## Data acquisition (NCBI / RefSeq)

If you want to test a new genome and obtain **Transformer scoring results**, download FASTA from NCBI as follows.

### Plant genomes

Download the **RefSeq CDS** file:

- **Genomic coding sequences (FASTA)**

This file is recommended for plant runs because PlantBGC produces **CDS/gene-level scores** and then aggregates them into **candidate loci**.

### Microbial genomes

Download the genome sequence file:

- **Genome sequences (FASTA)**

For microbial inputs, PlantBGC can score across the genome sequence workflow as supported by the repo configuration.

---

## Inputs

PlantBGC typically uses:

### Required

- **FASTA**
  - Plants: CDS FASTA (**Genomic coding sequences (FASTA)**)
  - Microbes: Genome FASTA (**Genome sequences (FASTA)**)

### Recommended (for genomic coordinates)

- **Genome annotations** (e.g., **GFF/GBFF**)  
  Used to map CDS units back to genomic coordinates for locus-level reporting.

### Optional (for proxy labeling / weak supervision)

- **GO/KEGG annotation tables**  
  Used for proxy labeling, weak supervision, and downstream enrichment summaries.

---

## Outputs

PlantBGC outputs are organized at three levels:

1. **Token / Pfam-level scores**
2. **CDS (gene)-level scores**
3. **Candidate loci (candidate BGC loci)** aggregated from consecutive high-scoring CDS

---

## Locus aggregation rule (paper-aligned)

A candidate locus is formed by:

- pooling **Pfam-level** scores into a **CDS-level** score
- selecting CDS with **score > 0.5**
- aggregating **strictly consecutive** selected CDS into loci
- enforcing **minimum length = 3 CDS** and **gap = 0**

---

## GO/KEGG Proxy Labels (optional)

PlantBGC can optionally generate **proxy labels** from functional annotations to support analysis and weak supervision.
In this project, **KEGG proxy is reported as _primary / secondary / mixed / review_** (paper-aligned).

> GO proxy will be documented in a separate section (to be added).

### KEGG proxy: how to obtain KO numbers

To produce KEGG proxy labels, you first need **KO numbers** (K identifiers) for genes/proteins in your predicted loci.

#### Recommended (simplest): KEGG web mapping (BlastKOALA / GhostKOALA)

Local KO mapping can be non-trivial due to database licensing/availability, profile resources, and environment setup.
For most users, we recommend using the official KEGG KO assignment service:

1. Prepare a FASTA file of coding sequences/proteins to annotate  
   - For plants, we typically start from **RefSeq CDS FASTA** and then use the translated proteins (or protein FASTA if available).  
2. Upload the FASTA to **BlastKOALA** (or **GhostKOALA** for large-scale submissions)  
3. Download the KO assignment output (gene/protein → KO number mapping)  
4. Feed the KO mapping file into PlantBGC to generate locus-level KEGG proxy labels

PlantBGC will then aggregate KO-derived category signals within each **candidate BGC locus** and output a locus-level label:
**primary / secondary / mixed / review**.

#### Optional: local deployment (advanced)

If you prefer a fully local Predict, KO mapping can be performed with tools such as:
- **KofamScan** (profile HMM-based KO assignment using KOfam HMMs)
- other HMMER-based KO assignment workflows

For most workflows, the web-based KO assignment above is the fastest path to obtain KO numbers.

### Generating KEGG proxy labels in PlantBGC

Once you have the KO mapping (gene/protein → KO), run PlantBGC’s proxy module to:
- map KO to functional categories used by our proxy scheme
- aggregate per-gene evidence into **locus-level** proxy labels
- export per-locus label summaries for downstream analysis

> The exact command depends on the repo version. See `scripts/` (or the Predict entrypoint) for the KEGG proxy labeling utility.



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

This project builds on ideas from genome mining and domain-sequence modeling, and is inspired by prior BGC detection Predicts in microbes.
