<img src="assets/semantic_context_tokens.png" alt="# Semantic Context Tokens" width="900"><br>
[<img src="https://img.shields.io/badge/Read%20Paper-PDF-black?style=for-the-badge&labelColor=0057FF&logo=adobeacrobat&logoColor=white" alt="Read Paper"/>](https://drive.google.com/file/d/1f_bSPjfhiGOoaf3Gu1gnbcc2W1wNWSec/view?usp=sharing)


A hybrid tokenization framework that combines **coarse semantic context tokens** with **fine-grained sub-word tokens** to improve narrative cohesion and creativity in Large Language Models (LLMs). This research builds upon Meta's [Large Concept Models (LCMs)](https://github.com/facebookresearch/large_concept_model) by exploring a quantized, hierarchical generation paradigm ("big picture to small") mimicking human cognitive processes. This was a collaborative group research project conducted as part of my MSc programme at University College London.

<p>
  <a href="#overview"><img src="https://img.shields.io/badge/Overview-111111?style=for-the-badge" alt="Overview"></a>
  <a href="#methodology"><img src="https://img.shields.io/badge/Methodology-111111?style=for-the-badge" alt="Methodology"></a>
  <a href="#training-configuration"><img src="https://img.shields.io/badge/Training Config-111111?style=for-the-badge" alt="Training Configuration"></a>
  <a href="#results"><img src="https://img.shields.io/badge/Results-111111?style=for-the-badge" alt="Results"></a>
  <a href="#key-findings"><img src="https://img.shields.io/badge/Key Findings-111111?style=for-the-badge" alt="Key Findings"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-111111?style=for-the-badge" alt="Citation"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-111111?style=for-the-badge" alt="License"></a>
</p>

## Overview

With pre-training data becoming increasingly limited, novel model architectures present a promising avenue to further improve language model performance. Large Concept Models (LCMs) have demonstrated competitive results against traditional LLMs by operating on coarse higher-level semantic representations. Our research expands upon LCMs by introducing a **hybrid token approach** that combines coarse semantic tokens with fine-grained tokens for autoregressive text generation.

This framework draws inspiration from how humans typically think—**starting with a high-level idea before filling in the details**. By embedding and clustering sentences to create discrete semantic context tokens, then conditioning language models on these tokens, we encourage a "big picture to small" generation paradigm that benefits creative domains like story generation.

**Key Contributions:**
- **Coarse-to-Fine Generation:** High-level semantic clusters serve as "plans" that condition the generation of detailed text
- **Hierarchical Representation:** Inspired by audio generation (e.g., AudioLM) where multi-level tokenization improves global consistency
- **Resource Efficient:** Benchmarked using the TinyStories dataset, demonstrating that structural supervision can compensate for limited model scale (1M–33M parameters)
- **Statistically Significant Improvements:** 33M parameter models with prepended tokens significantly outperform vanilla baselines (p = 0.029)


## Methodology

### Story Segmentation
Stories are partitioned into chunks using the **SAT (Segment Any Text)** model, which provides punctuation-agnostic sentence splitting. Chunks are processed as either:
- **1-sentence segments** — Fine-grained semantic units
- **2-sentence segments** — Broader narrative context (best performing)

### Data Preprocessing
To ensure semantic clustering focuses on narrative function rather than specific characters:
- **Named Entity Masking:** SpaCy NER identifies proper names and replaces them with `<unk>` tokens
- **Pronoun Substitution:** Regex-based masking of gendered pronouns ('she', 'he', 'her', etc.)

### Embedding & Clustering
Each preprocessed segment is:
1. **Embedded** using the Stella model (`stella-en-1.5B-v5`) into a 1536-dimensional vector space
2. **Clustered** via K-Means into k ∈ {5, 7, 15, 32} semantic groups
3. **Mapped** to discrete learned tokens (e.g., `<cluster_3>`)

### Token Placement Strategies
| Strategy | Description | Best For |
|----------|-------------|----------|
| **Prepending** | Token appears before the segment, requiring the model to "plan" before generating | Larger models (33M) |
| **Postpending** | Token appears after the segment as a "pseudo-summary" | Smaller models (1M) |

**Example — Prepending (2-sentence segments):**
```
<cluster_3> Once upon a time, there was a very loyal dog. His name was Spot.
<cluster_0> One day, he saw a very shiny rock. He picked it up and started to polish it.
<cluster_4> But then, something peculiar happened. The rock suddenly started to grow larger.
<cluster_2> They had become friends and they were going to stay connected forever.
```

### Emergent Cluster Semantics

Our best-performing configuration (2-sentence, 5 clusters) produced interpretable narrative groupings:

| Cluster | Narrative Function | Example |
|---------|-------------------|---------|
| 3 | **Introduction** | "Once upon a time, there was a wise man..." |
| 0 | **Plot Setting** | "One day, he saw a very shiny rock..." |
| 4 | **Twist** | "But then, something peculiar happened..." |
| 2 | **Resolution** | "They had become friends and they were going to stay connected forever." |
| 1 | **Ending** | "His mom laughed and gave him a hug. The End!" |


## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 5e-6 |
| Batch Size | 4 (with 2-step gradient accumulation) |
| Early Stopping | Patience of 3 validation steps |
| Hardware | NVIDIA T4 GPUs (Google Colab) & RTX 3090|
| Model Variants | 32 context-token configs + 2 vanilla baselines |


## Results

### Quantitative Evaluation
Models were evaluated using GPT-4o-mini as an LLM judge, rating continuations on grammar, creativity, and consistency (1–10 scale) across 20 evaluation prompts with 10 completions each.

| Model | Grammar | Creativity | Consistency | Average ± SE |
|-------|---------|------------|-------------|--------------|
| **33M-prepend-sentences_2-clusters_5** | **7.68** | **6.62** | **8.05** | **7.45 ± 0.080** |
| 33M-vanilla-fine-tune | 7.54 | 6.47 | 7.79 | 7.27 ± 0.081 |
| 33M TinyStories (original) | 7.55 | 6.62 | 8.03 | 7.40 ± 0.071 |
| | | | | |
| 1M-postpend-sentences_1-clusters_32 | 4.74 | **5.64** | **4.13** | **4.84 ± 0.045** |
| 1M-vanilla-fine-tune | **4.88** | 5.55 | 4.04 | 4.82 ± 0.045 |
| 1M TinyStories (original) | 5.20 | 5.78 | 4.31 | 5.10 ± 0.042 |

### Statistical Significance
One-tailed difference-of-means t-tests at 5% significance level:

| Comparison | p-value | Significant? |
|------------|---------|--------------|
| 33M cluster vs. 33M vanilla | **0.029** | Yes |
| 33M cluster vs. 33M original | 0.691 | No |
| 1M cluster vs. 1M vanilla | 0.424 | No |
| 1M cluster vs. 1M original | 2.68 × 10⁻⁵ | Original wins |

### Qualitative Comparison
**Prompt:** *"Once upon a time there was a small brown mouse who lived in a big tree. The mouse wanted to be better friends with the birds that flew around the tree, but they were always too afraid to let him get close. One day, the mouse spoke to the birds bravely, 'Please don't be..."*

| Model | Completion Excerpt | Scores |
|-------|-------------------|--------|
| **33M + Context Tokens** | *"...scared. Can someone trust me?" `<cluster_2>` The mouse asked, "Can I trust you?" `<cluster_2>` The birds looked up, and sure enough, the mouse was indeed friendly. "Sure you can trust us," they said. `<cluster_1>` The mouse was so excited..."* | G:8 C:7 Co:9 |
| 33M Vanilla | *"...scared, my friends, I want to make sure you are okay and always look after me." The birds looked down at the mouse with fear and sadness. Finally the birds' fear filled them with courage..."* | G:8 C:7 Co:9 |


## Key Findings

### 1. Model Capacity Matters
The benefits of semantic context tokens are **contingent on model capacity**:
- **33M models:** Significant improvements over vanilla baselines
- **1M models:** No significant performance gains — limited capacity cannot leverage the additional structure

### 2. Prepending vs. Postpending
- **Prepending** works better for larger models (33M) — they can handle the harder "planning" task
- **Postpending** works better for smaller models (1M) — simpler "summarization" task within capacity limits

### 3. Structure Over Semantics
The best configuration (2-sentence, 5 clusters) suggests gains may stem from **reinforcing narrative structure** rather than capturing deeper semantic content. Context tokens may function as structural markers that help models organize narrative progression.

### 4. Data Efficiency
Fine-tuning on just **1% of the original TinyStories corpus** (5,000 stories) with context tokens outperformed vanilla fine-tuning, demonstrating the approach's data efficiency.


## Citation
If you use this code or methodology in your research, please cite:

```bibtex
@misc{semantic-context-tokens,
  author = {Group Sem-antics},
  title = {Semantic Context Tokens},
  year = {2025},
  url = {https://github.com/johnamit/semantic-context-tokens}
}
```


**Large Concept Models:**
```bibtex
@article{barrault2024large,
  title={Large concept models: Language modeling in a sentence representation space},
  author={Barrault, Lo{\"\i}c and Duquenne, Paul-Ambroise and Elbayad, Maha and Kozhevnikov, Artyom and Alastruey, Belen and Andrews, Pierre and Coria, Mariano and Couairon, Guillaume and Costa-juss{\`a}, Marta R and Dale, David and others},
  journal={arXiv preprint arXiv:2412.08821},
  year={2024}
}
```


**Tinystories:**
```bibtex
@article{eldan2023tinystories,
  title={Tinystories: How small can language models be and still speak coherent english?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

**AudioLM:**
```bibtex
@article{borsos2023audiolm,
  title={Audiolm: a language modeling approach to audio generation},
  author={Borsos, Zal{\'a}n and Marinier, Rapha{\"e}l and Vincent, Damien and Kharitonov, Eugene and Pietquin, Olivier and Sharifi, Matt and Roblek, Dominik and Teboul, Olivier and Grangier, David and Tagliasacchi, Marco and others},
  journal={IEEE/ACM transactions on audio, speech, and language processing},
  volume={31},
  pages={2523--2533},
  year={2023},
  publisher={IEEE}
}
```

**Segment any text:**
```bibtex
@article{frohmann2024segment,
  title={Segment any text: A universal approach for robust, efficient and adaptable sentence segmentation},
  author={Frohmann, Markus and Sterner, Igor and Vuli{\'c}, Ivan and Minixhofer, Benjamin and Schedl, Markus},
  journal={arXiv preprint arXiv:2406.16678},
  year={2024}
}
```


## License
This project is released under the MIT License.
