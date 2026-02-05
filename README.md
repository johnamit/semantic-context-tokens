<img src="assets/icons/sit_faf_edit2.png" alt="ContraCTGAN" width="900"><br>
[<img src="https://img.shields.io/badge/Read%20Paper-PDF-black?style=for-the-badge&labelColor=0057FF&logo=adobeacrobat&logoColor=white" alt="Read Paper"/>](https://drive.google.com/file/d/1f_bSPjfhiGOoaf3Gu1gnbcc2W1wNWSec/view?usp=sharing)


A hybrid tokenization framework that combines **coarse semantic context tokens** with **fine-grained sub-word tokens** to improve narrative cohesion and creativity in Large Language Models (LLMs). This research builds upon Meta's [Large Concept Models (LCMs)](https://github.com/facebookresearch/large_concept_model) by exploring a quantized, hierarchical generation paradigm ("big picture to small") mimicking human cognitive processes. This was a collaborative group research project conducted as part of my MSc programme at University College London.


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


## References
- Barrault et al. (2024). [Large Concept Models: Language Modeling in a Sentence Representation Space](https://arxiv.org/abs/2412.08821)
- Eldan & Li (2023). [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- Borsos et al. (2023). [AudioLM: A Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143)
- Frohmann et al. (2024). [Segment Any Text: A Universal Approach for Robust Sentence Segmentation](https://arxiv.org/abs/2406.16678)



## License
This project is released under the MIT License.
