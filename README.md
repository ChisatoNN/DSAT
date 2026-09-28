# Cross-Model Dense Semantic Alignment for Open-Vocabulary CAM Generation

**Official implementation (in preparation)** of the paper **"Cross-Model Dense Semantic Alignment for Open-Vocabulary CAM Generation"**.

> **Release status:** The code and documentation are being organized. Installation instructions, pretrained checkpoints, and runnable training/evaluation commands will be added as they become available. The results below are reported in the manuscript; the repository is not yet a complete reproduction package.

## Overview

This work introduces a cross-model dense semantic alignment framework for open-vocabulary class activation map (CAM) generation. A **Dense Semantic Alignment Transformer (DSAT)** learns to map **SAM pre-neck features** to a **CLIP-compatible dense semantic space**, using **CLIP Surgery dense features** as training targets.

During training, both pretrained image encoders are frozen, and only DSAT is optimized. During inference, the CLIP Surgery **image encoder** is removed. Query-conditioned CAMs are generated using the SAM image encoder, DSAT, and the frozen CLIP text encoder. The same SAM image encoding pass also provides the image embeddings for optional CAM-guided SAM mask decoding.

**Key features**

- **Dense semantic transfer:** align SAM pre-neck representations with CLIP Surgery dense teacher features.
- **Context-aware mapping:** use DSAT to model interactions among spatial tokens.
- **Shared visual encoding:** generate CAMs and decode SAM masks without a separate CLIP Surgery image encoding pass at inference.
- **Frozen foundation models:** optimize DSAT while keeping the pretrained SAM and CLIP Surgery image encoders frozen.

<!-- TODO: Add the framework illustration when the figure is ready.
![Framework](assets/framework.png)
-->

## Results

The following numbers are reported in the manuscript under a unified **query-conditioned** evaluation protocol. The image-level category list is supplied as the query set for each evaluation image. The reported mIoU values include the background category.

### Direct CAM localization

| Method | VOC2012 | VOC Context | COCO 2017 | Mean |
|:--|--:|--:|--:|--:|
| CLIP Surgery | 56.7 | 42.9 | 33.6 | 44.4 |
| **Ours** | **61.0** | **47.3** | **35.8** | **48.0** |

### CAM-guided SAM segmentation

| Method | VOC2012 | VOC Context | COCO 2017 | Mean |
|:--|--:|--:|--:|--:|
| CLIP Surgery + SAM | 63.3 | 47.9 | 37.6 | 49.6 |
| **Ours + SAM** | **65.4** | **49.8** | **38.1** | **51.1** |

### End-to-end inference efficiency

Measured on the **VOC2012 validation set (1,449 images)** with an **NVIDIA GeForce RTX 5070 Ti**, batch size 1. The two complete CAM-guided SAM pipelines were evaluated independently over three runs after warm-up. Text embeddings were cached; disk I/O and prediction saving were excluded.

| Method | Image encoders | Latency (ms/image) | Throughput (images/s) | Peak GPU memory (GiB) |
|:--|--:|--:|--:|--:|
| CLIP Surgery + SAM | 2 | 164.96 | 6.06 | 3.05 |
| **Ours + SAM** | **1** | **139.11** | **7.19** | **2.80** |

The shared-encoding pipeline reduces average end-to-end inference latency by **15.7%** relative to the evaluated CLIP Surgery + SAM pipeline.

## Method at a glance

| Component | Configuration / role |
|:--|:--|
| Dense semantic teacher | CLIP Surgery, CLIP ViT-B/16 image encoder (frozen during training) |
| Structural feature source | SAM ViT-B pre-neck features (frozen image encoder) |
| Alignment network | DSAT with four ViT encoder blocks |
| DSAT input | SAM pre-neck features, `64 × 64 × 768` |
| DSAT output | Aligned features, `32 × 32 × 512` (default) |
| Default training objective | Dense feature regression using mean squared error (MSE) |
| CAM generation | Position-wise similarity to frozen CLIP text embeddings |
| Optional downstream segmentation | CAM-derived positive points and boxes with the pretrained SAM prompt and mask decoders |

The standard alignment setting uses unlabeled training images from VOC2012, VOC Context, and COCO 2017. An additional experiment trains DSAT on an external ImageNet subset. Neither category labels nor pixel-level annotations are used as alignment supervision.

<!--
## Installation

**Coming soon.** The tested Python, PyTorch, CUDA, and package versions will be documented with the initial code release.

 TODO: After releasing the actual files, replace this section with tested commands.
Example structure (not executable until the corresponding files exist):

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
# Install the exact dependencies specified by the released project.
```

## Repository structure

**To be updated after code cleanup.** Document the *actual* layout here rather than treating the example below as an existing directory tree.

 TODO: Replace with the real directory tree, e.g., model code, dataset loaders,
training, inference, evaluation, configuration files, and assets.

## Pretrained models and datasets

The implementation depends on pretrained **CLIP Surgery** and **SAM ViT-B** checkpoints. Checkpoint sources, exact filenames, expected directory locations, and applicable third-party licenses will be provided with the code release.

Evaluated datasets:

- **PASCAL VOC2012** — 20 foreground categories.
- **PASCAL VOC Context** — 59 foreground categories in the reported protocol.
- **COCO 2017** — 80 foreground categories.
- **ImageNet subset** — used only for the external-data alignment-training experiment.

**TODO:** Add official dataset links, the exact dataset splits, directory structures, and preprocessing instructions.

## Training

**Coming soon.** The paper's default DSAT training configuration uses four ViT encoder blocks, MSE feature alignment, Adam, 30 epochs, batch size 10, and a cosine-annealed learning rate from `2e-4` to `1e-4`. Both pretrained image encoders remain frozen.

**TODO:** Publish the actual training entry point, configuration files, random-seed policy, data preparation steps, and a tested training command.

## Inference and evaluation

**Coming soon.** The release will document how to:

1. Generate query-conditioned CAMs with the SAM image encoder, DSAT, and frozen CLIP text encoder.
2. Optionally convert CAMs into point/box prompts for SAM mask decoding while reusing the SAM image embeddings.
3. Evaluate direct CAM and downstream segmentation mIoU using the manuscript's protocol.

**Evaluation note:** The manuscript uses the ground-truth **image-level category list as the evaluation query set**, identically for compared methods. This protocol is not a fully automatic discovery of all category names in an image. For direct CAM evaluation, per-category maps are normalized to `[0, 1]` and a foreground/background threshold of `0.60` is used.

**TODO:** Add exact commands, expected output formats, checkpoint loading instructions, and examples using arbitrary user-supplied category queries.

## Reproducibility checklist

- [ ] Publish source code and tested environment specifications.
- [ ] Publish dataset preparation and split instructions.
- [ ] Provide checkpoint download instructions (subject to upstream licenses).
- [ ] Publish DSAT training configurations and commands.
- [ ] Publish CAM inference and downstream SAM decoding commands.
- [ ] Publish evaluation scripts and the exact benchmark protocols.
- [ ] Verify the released results against the manuscript tables.


## Citation

If you use this work, please cite the paper. A BibTeX entry and publication link will be added when the bibliographic information is available.

**Paper:** *Cross-Model Dense Semantic Alignment for Open-Vocabulary CAM Generation*  
**Authors:** Wenjie Zhu, Jieru Wei, and Jiandong Shang.

## Acknowledgments

This work builds on **CLIP Surgery** and the **Segment Anything Model (SAM)**. Please also follow their original repositories' citation and license requirements when using their implementations or pretrained checkpoints.

## License

**To be announced.** A repository license will be selected and added before the public code release. Third-party models, checkpoints, and datasets remain subject to their respective licenses and terms.
-->
