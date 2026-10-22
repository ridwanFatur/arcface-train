# Face Recognition Embedding Training with ViT-S

Train a Vision Transformer Small (ViT-S) backbone to learn 512-dimensional face embeddings using the ArcFace training pipeline from InsightFace. Evaluate the trained model on standard face verification benchmarks to measure its ability to distinguish between genuine and impostor face pairs.

## Overview

This project explores training a face recognition backbone from scratch using the `vit_s` architecture provided by InsightFace's `arcface_torch` toolkit.

The model is trained using the training split of the face recognition dataset. Its learned embeddings are then evaluated on six standard face verification benchmarks.

**Objectives:**
- Train a ViT-S backbone for face recognition.
- Generate 512-dimensional face embeddings.
- Evaluate verification accuracy across different benchmarks.
- Analyze the model's generalization to variations in age, pose, and facial appearance.

## Model Architecture

The project uses the `vit_s` backbone from `arcface_torch`.

```python
elif name == "vit_s":
    num_features = kwargs.get("num_features", 512)
    from .vit import VisionTransformer

    return VisionTransformer(
        img_size=112,
        patch_size=9,
        num_classes=num_features,
        embed_dim=512,
        depth=12,
        num_heads=8,
        drop_path_rate=0.1,
        norm_layer="ln",
        mask_ratio=0.1,
    )
```

| Parameter | Value |
|---|---|
| Backbone | ViT-S |
| Input image size | 112 × 112 |
| Patch size | 9 × 9 |
| Embedding dimension | 512 |
| Transformer depth | 12 |
| Attention heads | 8 |
| Drop path rate | 0.1 |
| Normalization | LayerNorm |
| Mask ratio | 0.1 |

The backbone produces a 512-dimensional feature representation for each input face image. These embeddings are used for face verification by comparing the similarity between two faces.

## Dataset

The dataset is available from Google Drive:

[Download Face Recognition Dataset](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view)

The dataset directory contains the following files:

```text
dataset/
├── agedb_30.bin
├── calfw.bin
├── cfp_ff.bin
├── cfp_fp.bin
├── cplfw.bin
├── lfw.bin
├── property
├── train.csv
├── train.lst
├── train.idx
├── train.rec
└── train/
```

### Training Data

Only the training dataset is used to train the backbone:

- `train.rec`
- `train.idx`
- `train.lst`
- `train.csv` (if required by the training pipeline)
- `train/` (if used by the dataset loader)

The actual training data loader should be configured to use the training split only.

### Evaluation Benchmarks

The following `.bin` files are reserved for evaluation and are not used for training:

| Dataset | Primary evaluation focus |
|---|---|
| `agedb_30.bin` | Age variation |
| `calfw.bin` | Cross-age face verification |
| `cfp_ff.bin` | Frontal-to-frontal verification |
| `cfp_fp.bin` | Frontal-to-profile verification |
| `cplfw.bin` | Cross-pose face verification |
| `lfw.bin` | General face verification |

Keeping the evaluation benchmarks separate from the training data helps measure the model's ability to generalize to unseen verification pairs.

## Training Toolkit

The project uses the `arcface_torch` implementation from InsightFace.

Clone the repository using a shallow, blob-filtered clone with sparse checkout:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/deepinsight/insightface.git

cd insightface

git sparse-checkout set recognition/arcface_torch
```

Refer to the `recognition/arcface_torch` directory for the training implementation, model definitions, dataset loaders, and verification utilities.

Configure the training pipeline to use the `vit_s` backbone and the training dataset described above.

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy:** Mean face verification accuracy across evaluation folds, with the reported standard deviation.
- **VAL @ FAR = 1e-3:** Verification rate at a target false acceptance rate of \(10^{-3}\), with the reported standard deviation.
- **FAR:** Observed false acceptance rate associated with the reported verification threshold.

Accuracy measures the model's ability to classify matching and non-matching face pairs. VAL at a low FAR is particularly relevant when false identity acceptance must be minimized.

## Evaluation Results

The following results were obtained from the trained ViT-S backbone.

| Benchmark | Accuracy | Accuracy Std. | VAL @ FAR = 1e-3 | VAL Std. | FAR |
|---|---:|---:|---:|---:|---:|
| AgeDB-30 | 0.7577 | 0.0172 | 0.0320 | 0.0150 | 0.001333 |
| CALFW | 0.8288 | 0.0105 | 0.1990 | 0.0247 | 0.001000 |
| CFP-FP | 0.7356 | 0.0177 | 0.0383 | 0.0130 | 0.001143 |
| CFP-FF | 0.9416 | 0.0092 | 0.5749 | 0.0812 | 0.001143 |
| CPLFW | 0.6517 | 0.0223 | 0.0047 | 0.0048 | 0.001333 |
| LFW | 0.9472 | 0.0129 | 0.7543 | 0.0441 | 0.001333 |

### Results Summary

The model performs relatively well on LFW and CFP-FF, where the reported verification accuracies are 94.72% and 94.16%, respectively.

However, performance decreases substantially on benchmarks involving more challenging variations:

- **AgeDB-30:** 75.77% accuracy.
- **CFP-FP:** 73.56% accuracy, indicating difficulty with frontal-to-profile face verification.
- **CPLFW:** 65.17% accuracy, showing substantial difficulty with cross-pose verification.

The low VAL values at FAR = 1e-3 on AgeDB-30, CFP-FP, and CPLFW also indicate that the model struggles to maintain a high verification rate under a strict false acceptance constraint.

Overall, the current results suggest that the model has not yet learned sufficiently robust face representations across age and pose variations.

## Limitations and Future Improvements

Potential directions for further experimentation include:

1. **Training duration:** Evaluate whether additional training improves convergence and verification performance.
2. **Training configuration:** Investigate the learning rate, batch size, optimizer, and learning-rate schedule.
3. **ArcFace loss:** Review the ArcFace margin, scale, and classification head configuration.
4. **Data preprocessing:** Verify face alignment, normalization, image quality, and input color conventions.
5. **Embedding evaluation:** Check feature normalization, cosine similarity, and verification threshold selection.
6. **Benchmark analysis:** Inspect false acceptance and false rejection cases to identify weaknesses in age and pose generalization.
7. **Pretrained initialization:** Compare training from scratch against pretrained backbone initialization.

These are potential investigation areas rather than confirmed causes of the current results.

## Conclusion

This project establishes a baseline for training a ViT-S face recognition backbone with 512-dimensional embeddings using InsightFace's `arcface_torch` toolkit.

The model achieves strong accuracy on LFW and CFP-FF but shows limited generalization on age- and pose-related benchmarks, particularly CPLFW and CFP-FP. Further experiments are needed to improve embedding robustness and verification performance at low false acceptance rates.

The current results serve as a baseline for subsequent architecture, training, and evaluation improvements.
