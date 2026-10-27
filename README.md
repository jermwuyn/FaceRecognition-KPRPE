# Face Recognition from Scratch with Vision Transformer and KP-RPE

> **Status (October 2026):** Model components, LFW preprocessing, and data loading are implemented. Existing tests pass (**33 tests, 12,041 subtests**). **End-to-end training, identity verification metrics, and a live demo have not yet been completed.**

An independent, research-oriented implementation of **Vision Transformer (ViT) + KeyPoint Relative Position Encoding (KP-RPE) + ArcFace** for face recognition, developed with PyTorch and designed for an 8 GB laptop GPU.

The project is **inspired by**, rather than a faithful reproduction of, [KeyPoint Relative Position Encoding for Face Recognition (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.html).

## 1. Research goals

Face recognition needs to distinguish identities despite changes in pose, expression, illumination, and image quality. This project investigates whether facial geometry can help a compact ViT learn more discriminative face representations.

Research questions (not yet answered by experiments):

- Does KP-RPE improve verification compared with the same ViT without keypoint-conditioned attention bias?
- How robust are learned embeddings to landmark noise and misalignment?
- What training configuration fits within 8 GB VRAM when training recognition weights from scratch?

## 2. Implemented model

```text
RGB face image [B, 3, 112, 112]     Facial landmarks [B, 5, 2]
                |                              | (normalized to [0, 1])
      Patch embedding (8x8)                     |
                |                              v
      196 tokens x 192 dims                  KP-RPE
                |                              |
      Learnable absolute positions       Attention bias [B, 1, 196, 196]
                |                              |
                +------------------------------+
                |
      6 x Transformer Encoder Blocks
      (3 attention heads, MLP ratio 4)
                |
      Token features [B, 196, 192]
                |
       Mean pooling + LayerNorm
                |
          Linear 192 -> 256
                |
          L2 normalization
                |
       Face embedding [B, 256]
                |
       +--------+---------+
       |                  |
  Training only      Verification / demo (planned)
  ArcFace head       Cosine similarity
  296 classes        No classification head required
       |
  Cross-entropy loss (planned training step)
```

**Implemented components** in `src/face_recognition/models/`:

| Module | Description |
| --- | --- |
| `vit.py` | Patch embedding, multi-head self-attention, transformer blocks, ViT backbone, face embedding head, and `FaceRecognitionModel` wrapper |
| `kprpe.py` | Five-keypoint relative geometry, learned bucket projections, and keypoint-conditioned attention bias |
| `arcface.py` | ArcFace classification head with angular margin (default `m=0.5`, `s=30`), for training identities |

Implementation choices: no CLS token, mean pooling for token aggregation, a shared KP-RPE bias across all six transformer blocks and three attention heads, and a compact 256-dimensional embedding. These are **project design choices**, not a claim of exact equivalence to the CVPR paper's reference model.

**From-scratch recognition:** The ViT backbone, KP-RPE projection, embedding head, and ArcFace weights are initialized without pretrained face-recognition weights. **YuNet is a separately pretrained detector** used only to extract the five facial landmarks; therefore the complete preprocessing system is not pretrained-free.

## 3. Data: LFW and YuNet

The current development dataset is **Labeled Faces in the Wild (LFW)**. The project uses the five landmarks extracted using a YuNet ONNX detector: left eye, right eye, nose, left mouth corner, and right mouth corner (consistent CSV column ordering).

The dataset was filtered to identities with at least five available images and split into **identity-disjoint** train/validation/test partitions. The following counts come from the project's data preparation and successful dataset loading checks:

| Split | Identities | Selected images | Images with accepted landmarks |
| --- | ---: | ---: | ---: |
| Train | 296 | 4,161 | 4,145 |
| Validation | 63 | 836 | 834 |
| Test | 64 | 988 | 984 |
| **Total** | **423** | **5,985** | **5,963** |

YuNet failed to produce accepted landmarks for 22 selected images. The original LFW collection contained 13,233 images spanning 5,749 identities.

Input preparation currently resizes each 250 x 250 RGB image to 112 x 112 without cropping and scales its pixel values to `[0, 1]`. Landmark coordinates are divided by 250, yielding normalized coordinates in `[0, 1]`; this is geometrically consistent with uniform whole-image resizing. The code does **not yet implement augmentation**.

`LFWDataset` returns image, landmarks, label, identity, and relative image path. Training labels are in `[0, 295]`; validation/test labels are set to `-1` because these identities are unseen during training. Evaluation will use identity strings and embedding similarities.

**Protocol limitation:** These custom identity-disjoint LFW splits are **not the same as the official LFW 6,000-pair verification protocol**. Future scores from these splits must be labeled as custom-split results, not reported as official LFW benchmark accuracy. The final test split should not be used for training or checkpoint selection.

### Data availability and licensing

LFW images and derived annotations are kept **outside the Git repository**. Obtain the dataset from an authorized source and follow its terms; do not redistribute images through this project. The YuNet ONNX model is included under `models/yunet/` alongside its MIT license notice (verify that the license covers the precise distributed model asset). See [LFW](https://vis-www.cs.umass.edu/lfw/) and the included `models/yunet/LICENSE`.

## 4. Repository structure

```text
FaceRecognition-KPRPE/
├── .gitignore
├── README.md
├── data/                           # Local-only datasets/annotations (gitignored)
│   ├── lfw/
│   │   ├── lfw_home/              # LFW image files
│   │   └── manifests/             # Identity-disjoint splits and label mapping
│   └── yunet/
│       └── annotations/
│           └── yunet_2026may/     # train/validation/test CSVs and checks
├── models/
│   └── yunet/
│       ├── LICENSE
│       └── face_detection_yunet_2026may.onnx
├── src/
│   └── face_recognition/
│       ├── data/
│       │   ├── lfw_dataset.py
│       │   └── dataloader.py
│       ├── models/
│       │   ├── vit.py
│       │   ├── kprpe.py
│       │   └── arcface.py
│       ├── training/               # In development
│       ├── evaluation/             # In development
│       └── inference/              # In development
├── scripts/
│   ├── prepare_lfw_manifest.py
│   ├── extract_lfw_landmarks.py
│   └── check_yunet_landmarks.py
├── tests/
│   ├── test_lfw_manifest.py
│   ├── test_lfw_landmarks.py
│   ├── test_yunet_landmarks.py
│   └── test_model_pipeline.py
└── outputs/                        # Local results (gitignored)
```

Empty package directories also contain `__init__.py` files. This tree describes the current layout; no `trainer.py`, `verification.py`, demo script, or training config has been implemented yet.

## 5. Environment and setup

The development environment reported for this project:

| Component | Configuration |
| --- | --- |
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM |
| Python | 3.11.17 (`D:\CondaEnvs\ai\python.exe`) |
| PyTorch | 2.11 (CUDA-enabled environment) |
| Data/preprocessing | OpenCV, NumPy |
| Development | VS Code, PowerShell, pytest |

A fully pinned dependency manifest is **not yet available**. The code requires PyTorch, NumPy, OpenCV and, to run the test suite, pytest. Use a compatible CUDA-enabled PyTorch distribution for your hardware. For an existing correctly configured environment, execute these commands at the repository root:

```powershell
# Run the existing tests
& "D:\CondaEnvs\ai\python.exe" -B -m pytest tests -q

# Check that the local training dataset and DataLoader resolve correctly
& "D:\CondaEnvs\ai\python.exe" -B -c "from src.face_recognition.data.dataloader import create_dataloader; batch = next(iter(create_dataloader('train', batch_size=16))); print(batch['image'].shape, batch['landmarks'].shape, batch['label'].shape)"
```

The latter command requires local LFW images and YuNet annotation CSVs at the paths in the repository tree. It is **not expected to work immediately after a fresh clone** without preparing the dataset. Consult the preparation scripts and their arguments before running them. No fully documented one-command data bootstrap is available yet.

## 6. Verification completed

After restructuring the project directories, the following local checks succeeded:

- Loaded train, validation and test datasets (4,145 / 834 / 984 accepted samples).
- Constructed a training batch with image shape `[16, 3, 112, 112]`, landmark shape `[16, 5, 2]`, and label shape `[16]`.
- Ran the existing suite: **33 passed, 12,041 subtests passed** (26 October 2026).

The passing tests cover existing checks only. **They do not establish successful end-to-end ViT + KP-RPE + ArcFace training, model convergence, or face-recognition accuracy.** Some prior component tests may need expanding to exercise the now-integrated full model.

## 7. Planned training and evaluation

The planned training loop will take image batches and five normalized landmarks, compute face embeddings through ViT + KP-RPE, pass train labels to the ArcFace head, and optimize cross-entropy loss. Proposed initial batch size is 16, subject to available 8 GB GPU memory. Optimizer, learning-rate schedule, mixed precision, checkpoint selection and augmentation have **not yet been finalized or benchmarked**.

Evaluation will compare L2-normalized face embeddings with cosine similarity. Planned reporting includes ROC, EER, and TAR@FAR where statistically supported; thresholds must be selected using development data rather than the final test split. A controlled ViT-only versus ViT + KP-RPE comparison and a webcam/enrollment demo are also planned.

**No training accuracy, verification accuracy, EER, benchmark scores, or demo results are available yet.**

## 8. Roadmap

- [x] Define project goals, initialization policy, and architecture.
- [x] Build identity-disjoint LFW manifests and check identity overlap.
- [x] Extract five YuNet landmarks and audit accepted images.
- [x] Implement ViT, KP-RPE, face embedding head, and ArcFace classification head.
- [x] Implement LFW Dataset and DataLoader.
- [x] Reorganize model/data/training/evaluation/inference packages.
- [x] Pass existing tests after restructuring (33 tests; 12,041 subtests).
- [ ] Validate integrated forward/backward path through ViT + KP-RPE + ArcFace.
- [ ] Implement a training loop, optimizer, checkpoints, and training metrics.
- [ ] Train the recognition network from random initialization.
- [ ] Implement identity-disjoint verification evaluation.
- [ ] Compare a ViT-only baseline with the KP-RPE variant.
- [ ] Develop image/webcam face recognition demo.
- [ ] Report reproducible configurations, results, limitations, and ethical considerations.

## 9. References

1. Minchul Kim, Yiyang Su, Feng Liu, Anil Jain, and Xiaoming Liu. **[KeyPoint Relative Position Encoding for Face Recognition](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.html)**. CVPR 2024.
2. Alexey Dosovitskiy et al. **[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)**. ICLR 2021.
3. Jiankang Deng et al. **[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)**. CVPR 2019.
4. Minchul Kim, Anil K. Jain, and Xiaoming Liu. **[AdaFace: Quality Adaptive Margin for Face Recognition](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html)**. CVPR 2022 (related work; not currently implemented).
5. **[Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/)**, University of Massachusetts Amherst.
