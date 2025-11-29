# Video Captioning - PyTorch Implementation

Video Captioning is a sequential learning model that employs an encoder-decoder architecture. It accepts a video as input and produces a descriptive caption that summarizes the content of the video.

The significance of captioning stems from its capacity to enhance accessibility to videos in various ways. An automated video caption generator aids in improving the searchability of videos on websites. Additionally, it facilitates the grouping of videos based on their content by making the process more straightforward.


<h2 id="Dataset">Dataset</h2>

This project utilizes the <a href="https://opendatalab.com/MSVD">MSVD</a> dataset, which consists of 1450 training videos and 100 testing videos, to facilitate the development of video captioning models.


## Features

### Model Architecture
- **Encoder-Decoder Architecture**: LSTM-based encoder for video features, LSTM decoder with attention for caption generation
- **Attention Mechanisms**: Multiple attention types (Bahdanau, Luong, Multi-head)
- **Feature Extraction**: Support for VGG16, ResNet50, and custom CNN backbones
- **Flexible Generation**: Greedy search and beam search decoding

### PyTorch Implementation
- **Clean Architecture**: Modular design with separate components for data, models, training, and inference
- **Configuration Management**: Centralized configuration system with dataclasses
- **Advanced Training**: Modern training loop with validation, checkpointing, and early stopping
- **Experiment Tracking**: Integration with Weights & Biases and TensorBoard
- **Comprehensive Metrics**: BLEU, METEOR, ROUGE, CIDEr, and diversity metrics

### Production Ready
- **CLI Interface**: Command-line tools for training, inference, and preprocessing
- **Batch Processing**: Efficient batch inference for multiple videos
- **Checkpoint Management**: Automatic model saving and loading
- **Error Handling**: Robust error handling and logging
- **Documentation**: Comprehensive documentation and examples

## Project Structure

```
Video-Captioning/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py              # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # PyTorch Dataset classes
│   │   └── vocabulary.py          # Vocabulary management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py             # Video encoder models
│   │   ├── decoder.py             # Caption decoder models
│   │   ├── attention.py           # Attention mechanisms
│   │   └── video_captioning_model.py  # Main model
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # Training loop
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py           # Inference pipeline
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py             # Evaluation metrics
│       └── checkpoint.py          # Checkpoint management
├── train.py                       # Training script
├── predict.py                     # Inference script
├── preprocess.py                  # Preprocessing script
├── requirements_pytorch.txt       # Dependencies
└── README_PYTORCH.md             # This file
```

## 🛠️ Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg (for video processing)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/angadbawa/Video-Captioning
cd Video-Captioning

# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements_pytorch.txt

```

## Quick Start

### 1. Data Preparation

First, prepare your video data and extract features:

```bash
# Extract features from videos
python preprocess.py \
    --video-dir /path/to/videos \
    --output-dir /path/to/features \
    --model-type vgg16 \
    --create-dataset \
    --captions-file /path/to/captions.csv \
    --dataset-output dataset.csv
```

### 2. Training

Train the video captioning model:

```bash
# Basic training
python train.py \
    --data-file dataset.csv \
    --checkpoint-dir checkpoints \
    --batch-size 32 \
    --epochs 100 \
    --wandb

# Resume from checkpoint
python train.py \
    --data-file dataset.csv \
    --checkpoint-dir checkpoints \
    --resume checkpoints/checkpoint_epoch_0050.pth
```

### 3. Inference

Generate captions for videos:

```bash
# Single video prediction
python predict.py single \
    --model-path checkpoints/model_for_inference.pth \
    --video-path /path/to/video.mp4 \
    --method beam \
    --beam-size 5

# Batch prediction
python predict.py batch \
    --model-path checkpoints/model_for_inference.pth \
    --video-list /path/to/video_directory \
    --output results.json \
    --captions-file captions.txt

# Multiple diverse captions
python predict.py multiple \
    --model-path checkpoints/model_for_inference.pth \
    --video-path /path/to/video.mp4 \
    --num-captions 5 \
    --method beam
```

## Model Architecture

### Encoder
- **Video Feature Extraction**: Pre-trained CNN (VGG16/ResNet50) for frame-level features
- **Temporal Encoding**: Bidirectional LSTM to capture temporal dependencies
- **Feature Projection**: Linear layers to match encoder-decoder dimensions

### Decoder
- **Word Embedding**: Learnable word embeddings for vocabulary
- **LSTM Decoder**: Multi-layer LSTM for sequential caption generation
- **Attention Mechanism**: Configurable attention (Bahdanau/Luong/Multi-head)
- **Output Projection**: Linear layer to vocabulary size with softmax

### Attention Mechanisms

#### Bahdanau Attention (Additive)
```python
attention_scores = linear(tanh(W_encoder * encoder_outputs + W_decoder * decoder_hidden))
```

#### Luong Attention (Multiplicative)
```python
attention_scores = decoder_hidden^T * W * encoder_outputs  # General
attention_scores = decoder_hidden^T * encoder_outputs      # Dot
```

#### Multi-Head Attention
```python
attention_output = MultiHead(Q=decoder_hidden, K=encoder_outputs, V=encoder_outputs)
```
