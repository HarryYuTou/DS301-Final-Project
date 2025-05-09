# DS301-Final-Project: Hate Speech Detection Fusing Texts and Images

This project explores various deep learning approaches for classifying sentiment in meme-based social media posts from the MMHS150K dataset. It compares single-modal models (image-only and text-only), model with captioning + text extraction from images combined with original text, and multimodal model (image+text)

---

## Models Used

### 1. **Text-Only Model**
- **Model**: LSTM
- **Input**: tweet text
- **Purpose**: Establish a baseline for text-based sentiment classification.

### 2. **Image-Only Model**
- **Model**: ResNet-18 
- **Input**: Meme image
- **Purpose**: Classify sentiment using only image content.

### 3. **Captioning + Text extraction Model**
- **Captioning**: ViT-GPT2
- **Text Extraction**: EasyOCR
- **Fusion**: Concatenated with original tweet and fed into LSTM
- **Purpose**: Capture embedded textual meaning within memes or general information from images and combine with original text to do the classification.

### 4. **Multimodal Model**
- **Image Branch**: ResNet-18 
- **Text Branch**: BERT 
- **Fusion**: Feature concatenation before classification
- **Purpose**: Leverage both visual and textual cues for prediction.

---

## Experiments

- Dataset: **MMHS150K**
- Tasks:
  - Multi-label classification (e.g., hate speech, offensive language)
- Evaluation Metrics:
  - Jaccard Accuracy
  - F1 Score

---

## Implementation Details

Below are the technical details and training procedures for each of the four models evaluated in this project:

---

### 1. Pure Text Model (LSTM)

- **Input**: Original tweet text only
- **Text Preprocessing**:
  - Lowercasing
  - Tokenization
- **Architecture**:
  - Input dim=20000, output dim=128, input length=100
  - LSTM layer with 256 units, 0.2 dropout
  - Dense layer with 32 units and tanh
  - Dense layer with 6 units and sigmoid
- **Loss Function**: binary_crossentropy 
- **Optimizer**: SGD
- **Batch Size**: 64
- **Epochs**: 20
- **lr**:0.01
- **Momentum**:0.1
- **Train/Val/Test Split**: 80/10/10
- **Output**: Multi-label classification for sentiment categories

---

### 2. Pure Image Model (ResNet-18)

- **Input**: Meme image only
- **Image Preprocessing**:
  - Resized to 224×224
  - Normalized using ImageNet statistics
- **Architecture**:
  - Pretrained ResNet-18 
  - Final fc layer replaced with a linear layer
  - Outputting size of train set - 1
  - Multi-label classification with one output per label
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: SGD
- **Epochs**: 10
- **lr**:0.0001
- **Momentum**:0.9
- **Weight Decay**: 0.00001
- **Output**: Multi-label classification for sentiment categories

---

### 3. Caption + OCR + Tweet Fusion Model (Text-Only)

- **Input**: 
  - Original tweet
  - Text extracted via EasyOCR from meme 
  - Caption generated using vit-gpt2
- **Text Fusion**:
  - Concatenate caption, extracted text and text tweet
- **Tokenizer**: Same tokenizer used across all fused text
- **Model**: LSTM (same architecture as the Pure Text model)
- **Architecture**:
  - Input dim=20000, output dim=192, input length=100
  - LSTM layer with 64 units, 0.0 dropout
  - Dense layer with 64 units and relu
  - Dense layer with 6 units and sigmoid
- **Loss Function**: binary_crossentropy 
- **Optimizer**: rmsprop
- **Batch Size**: 64
- **Epochs**: 20
- **lr**:0.0042
- **Momentum**:0.6
- **Train/Val/Test Split**: 80/10/10
- **Output**: Multi-label classification for sentiment categories
---

### 4. Multimodal Model (Image + Text Fusion)

- **Input**:
  - Image: Meme image
  - Text: Original tweet only
- **Image Branch**:
  - ResNet-18 (pretrained, final layer modified to output 512-dim features)
- **Text Branch**:
  - BERT (bert-base-uncased) for tokenization and embedding
  - CLS token representation (768-dim)
- **Fusion Strategy**:
  - Concatenate [Image_512 || Text_768] → 1280-dim vector
- **Training**:
  - BERT: frozen
  - ResNet-18: frozen
  - Only fusion and final layers trained
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 10


---

## Key Observations

- The **caption+text extraction fusion model** outperformed all others in terms of Jaccard accuracy.
- The **multimodal model** underperformed likely due to:
  - Feature imbalance between image and text branches.
  - Potential contradictory or weakly correlated information conveyed by texts and images, which may introduce noise and confuse the classifier.

