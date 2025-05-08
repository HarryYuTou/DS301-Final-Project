# DS301-Final-Project: Hate Speech Detection Fusing Texts and Images

This project explores various deep learning approaches for classifying sentiment in meme-based social media posts from the MMHS150K dataset. It compares single-modal models (image-only and text-only), model with captioning + text extraction from images combined with original text, and multimodal model (image+text)

---

## Models Used

### 1. **Text-Only Model**
- **Architecture**: LSTM
- **Input**: tweet text
- **Purpose**: Establish a baseline for text-based sentiment classification.

### 2. **Image-Only Model**
- **Architecture**: ResNet-18 
- **Input**: Meme image
- **Purpose**: Classify sentiment using only image content.

### 3. **Captioning + Text extraction Model**
- **Captioning**: ViT-GPT2
- **Text Extraction**: EasyOCR
- **Fusion**: Concatenated with original tweet and fed into LSTM
- **Purpose**: Capture embedded textual meaning within memes and combine with original text to do the classification.

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
  - 1-layer LSTM 
  - Followed by fully connected output layer
- **Loss Function**: binary_crossentropy 
- **Optimizer**: Adam 
- **Batch Size**: 32
- **Epochs**: 20
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
  - All convolutional layers frozen; only last FC layers trained
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: SGD
- **Batch Size**: 32
- **Epochs**: 10
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
- **Training Strategy**: Identical to pure text pipeline

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

