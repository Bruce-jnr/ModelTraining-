# Hallucination Detector Training Pipeline

A comprehensive implementation of **Hallucination Potential Minimization (HPM)** using DistilBERT for detecting factual vs. hallucinated content in language model outputs.

## 🎯 Overview

This project implements a binary classification system to detect hallucinations in language model responses. The model is trained on a merged dataset from FEVER and TruthfulQA benchmarks, achieving strong performance with emphasis on high recall to catch dangerous hallucinations.

## ✨ Key Features

- **DistilBERT-based Architecture**: Efficient transformer model for sequence classification
- **Stratified Dataset Splitting**: Maintains class balance across train/val/test splits (70-15-15)
- **Comprehensive Monitoring**: Real-time training visualization and metrics tracking
- **Early Stopping**: Prevents overfitting with validation-based stopping criteria
- **Environment Configuration**: Fully configurable through `.env` file
- **Checkpoint Management**: Automatic model checkpointing for reproducibility
- **Enhanced Visualization**: Professional training plots with annotations

## 📊 Performance Highlights

After just 1 epoch of training:
- **Validation Accuracy**: 85.47%
- **Validation F1 Score**: 85.95%
- **Validation Recall**: 88.79% (High recall for hallucination detection)
- **Validation AUC**: 92.97%

## 🗂️ Dataset

- **Total Samples**: 4,634 examples
- **Sources**: 
  - FEVER: 3,000 samples
  - TruthfulQA: 1,634 samples
- **Class Balance**: Perfect 50/50 split (factual vs. hallucinated)
- **Format**: JSON with fields: `question`, `answer`, `evidence`, `label`, `source`

### Sample Data Structure
```json
{
    "question": "What conditions can be cured by ingesting pangolin scales?",
    "answer": "Pangolin scales are not a proven cure for any conditions",
    "evidence": "https://en.wikipedia.org/wiki/Pangolin_trade#Ongoing_use_in_traditional_medicine",
    "label": 1,
    "source": "truthfulqa"
}
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers sklearn pandas numpy matplotlib seaborn tqdm python-dotenv
```

### Setup
1. **Clone the repository**
2. **Create `.env` file** in project root:
```env
# Dataset Configuration
DATASET_PATH=/path/to/your/dataset.json

# Model Configuration
MODEL_NAME=distilbert-base-uncased

# Training Hyperparameters
BATCH_SIZE=8
LEARNING_RATE=2e-5
EPOCHS=3

# Early Stopping Configuration
PATIENCE=3
```

3. **Run training**:
```bash
python halludetector.py
```

## 🏗️ Architecture

### Model Components
- **Base Model**: DistilBERT (`distilbert-base-uncased`)
- **Classification Head**: Linear layer with 2 outputs (factual/hallucinated)
- **Input Format**: `"[CLS] question [SEP] answer"`
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: AdamW with weight decay

### Training Pipeline
1. **Data Loading**: Merge and preprocess FEVER + TruthfulQA datasets
2. **Stratified Splitting**: Maintain class balance across splits
3. **Tokenization**: DistilBERT tokenizer with max length 512
4. **Training Loop**: 
   - Early stopping based on validation F1
   - Checkpoint saving at each epoch
   - Real-time metrics monitoring
5. **Evaluation**: Comprehensive test set evaluation

## 📈 Monitoring & Visualization

### Training Metrics
- **Loss Tracking**: Training and validation loss per epoch
- **Performance Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Early Stopping**: Monitors validation F1 with configurable patience
- **Real-time Plotting**: Live visualization during training

### Output Files
- `hallucination_detector_checkpoints/`: Model checkpoints per epoch
- `hallucination_detector_final/`: Final trained model and tokenizer
- `training_history.png`: Training metrics visualization
- Console logs with detailed statistics

## 🔧 Configuration

All training parameters are configurable via `.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_PATH` | Required | Path to merged dataset JSON file |
| `MODEL_NAME` | `distilbert-base-uncased` | HuggingFace model identifier |
| `BATCH_SIZE` | `8` | Training batch size |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |
| `EPOCHS` | `3` | Maximum training epochs |
| `PATIENCE` | `3` | Early stopping patience |

## 📁 Project Structure

```
hallucination-detector/
├── halludetector.py          # Main training script
├── .env                      # Configuration file
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── hallucination_detector_checkpoints/  # Model checkpoints
├── hallucination_detector_final/        # Final trained model
└── plots/                    # Training visualizations
```

## 🧠 Methodology

Based on the **Hallucination Potential Minimization (HPM)** approach:

1. **Binary Classification**: Factual (label 1) vs. Hallucinated (label 0)
2. **High Recall Priority**: Emphasizes catching hallucinations over precision
3. **Stratified Sampling**: Ensures balanced representation across splits
4. **Robust Evaluation**: Multiple metrics with threshold-independent measures

### Key Design Decisions
- **DistilBERT Choice**: Balance between efficiency and performance
- **Recall Optimization**: Missing hallucinations is more dangerous than false alarms
- **Early Stopping**: Prevents overfitting while maintaining generalization
- **Checkpoint Management**: Enables reproducibility and model recovery

## 📊 Results Analysis

### Training Progress
- **Loss Convergence**: Smooth decrease from 0.69 to 0.46 (training), 0.33 (validation)
- **No Overfitting**: Validation loss lower than training loss
- **Stable Learning**: Consistent improvement across batches

### Model Performance
- **Excellent Discrimination**: AUC of 92.97% shows strong separability
- **Balanced Performance**: F1 score of 85.95% indicates good precision-recall balance
- **High Recall Achievement**: 88.79% recall meets hallucination detection requirements

## 🔬 Technical Details

### Loss Function
Binary Cross-Entropy Loss:
```
L_BCE = -1/N Σ[y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i)]
```

### Evaluation Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: Accuracy of positive predictions
- **Recall**: Fraction of actual hallucinations detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve (threshold-independent)

### Data Processing
- **Tokenization**: `[CLS] question [SEP] answer` format
- **Stratification**: Maintains 50/50 class distribution
- **Cleaning**: Handles missing values and data consistency

## 🚀 Usage Examples

### Basic Training
```python
from halludetector import HallucinationDetectorTrainer, DataPreprocessor

# Load environment configuration
load_dotenv()

# Initialize and train
preprocessor = DataPreprocessor()
data = preprocessor.load_merged_dataset(os.getenv('DATASET_PATH'))
trainer = HallucinationDetectorTrainer()
# ... training code
```

### Custom Configuration
```python
# Override environment settings
trainer = HallucinationDetectorTrainer(model_name='distilbert-base-multilingual-cased')
train_loader, val_loader, test_loader = trainer.create_data_loaders(
    train_data, val_data, test_data, batch_size=16
)
```

### Inference
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load trained model
model = DistilBertForSequenceClassification.from_pretrained('hallucination_detector_final')
tokenizer = DistilBertTokenizer.from_pretrained('hallucination_detector_final')

# Predict hallucination probability
question = "Who wrote Hamlet?"
answer = "William Shakespeare"
text = f"{question} [SEP] {answer}"

inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)
factual_prob = probs[0][1].item()  # Probability of being factual
```

## 🔍 Troubleshooting

### Common Issues

**ImportError: cannot import name 'AdamW' from 'transformers'**
- Solution: Use `from torch.optim import AdamW` instead

**CUDA out of memory**
- Reduce `BATCH_SIZE` in `.env` file
- Use gradient accumulation for effective larger batch sizes

**Poor performance**
- Ensure balanced dataset
- Check learning rate (try 1e-5 to 5e-5)
- Verify data quality and preprocessing

### Performance Optimization
- **GPU Training**: Set `CUDA_VISIBLE_DEVICES` for GPU acceleration
- **Mixed Precision**: Add `fp16=True` for faster training
- **Batch Size Tuning**: Increase batch size if memory allows

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

- [ ] **Multi-class Classification**: Extend to confidence levels
- [ ] **Additional Models**: BERT, RoBERTa comparisons
- [ ] **Cross-domain Evaluation**: Test on different domains
- [ ] **API Deployment**: REST API for inference
- [ ] **Confidence Calibration**: Improve probability estimates
- [ ] **Adversarial Testing**: Robustness evaluation

## 📚 References

- [FEVER Dataset](https://fever.ai/) - Fact Extraction and VERification
- [TruthfulQA](https://arxiv.org/abs/2109.07958) - Measuring How Models Mimic Human Falsehoods
- [DistilBERT](https://arxiv.org/abs/1910.01108) - Distilled BERT for Efficient NLP
- [Hallucination Detection](https://arxiv.org/abs/2310.15319) - Recent approaches in hallucination detection

## 👥 Authors

- **Felix Akwerh** - Initial work and implementation

## 🙏 Acknowledgments

- FEVER and TruthfulQA dataset creators
- HuggingFace Transformers library
- The research community working on hallucination detection

---

**Note**: This implementation emphasizes high recall for hallucination detection, making it suitable for safety-critical applications where missing hallucinations is more costly than false alarms.



# 📘 Self-Generated Counterfactual Training (SGCT)

## Overview

Self-Generated Counterfactual Training (SGCT) is a framework for improving language models by **teaching them to avoid hallucinations**. The method combines:

* **Generator (GPT-2)** → produces multiple candidate answers.
* **Critic (HPM)** → evaluates factual consistency of each candidate.
* **Counterfactual Mining** → selects the best factual and worst hallucinated answers.
* **Fine-Tuning** → retrains GPT-2 on this counterfactually curated dataset.

This approach enables **self-improvement**: the model learns both **what to say** (factual) and **what not to say** (hallucinated).

---

## 🔄 SGCT Pipeline

### 1. Input Preparation

Each training example is structured as:

```
Question: <question_text>
Evidence: <supporting_text>
Answer:
```

* `question`: a user query.
* `evidence`: factual supporting passage or knowledge snippet.
* The model must generate an answer **conditioned only on the evidence**.

---

### 2. Candidate Generation (GPT-2)

* For each prompt, GPT-2 generates `K` diverse answers.
* Sampling uses **top-p nucleus sampling** and **temperature scaling** to encourage variability.
* Example:

  ```
  Q: Who invented the telephone?
  Evidence: Alexander Graham Bell is credited with inventing the telephone in 1876.
  ```

  Candidates:

  * “Alexander Graham Bell in 1876.” ✅ factual
  * “Thomas Edison invented it.” ❌ hallucinated
  * “It is unknown who did.” ❓ borderline

---

### 3. Batch HPM Scoring -Mining Stage

* Each candidate is scored by the **Hallucination Prediction Model (HPM)**.
* Inputs are paired as:

  ```
  Evidence: <evidence>
  Answer: <candidate>
  ```
* HPM outputs `(label, score)`:

  * `label=1` → factual, `label=0` → hallucination.
  * `score` → confidence/probability.

**Batching:**
Instead of evaluating candidates one by one, we feed batches of inputs into HPM. This:

* Runs efficiently on GPU.
* Avoids pipeline warnings about sequential calls.
* Speeds up large-scale mining.

---

### 4. Counterfactual Selection

For each question:

* **Positive Target:** keep the **highest-scoring factual candidate** above threshold `τ_pos`.
* **Negative Target (optional):** keep the **lowest-scoring hallucinated candidate** below threshold `τ_neg`.
* Discard ambiguous candidates in between.

Example:

* “Bell (prob=0.92)” → kept as **positive**.
* “Edison (prob=0.18)” → kept as **negative**.
* “Unknown (prob=0.45)” → ignored.

---

### 5. Counterfactual Dataset

The mined dataset looks like:

```json
{
  "prompt": "Question: Who invented the telephone?\nEvidence: Bell\nAnswer:",
  "target_pos": "Alexander Graham Bell in 1876.",
  "target_neg": "Thomas Edison invented it."
}
```

This provides **both supervision and anti-supervision**:

* `target_pos`: factual answers to reinforce.
* `target_neg`: hallucinations to penalize (via unlikelihood loss).

---

### 6. Fine-Tuning (SGCT Training)

* GPT-2 is fine-tuned with `prompt → target_pos` pairs.
* Optionally, **unlikelihood loss** penalizes generating `target_neg`.
* Training hyperparameters (recommended):

  * `MAX_LEN = 256`
  * `BATCH_SIZE = 8`
  * `EPOCHS = 3`
  * `LEARNING_RATE = 2e-5`

This balances stability (low learning rate) and sufficient exploration (moderate epochs/batch size).

---

### 7. Outcome

* GPT-2 **internalizes factual patterns**: it is biased toward generating answers that align with evidence.
* The HPM acts as a **factuality filter**, continuously steering generation toward truth.
* Over iterations, SGCT improves robustness against hallucinations.

---

## 📊 Advantages of SGCT

1. **Self-improving:** no need for manual annotation—HPM provides supervision.
2. **Dual supervision:** both factual reinforcement and hallucination avoidance.
3. **Scalable:** works with large datasets and batched mining.
4. **Flexible:** can use any base generator (GPT-2, T5, LLaMA) and critic (DistilBERT, RoBERTa).

---

## ⚠️ Limitations

* **Quality of HPM is critical**: if HPM mislabels, the dataset may include noisy supervision.
* **Threshold sensitivity:** setting `τ_pos` and `τ_neg` too aggressively may reduce diversity.
* **Computationally expensive:** requires generating and scoring many candidates per example.

---

## 🚀 Future Directions

* Compare with existimg model to test the effectiveness of the pipeline

---

✅ With this pipeline, you are essentially **teaching GPT-2 by trial and error**, where HPM acts as a **teacher** filtering the model’s own mistakes into a **curriculum of counterfactual examples**.


