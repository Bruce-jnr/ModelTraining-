import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationDataset(Dataset):
    """Custom dataset for hallucination detection"""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        # Format: [CLS] question [SEP] answer
        text = f"{item['question']} [SEP] {item['answer']}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


class DataPreprocessor:
    """Handles dataset preparation and preprocessing"""

    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def load_merged_dataset(self, dataset_path):
        """Load the pre-merged FEVER and TruthfulQA dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Ensure we have the required columns
            required_columns = ['question', 'answer', 'label']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in dataset")

            # Add evidence column if not present (though it should be there)
            if 'evidence' not in df.columns:
                df['evidence'] = ''

            # Log dataset statistics
            logger.info(f"Loaded dataset with {len(df)} samples")
            logger.info(f"Columns: {list(df.columns)}")

            # Show distribution by source if available
            if 'source' in df.columns:
                source_dist = df['source'].value_counts()
                logger.info(f"Source distribution: {source_dist.to_dict()}")

            # Show label distribution
            label_dist = df['label'].value_counts()
            logger.info(f"Label distribution: {label_dist.to_dict()}")
            logger.info(
                f"  Label 1 (Factual): {label_dist.get(1, 0)} samples ({label_dist.get(1, 0) / len(df) * 100:.1f}%)")
            logger.info(
                f"  Label 0 (Hallucinated): {label_dist.get(0, 0)} samples ({label_dist.get(0, 0) / len(df) * 100:.1f}%)")

            # Show sample data
            logger.info("Sample entries:")
            for i, sample in enumerate(df.head(2).to_dict('records')):
                logger.info(f"  Sample {i + 1}: {sample}")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset from {dataset_path}: {str(e)}")
            raise

    def merge_datasets(self, fever_path=None, truthfulqa_path=None,
                       fever_df=None, truthfulqa_df=None):
        """Merge FEVER and TruthfulQA datasets"""

        if fever_df is None and fever_path:
            fever_df = self.load_fever_data(fever_path)
        if truthfulqa_df is None and truthfulqa_path:
            truthfulqa_df = self.load_truthfulqa_data(truthfulqa_path)

        # Merge datasets
        merged_data = pd.concat([fever_df, truthfulqa_df], ignore_index=True)

        # Shuffle the dataset
        merged_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Clean data
        merged_data = merged_data.dropna()

        logger.info(f"Merged dataset size: {len(merged_data)}")
        logger.info(f"Class distribution: {merged_data['label'].value_counts().to_dict()}")

        return merged_data

    def stratified_split(self, data, train_size=0.7, val_size=0.15, test_size=0.15):
        """Perform stratified split maintaining class balance"""

        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"

        # First split: separate train from (val + test)
        X = data.drop('label', axis=1)
        y = data['label']

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size),
            stratify=y, random_state=42
        )

        # Second split: separate val from test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size / (val_size + test_size),
            stratify=y_temp, random_state=42
        )

        # Recombine features and labels
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        logger.info(f"Train size: {len(train_data)} ({len(train_data) / len(data) * 100:.1f}%)")
        logger.info(f"Val size: {len(val_data)} ({len(val_data) / len(data) * 100:.1f}%)")
        logger.info(f"Test size: {len(test_data)} ({len(test_data) / len(data) * 100:.1f}%)")

        # Log class distributions
        for name, dataset in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            dist = dataset['label'].value_counts(normalize=True)
            logger.info(f"{name} class distribution: {dist.to_dict()}")

        return train_data, val_data, test_data


class HallucinationDetectorTrainer:
    """Main trainer class for the hallucination detector"""

    def __init__(self, model_name='distilbert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_accuracies = []

        # Early stopping parameters
        self.best_f1 = 0
        self.patience_counter = 0

    def create_data_loaders(self, train_data, val_data, test_data, batch_size=8):
        """Create data loaders for training, validation, and testing"""

        train_dataset = HallucinationDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = HallucinationDataset(val_data, self.tokenizer, self.max_length)
        test_dataset = HallucinationDataset(test_data, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def initialize_model(self):
        """Initialize the DistilBERT model for sequence classification"""

        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "HALLUCINATED", 1: "FACTUAL"},
            label2id={"HALLUCINATED": 0, "FACTUAL": 1}
        )
        self.model.to(self.device)

        return self.model

    def setup_optimizer_scheduler(self, train_loader, epochs=None, lr=None):
        """Setup optimizer and learning rate scheduler"""

        # Get hyperparameters from environment variables with defaults
        if epochs is None:
            epochs = int(os.getenv('EPOCHS', 3))
        if lr is None:
            lr = float(os.getenv('LEARNING_RATE', 2e-5))

        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning Rate: {lr}")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        return epochs, lr

    def compute_metrics(self, predictions, labels):
        """Compute evaluation metrics"""

        preds = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1
        )
        accuracy = accuracy_score(labels, preds)

        # Compute AUC
        probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
        auc = roc_auc_score(labels, probs)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

    def validate(self, val_loader):
        """Validation step"""

        self.model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_val_loss += outputs.loss.item()
                predictions.extend(outputs.logits.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        metrics = self.compute_metrics(np.array(predictions), np.array(true_labels))

        return avg_val_loss, metrics

    def save_checkpoint(self, epoch, checkpoint_dir="checkpoints"):
        """Save model checkpoint"""

        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'val_accuracies': self.val_accuracies
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def early_stopping_check(self, val_f1, patience=3):
        """Check for early stopping condition"""

        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.patience_counter = 0
            return False  # Continue training
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                return True  # Stop training

        return False

    def plot_training_history(self, save_path="training_history.png"):
        """Plot training history"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Training and validation loss
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Validation F1 Score
        ax2.plot(epochs, self.val_f1_scores, 'g-', label='Validation F1')
        ax2.set_title('Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)

        # Validation Accuracy
        ax3.plot(epochs, self.val_accuracies, 'm-', label='Validation Accuracy')
        ax3.set_title('Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)

        # All metrics combined
        ax4.plot(epochs, self.val_f1_scores, 'g-', label='F1 Score')
        ax4.plot(epochs, self.val_accuracies, 'm-', label='Accuracy')
        ax4.set_title('Validation Metrics')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

        logger.info(f"Training history plot saved: {save_path}")

    def train(self, train_loader, val_loader, epochs=3, patience=3,
              checkpoint_dir="checkpoints", save_best_only=False):
        """Main training loop with monitoring and early stopping"""

        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {epochs}, Patience: {patience}")

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Training phase
            self.model.train()
            total_train_loss = 0

            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                total_train_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg Loss': f"{total_train_loss / (batch_idx + 1):.4f}"
                })

                # Validation monitoring every few steps
                if batch_idx % 100 == 0 and batch_idx > 0:
                    logger.info(f"Step {batch_idx}: Current loss = {loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation phase
            logger.info("Running validation...")
            avg_val_loss, val_metrics = self.validate(val_loader)

            self.val_losses.append(avg_val_loss)
            self.val_f1_scores.append(val_metrics['f1'])
            self.val_accuracies.append(val_metrics['accuracy'])

            # Log metrics
            logger.info(f"Epoch {epoch + 1} Results:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val Precision: {val_metrics['precision']:.4f}")
            logger.info(f"  Val Recall: {val_metrics['recall']:.4f}")
            logger.info(f"  Val F1: {val_metrics['f1']:.4f}")
            logger.info(f"  Val AUC: {val_metrics['auc']:.4f}")

            # Save checkpoint
            if not save_best_only or val_metrics['f1'] > self.best_f1:
                self.save_checkpoint(epoch + 1, checkpoint_dir)

            # Early stopping check
            if self.early_stopping_check(val_metrics['f1'], patience):
                logger.info("Training stopped early due to no improvement")
                break

        logger.info("Training completed!")
        logger.info(f"Best validation F1 score: {self.best_f1:.4f}")

        # Plot training history
        self.plot_training_history()

    def evaluate_test_set(self, test_loader):
        """Evaluate model on test set"""

        logger.info("Evaluating on test set...")

        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions.extend(outputs.logits.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        test_metrics = self.compute_metrics(np.array(predictions), np.array(true_labels))

        logger.info("Test Set Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  AUC: {test_metrics['auc']:.4f}")

        return test_metrics


def main():
    """Main execution function"""

    # Load environment variables
    load_dotenv()

    # Get configuration from environment variables
    dataset_path = os.getenv('DATASET_PATH')
    model_name = os.getenv('MODEL_NAME', 'distilbert-base-uncased')
    batch_size = int(os.getenv('BATCH_SIZE', 8))
    learning_rate = float(os.getenv('LEARNING_RATE', 2e-5))
    epochs = int(os.getenv('EPOCHS', 3))
    patience = int(os.getenv('PATIENCE', 3))

    # Validate required environment variables
    if not dataset_path:
        raise ValueError("DATASET_PATH not found in environment variables. Please check your .env file.")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    # Log configuration
    logger.info("=== Training Configuration ===")
    logger.info(f"Dataset Path: {dataset_path}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Patience: {patience}")
    logger.info("==============================")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    logger.info(f"Loading dataset from: {dataset_path}")
    merged_data = preprocessor.load_merged_dataset(dataset_path)

    # Clean data
    merged_data = merged_data.dropna()
    logger.info(f"Dataset after cleaning: {len(merged_data)} samples")
    logger.info(f"Class distribution: {merged_data['label'].value_counts().to_dict()}")

    # Perform stratified split
    train_data, val_data, test_data = preprocessor.stratified_split(merged_data)

    # Initialize trainer with model from environment
    trainer = HallucinationDetectorTrainer(model_name=model_name)

    # Create data loaders with batch size from environment
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_data, val_data, test_data, batch_size=batch_size
    )

    # Initialize model
    model = trainer.initialize_model()

    # Setup optimizer and scheduler with parameters from environment
    actual_epochs, actual_lr = trainer.setup_optimizer_scheduler(
        train_loader, epochs=epochs, lr=learning_rate
    )

    # Train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=actual_epochs,
        patience=patience,
        checkpoint_dir="hallucination_detector_checkpoints"
    )

    # Evaluate on test set
    test_metrics = trainer.evaluate_test_set(test_loader)

    # Save final model
    trainer.model.save_pretrained("hallucination_detector_final")
    trainer.tokenizer.save_pretrained("hallucination_detector_final")

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()