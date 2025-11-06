import torch
import logging
import sys
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
os.environ["WANDB_DISABLED"] = "true"


# Set up logging to print to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    print(f"Eval metrics: accuracy={acc:.4f}, f1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}", flush=True)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("Loading BASIL dataset...", flush=True)
    dataset = load_dataset("horychtom/BASIL")

    print("Splitting train/validation...", flush=True)
    train_testvalid = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_testvalid['train']
    val_dataset = train_testvalid['test']

    print("Initializing tokenizer and model...", flush=True)
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    print("Tokenizing datasets...", flush=True)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'lex_label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'lex_label'])

    train_dataset = train_dataset.rename_column("lex_label", "labels")
    val_dataset = val_dataset.rename_column("lex_label", "labels")

    print("Setting up training arguments...", flush=True)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=1,           # Log every step for detailed output
        log_level="info",
        log_level_replica="info",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=None             # Only log to stdout, not wandb/tensorboard
    )

    print("Initializing Trainer...", flush=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...", flush=True)
    trainer.train()
    print("Training complete!", flush=True)

    print("Saving final model and tokenizer...", flush=True)
    trainer.save_model("./models/fine_tuned_distilbert_basil_epoch_10")
    tokenizer.save_pretrained("./models/fine_tuned_distilbert_basil_epoch_10")
    print("Model and tokenizer saved.", flush=True)

if __name__ == "__main__":
    main()
