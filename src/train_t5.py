"""
T5/FLAN-T5 Fine-tuning Script

Fine-tunes a FLAN-T5 or T5-base model for text-to-JSON generation
using the construction rules dataset.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import numpy as np


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_metrics(eval_pred):
    """
    Compute custom metrics: exact match and JSON validity rate.
    """
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    # Note: predictions are logits, we need to decode them
    # This is a simplified version - in practice, you'd decode here
    # For now, we'll compute basic metrics
    
    # Exact match would require decoding, so we'll use a placeholder
    # In a real implementation, you'd decode predictions and compare strings
    
    return {
        "exact_match": 0.0,  # Placeholder - would need decoding
        "json_validity": 0.0  # Placeholder - would need decoding
    }


def prepare_dataset(jsonl_path: str, tokenizer, max_input_length: int = 512, max_target_length: int = 512):
    """
    Prepare the dataset for training.

    Args:
        jsonl_path: Path to JSONL dataset
        tokenizer: Tokenizer instance
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
    """
    # Load data
    data = load_jsonl_dataset(jsonl_path)

    # Prepare inputs and targets
    inputs = [item["input_text"] for item in data]
    targets = [item["target_json"] for item in data]

    # Create dataset from raw data
    dataset_dict = {
        "input_text": inputs,
        "target_json": targets
    }

    dataset = Dataset.from_dict(dataset_dict)

    # Define tokenization function
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding=False  # Dynamic padding will be done by DataCollator
        )

        # Tokenize targets (labels)
        labels = tokenizer(
            text_target=examples["target_json"],
            max_length=max_target_length,
            truncation=True,
            padding=False  # Dynamic padding will be done by DataCollator
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def train_model(
    model_name: str = "google/flan-t5-base",
    dataset_path: str = "./dataset/construction_ashrae_2013.jsonl",
    output_dir: str = "./models/flan_t5_construction_ashrae_2013",
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    num_epochs: int = 5,
    weight_decay: float = 0.01,
    max_input_length: int = 512,
    max_target_length: int = 512,
    eval_split: float = 0.1
):
    """
    Train a T5/FLAN-T5 model for text-to-JSON generation.
    
    Args:
        model_name: HuggingFace model name (e.g., "google/flan-t5-base" or "t5-base")
        dataset_path: Path to JSONL training dataset
        output_dir: Directory to save the fine-tuned model
        learning_rate: Learning rate for training
        batch_size: Training batch size
        num_epochs: Number of training epochs
        weight_decay: Weight decay for regularization
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        eval_split: Fraction of data to use for evaluation
    """
    print("="*80)
    print("FINE-TUNING T5/FLAN-T5 MODEL")
    print("="*80)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Prepare dataset
    print(f"\nLoading dataset from {dataset_path}")
    dataset = prepare_dataset(dataset_path, tokenizer, max_input_length, max_target_length)
    
    # Split into train and eval
    if eval_split > 0:
        dataset = dataset.train_test_split(test_size=eval_split, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        print(f"Train examples: {len(train_dataset)}")
        print(f"Eval examples: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Calculate reasonable eval/save steps
    # For a dataset with ~2500 examples and batch_size=8, we have ~310 steps per epoch
    # Evaluating 2-3 times per epoch is reasonable
    steps_per_epoch = len(train_dataset) // batch_size
    eval_save_steps = max(50, steps_per_epoch // 3)  # Eval 3 times per epoch, minimum 50 steps

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_steps=eval_save_steps,
        eval_steps=eval_save_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
        predict_with_generate=False,  # Set to True if you want to see generated text during eval
        fp16=torch.cuda.is_available(),  # Enable mixed precision training on GPU
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,  # Uncomment if implementing metrics
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n[OK] Training complete! Model saved to {output_dir}")
    
    # Print some evaluation metrics
    if eval_dataset:
        print("\nRunning final evaluation...")
        eval_results = trainer.evaluate()
        print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune T5/FLAN-T5 for text-to-JSON")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-base",
        help="HuggingFace model name (default: google/flan-t5-base)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset/construction_ashrae_2013.jsonl",
        help="Path to JSONL dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/flan_t5_construction_ashrae_2013",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Get absolute paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / args.dataset_path if not Path(args.dataset_path).is_absolute() else Path(args.dataset_path)
    output_dir = project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    train_model(
        model_name=args.model_name,
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay
    )

