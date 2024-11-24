import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample
from config import MODEL_NAME, DEVICE, EPOCHS
from utils import custom_collate_fn
import pandas as pd
import time

import ast
import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

def train_model(working_path):
    train_data = pd.read_csv(f"{working_path}train_data.csv")
    val_data = pd.read_csv(f"{working_path}val_data.csv")
    train_data = train_data.sample(frac = 0.1)
    val_data = val_data.sample(frac = 0.1)

    model = SentenceTransformer(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_examples = [InputExample(texts=[row['processed_text'], row['original_claim']], label=float(row['label'])) for _, row in train_data.iterrows()]
    val_examples = [InputExample(texts=[row['processed_text'], row['original_claim']], label=float(row['label'])) for _, row in val_data.iterrows()]

    train_dataloader = DataLoader(train_examples, batch_size=16, collate_fn=custom_collate_fn(tokenizer), shuffle=True)
    val_dataloader = DataLoader(val_examples, batch_size=16, collate_fn=custom_collate_fn(tokenizer), shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.2)
    triplet_margin_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    scaler = GradScaler()
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # Early stopping configuration
    early_stopping_patience = 3
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    num_epochs = EPOCHS
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0

        # Track epoch time
        epoch_start_time = time.time()

        for step, (features, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear unused memory

            # Move inputs and labels to the same device as the model
            features = [{key: val.to(DEVICE) for key, val in f.items()} for f in features]
            labels = labels.to(DEVICE)

            # Generate anchor, positive, and negative samples
            anchors = model(features[0])['sentence_embedding']
            positives = model(features[1])['sentence_embedding']
            negatives = torch.roll(positives, shifts=1, dims=0)  # Simple negative sampling (can be improved)

            # Mixed precision training
            with autocast():
                loss = triplet_margin_loss(anchors, positives, negatives)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} completed. Avg Training Loss = {avg_loss:.4f}. Time elapsed: {time.time() - epoch_start_time:.2f}s")

        # Validation
        model.eval()
        val_loss = 0
        val_start_time = time.time()
        with torch.no_grad():
            for val_step, (val_features, val_labels) in enumerate(val_dataloader):
                # Move inputs and labels to the same device as the model
                val_features = [{key: val.to(DEVICE) for key, val in f.items()} for f in val_features]

                # Generate validation embeddings
                anchors = model(val_features[0])['sentence_embedding']
                positives = model(val_features[1])['sentence_embedding']
                negatives = torch.roll(positives, shifts=1, dims=0)

                with autocast():
                    loss = triplet_margin_loss(anchors, positives, negatives)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation - Loss: {avg_val_loss:.4f}, Time elapsed: {time.time() - val_start_time:.2f}s")

        # Update learning rate scheduler
        lr_scheduler.step(avg_val_loss)

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            model.save('/home/stud/haroonm0/localdisk/FactCheck/kaggle/working/best_model')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered. Stopping training.")
                break

    # Save the final model
    model_save_path = '/home/stud/haroonm0/localdisk/FactCheck/kaggle/working/minilm-finetuned'
    model.save(model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")
    print("Model training complete.")
