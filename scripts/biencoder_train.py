from logging import config
import os
import yaml
import random
import mlflow
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from huggingface_hub import login

from scripts.biencoder_evaluate import compute_metrics
from models.biencoder import BiEncoderModel
from dataset import ZeroShotDataset, collate_fn


mlflow.set_tracking_uri("http://127.0.0.1:5000")

def hard_negative_contrastive_loss(
    similarity,
    positive_mask,
    num_hard_negatives=7
):
    """
    similarity: [B, M]
    positive_mask: [B, M]
    """

    B, M = similarity.shape
    device = similarity.device

    losses = []

    for i in range(B):

        sim_row = similarity[i]                      # [M]
        pos_mask = positive_mask[i].bool()           # [M]

        pos_indices = torch.where(pos_mask)[0]

        # Skip samples with no positives (safety)
        if len(pos_indices) == 0:
            continue

        neg_mask = ~pos_mask
        neg_indices = torch.where(neg_mask)[0]

        # If no negatives exist (rare edge case)
        if len(neg_indices) == 0:
            continue

        neg_scores = sim_row[neg_indices].detach()

        # Select top-k hardest negatives
        k = min(num_hard_negatives, len(neg_scores))
        hard_neg_relative = torch.topk(neg_scores, k).indices
        hard_neg_indices = neg_indices[hard_neg_relative]

        # Combine positives + hard negatives
        selected_indices = torch.cat([pos_indices, hard_neg_indices])

        selected_sim = sim_row[selected_indices]

        # Log-softmax over selected subset
        log_probs = selected_sim - torch.logsumexp(selected_sim, dim=0)

        # Positive positions are first in selected_indices
        pos_count = len(pos_indices)

        loss_i = -log_probs[:pos_count].mean()

        losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(losses).mean()


# --------------------------------------------------
# Zero-shot split (uses config now)
# --------------------------------------------------
def zero_shot_split(dataset, zero_shot_ratio=0.15, seed=42):

    all_labels = list(dataset.all_labels)
    random.Random(seed).shuffle(all_labels)

    split_idx = int((1 - zero_shot_ratio) * len(all_labels))
    train_labels = set(all_labels[:split_idx])
    zero_shot_labels = set(all_labels[split_idx:])

    train_data = []
    test_data = []

    for sample in dataset.data:
        labels = set(sample["labels"])

        # Train samples: only contain train labels
        if labels.issubset(train_labels):
            train_data.append(sample)

        # Test samples: contain at least one unseen label
        elif labels & zero_shot_labels:
            test_data.append(sample)

    return train_data, test_data, list(train_labels), list(zero_shot_labels)


# --------------------------------------------------
# Training
# --------------------------------------------------
def train(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------
    # Dataset
    # ----------------------------------------------
    dataset = ZeroShotDataset(config["data"]["synthetic_data_path"])

    train_data, test_data, raw_train_labels, raw_zero_shot_labels = zero_shot_split(
            dataset,
            zero_shot_ratio=0.15,
            seed=42,
        )

# Keep raw for indexing
    train_label_to_index = {l: i for i, l in enumerate(raw_train_labels)}
    num_train_labels = len(raw_train_labels)

    # Prompted versions for encoding
    train_labels = [f"This text is about {l}." for l in raw_train_labels]
    zero_shot_labels = [f"This text is about {l}." for l in raw_zero_shot_labels]
    

    train_data, val_data = train_test_split(
        train_data,
        test_size=0.1,
        random_state=42
    )
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")


    # ----------------------------------------------
    # DataLoaders
    # ----------------------------------------------
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            train_label_to_index,
            num_train_labels
        )
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch,
            train_label_to_index,
            num_train_labels
        )
    )

    # ----------------------------------------------
    # Model
    # ----------------------------------------------
    model = BiEncoderModel(config["model"]["name"])
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ----------------------------------------------
    # Training loop
    # ----------------------------------------------
    run_name = f"{config['model']['name']}-{config['training']['num_steps']}steps"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(config["model"])
        mlflow.log_params(config["training"])

        step = 0
        total_steps = config["training"]["num_steps"]
        val_interval = config["training"]["val_check_interval"]
        patience = config["training"].get("early_stopping_patience", 5)

        best_val_loss = float("inf")
        patience_counter = 0
        best_step = 0

        pbar = tqdm(total=total_steps)
        model.train()

        while step < total_steps:

            for texts, positive_mask in train_loader:

                if step >= total_steps:
                    break

                positive_mask = positive_mask.to(device)

                similarity = model(texts, train_labels, device)

                loss = hard_negative_contrastive_loss(
                    similarity,
                    positive_mask,
                    num_hard_negatives=config["training"].get("num_hard_negatives", 10)
            )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["max_grad_norm"]
                )

                optimizer.step()

                mlflow.log_metric("train_loss", loss.item(), step=step)

                if step % 50 == 0:
                    print(f"Step {step}: Train Loss {loss.item():.4f}")

                # ------------------------------------------
                # Validation + Early Stopping
                # ------------------------------------------
                if step % val_interval == 0:

                    model.eval()
                    val_loss = 0

                    with torch.no_grad():
                        for v_texts, v_mask in val_loader:
                            v_mask = v_mask.to(device)

                            v_sim = model(v_texts, train_labels, device)
                            v_loss = hard_negative_contrastive_loss(
                                v_sim,
                                v_mask
                            )

                            val_loss += v_loss.item()

                    avg_val_loss = val_loss / len(val_loader)
                    mlflow.log_metric("val_loss", avg_val_loss, step=step)

                    print(f"Step {step}: Val Loss {avg_val_loss:.4f}")

                    # 🔥 Check improvement
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_step = step
                        patience_counter = 0

                        print(f"New best model at step {step}. Saving...")

                        os.makedirs("hard-neg-minilm", exist_ok=True)
                        model.encoder.save_pretrained("hard-neg-minilm")
                        model.tokenizer.save_pretrained("hard-neg-minilm")

                    else:
                        patience_counter += 1
                        print(f"No improvement. Patience {patience_counter}/{patience}")

                    model.train()

                    # 🔥 Early stopping
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        step = total_steps  # force outer loop exit
                        break

                step += 1
                pbar.update(1)

        pbar.close()

        print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}")
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # ----------------------------------------------
        # Save model
        # ----------------------------------------------
        mlflow.log_artifacts("hard-neg-minilm")
        
        print("Training complete. Running zero-shot evaluation...")
        
        best_model = BiEncoderModel(config["model"]["name"])
        best_model.encoder = best_model.encoder.from_pretrained("hard-neg-minilm")
        best_model.tokenizer = best_model.tokenizer.from_pretrained("hard-neg-minilm")
        best_model.to(device)

        if config["model"].get("push_to_hub", False):
            login()
            best_model.encoder.push_to_hub(config["model"]["hub_repo"])
            best_model.tokenizer.push_to_hub(config["model"]["hub_repo"])

        # ----------------------------------------------
        # Zero-shot evaluation
        # ----------------------------------------------
        
        
        # all_eval_labels = train_labels + zero_shot_labels
        all_eval_labels = train_labels + zero_shot_labels
        

        metrics = compute_metrics(
            model=best_model,
            data=test_data,
            all_labels=all_eval_labels,
            zero_shot_labels=raw_zero_shot_labels,
            device=device,
            k=config["evaluation"]["top_k"]
        )

        print("Zero-Shot Test Metrics:", metrics)

        for key, value in metrics.items():
            mlflow.log_metric(f"test_{key}", value)

        print("Training + Evaluation complete.")


if __name__ == "__main__":
    train("config.yaml")

