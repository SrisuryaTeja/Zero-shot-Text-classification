import os
import yaml
import random
import mlflow
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from huggingface_hub import login

from scripts.polyencoder_evaluate import compute_metrics
from models.polyencoder import PolyEncoderModel
from dataset import ZeroShotDataset, collate_fn


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Multi-positive contrastive loss (numerically safe)

def multi_positive_contrastive_loss(similarity, positive_mask):
    log_probs = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)

    denom = positive_mask.sum(1).clamp(min=1)  # prevent divide by zero
    loss = -(log_probs * positive_mask).sum(1) / denom

    return loss.mean()


# Label sampling

def build_sampled_label_space(batch_positive_mask, num_negatives):
    B, L = batch_positive_mask.shape

    positive_indices = torch.where(batch_positive_mask.sum(0) > 0)[0].tolist()

    all_indices = set(range(L))
    negative_pool = list(all_indices - set(positive_indices))

    num_negatives = min(num_negatives, len(negative_pool))
    sampled_negatives = random.sample(negative_pool, num_negatives)

    return positive_indices + sampled_negatives



# Zero-shot label split

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

        if labels.issubset(train_labels):
            train_data.append(sample)
        elif labels & zero_shot_labels:
            test_data.append(sample)

    return train_data, test_data, list(train_labels), list(zero_shot_labels)



# Training

def train(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ZeroShotDataset(config["data"]["synthetic_data_path"])

    train_data, test_data, raw_train_labels, raw_zero_shot_labels = zero_shot_split(
        dataset,
        zero_shot_ratio=0.15,
        seed=42,
    )

    train_label_to_index = {l: i for i, l in enumerate(raw_train_labels)}
    num_train_labels = len(raw_train_labels)

    train_labels = [f"This text is about {l}." for l in raw_train_labels]
    zero_shot_labels = [f"This text is about {l}." for l in raw_zero_shot_labels]

    train_data, val_data = train_test_split(
        train_data,
        test_size=0.1,
        random_state=42
    )

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

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

    model = PolyEncoderModel(
        config["model"]["name"],
        num_poly_codes=config["model"]["num_poly_codes"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    save_dir = "poly-encoder-neg-60"
    os.makedirs(save_dir, exist_ok=True)

    with mlflow.start_run(run_name=f"{config['model']['name']}"):

        mlflow.log_params(config["model"])
        mlflow.log_params(config["training"])

        step = 0
        total_steps = config["training"]["num_steps"]
        val_interval = config["training"]["val_check_interval"]
        patience = config["training"].get("early_stopping_patience", 5)

        best_val_loss = float("inf")
        patience_counter = 0

        pbar = tqdm(total=total_steps)

        model.train()

        while step < total_steps:

            for texts, positive_mask in train_loader:

                if step >= total_steps:
                    break

                positive_mask = positive_mask.to(device)

                final_indices = build_sampled_label_space(
                    positive_mask,
                    config["training"]["num_negatives"]
                )

                sampled_labels = [train_labels[i] for i in final_indices]
                sampled_mask = positive_mask[:, final_indices]

                similarity = model(texts, sampled_labels, device)

                loss = multi_positive_contrastive_loss(
                    similarity,
                    sampled_mask
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["max_grad_norm"]
                )
                optimizer.step()

                mlflow.log_metric("train_loss", loss.item(), step=step)

                #  Validation 
                if step % val_interval == 0:

                    model.eval()
                    val_loss = 0

                    with torch.no_grad():
                        for v_texts, v_mask in val_loader:
                            v_mask = v_mask.to(device)

                            v_sim = model(v_texts, train_labels, device)
                            v_loss = multi_positive_contrastive_loss(
                                v_sim,
                                v_mask
                            )

                            val_loss += v_loss.item()

                    avg_val_loss = val_loss / len(val_loader)
                    mlflow.log_metric("val_loss", avg_val_loss, step=step)

                    print(f"Step {step} | Val Loss: {avg_val_loss:.4f}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0

                        torch.save(
                            model.state_dict(),
                            os.path.join(save_dir, "model.pt")
                        )

                        model.tokenizer.save_pretrained(save_dir)

                        print("Saved best model.")

                    else:
                        patience_counter += 1
                        print(f"No improvement ({patience_counter}/{patience})")

                    model.train()

                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        step = total_steps
                        break

                step += 1
                pbar.update(1)

        pbar.close()

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifacts(save_dir)

        print("Running zero-shot evaluation...")

        best_model = PolyEncoderModel(
            config["model"]["name"],
            num_poly_codes=config["model"]["num_poly_codes"]
        )

        best_model.load_state_dict(
            torch.load(os.path.join(save_dir, "model.pt"), map_location=device)
        )

        best_model.tokenizer = best_model.tokenizer.from_pretrained(save_dir)
        best_model.to(device)
        best_model.eval()

        all_eval_labels = list(set(train_labels + zero_shot_labels))

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
    train("polyencoder-config.yaml")