import torch
from tqdm import tqdm
def compute_metrics(model, data, all_labels, zero_shot_labels, device, k=5):

    model.eval()

    precision_total = 0
    recall_total = 0
    f1_total = 0
    mrr_total = 0
    valid_samples = 0

    zero_shot_label_set = set(zero_shot_labels)

    with torch.no_grad():

        for sample in tqdm(data, desc="Evaluating"):

            text = sample["text"]
            true_labels = set(sample["labels"]) & zero_shot_label_set

            if not true_labels:
                continue

            valid_samples += 1

            similarity = model([text], all_labels, device)
            scores = similarity.squeeze(0)

            ranked_indices = torch.argsort(scores, descending=True)

            topk_indices = ranked_indices[:k]

            predicted_labels = [
                all_labels[i].replace("This text is about ", "").replace(".", "")
                for i in topk_indices.tolist()
            ]

            hits = len(set(predicted_labels) & true_labels)

            precision = hits / k
            recall = hits / len(true_labels)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            precision_total += precision
            recall_total += recall
            f1_total += f1

            # MRR
            reciprocal_rank = 0
            for rank, idx in enumerate(ranked_indices.tolist()):
                raw_label = all_labels[idx].replace("This text is about ", "").replace(".", "")
                if raw_label in true_labels:
                    reciprocal_rank = 1 / (rank + 1)
                    break

            mrr_total += reciprocal_rank

    if valid_samples == 0:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_at_k": 0.0,
            "mrr": 0.0,
        }

    return {
        "precision_at_k": precision_total / valid_samples,
        "recall_at_k": recall_total / valid_samples,
        "f1_at_k": f1_total / valid_samples,
        "mrr": mrr_total / valid_samples,
    }