import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm


def training_batch(data, labels, model, criterion, optimizer, device="cuda"):
    model.train()
    data = data.to(device)
    labels = labels.to(device)
    output = model(data)
    loss = criterion(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@torch.inference_mode
def test_batch(data, labels, model, criterion, device="cuda"):
    model.eval()
    data = data.to(device)
    labels = labels.to(device)
    output = model(data)
    loss = criterion(output, labels)
    return loss


@torch.inference_mode
def accuracy_precision_recall(data, labels, model, device="cuda"):
    model.eval()
    data = data.to(device)
    labels = labels.to(device)
    output = model(data)
    _, pred_labels = output.max(-1)
    pred_labels = pred_labels.cpu().detach().numpy()
    true_labels = labels.cpu().detach().numpy()
    accuracy = (pred_labels == true_labels).mean()
    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    return accuracy, precision, recall


def train_model(model_to_train, criterion, optimizer, training_dl, test_dl, n_epochs=20):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    training_loss, test_loss = [], []
    training_metrics, test_metrics = [], []

    for epoch in range(n_epochs):
        print(f"** Starting epoch {epoch}/{n_epochs}")
        training_losses, test_losses = [], []
        training_accuracies, training_precisions, training_recalls = [], [], []
        test_accuracies, test_precisions, test_recalls = [], [], []

        model_to_train.train()
        for data, labels in tqdm(training_dl, "Training"):
            trng_batch_loss = training_batch(data, labels, model_to_train, criterion, optimizer)
            training_losses.append(trng_batch_loss.item())
            acc, prec, rec = accuracy_precision_recall(data, labels, model_to_train)
            training_accuracies.append(acc)
            training_precisions.append(prec)
            training_recalls.append(rec)
        training_per_epoch_metrics = {
            "loss": np.mean(training_losses),
            "accuracy": np.mean(training_accuracies),
            "precision": np.mean(training_precisions),
            "recall": np.mean(training_recalls),
        }

        model_to_train.eval()
        with torch.no_grad():
            for data, labels in tqdm(test_dl, "Testing"):
                tst_batch_loss = test_batch(data, labels, model_to_train, criterion)
                test_losses.append(tst_batch_loss.item())
                acc, prec, rec = accuracy_precision_recall(data, labels, model_to_train)
                test_accuracies.append(acc)
                test_precisions.append(prec)
                test_recalls.append(rec)
            test_per_epoch_metrics = {
                "loss": np.mean(test_losses),
                "accuracy": np.mean(test_accuracies),
                "precision": np.mean(test_precisions),
                "recall": np.mean(test_recalls),
            }

        training_loss.append(training_per_epoch_metrics["loss"])
        training_metrics.append(training_per_epoch_metrics)
        test_loss.append(test_per_epoch_metrics["loss"])
        test_metrics.append(test_per_epoch_metrics)

        print(f"Epoch: {epoch+1}/{n_epochs}\t| Training loss: {training_per_epoch_metrics['loss']:.4f} | ", end="")
        print(
            f"Training accuracy: {training_per_epoch_metrics['accuracy']:.4f} | Precision: {training_per_epoch_metrics['precision']:.4f} | Recall: {training_per_epoch_metrics['recall']:.4f} | Test loss: {test_per_epoch_metrics['loss']:.4f} | ",
            end="",
        )
        print(
            f"Test accuracy: {test_per_epoch_metrics['accuracy']:.4f} | Precision: {test_per_epoch_metrics['precision']:.4f} | Recall: {test_per_epoch_metrics['recall']:.4f}"
        )

    return model_to_train, training_loss, training_metrics, test_loss, test_metrics


def plot_training_test(training_metrics, test_metrics, n_epochs=20):
    N = np.arange(n_epochs) + 1
    fig = plt.figure(figsize=(16, 8))

    # Subplot for the loss curve
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(N, [x["loss"] for x in training_metrics], "r-s", linewidth=3, label="Training loss")
    ax1.plot(N, [x["loss"] for x in test_metrics], "b-p", linewidth=3, label="Test loss")
    ax1.set_title("Loss Curve", fontsize=15)
    ax1.set_xlabel("No. of epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend()

    # Subplot for the accuracy curve
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(N, [x["accuracy"] for x in training_metrics], "g-D", linewidth=3, label="Training accuracy")
    ax2.plot(N, [x["accuracy"] for x in test_metrics], "c-P", linewidth=3, label="Test accuracy")
    ax2.set_title("Accuracy Curve", fontsize=15)
    ax2.set_xlabel("No. of epochs", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend()

    # Subplot for the precision curve
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(N, [x["precision"] for x in training_metrics], "m-^", linewidth=3, label="Training precision")
    ax3.plot(N, [x["precision"] for x in test_metrics], "y-o", linewidth=3, label="Test precision")
    ax3.set_title("Precision Curve", fontsize=15)
    ax3.set_xlabel("No. of epochs", fontsize=12)
    ax3.set_ylabel("Precision", fontsize=12)
    ax3.legend()

    # Subplot for the recall curve
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(N, [x["recall"] for x in training_metrics], "k-x", linewidth=3, label="Training recall")
    ax4.plot(N, [x["recall"] for x in test_metrics], "orange", linewidth=3, label="Test recall")
    ax4.set_title("Recall Curve", fontsize=15)
    ax4.set_xlabel("No. of epochs", fontsize=12)
    ax4.set_ylabel("Recall", fontsize=12)
    ax4.legend()

    plt.tight_layout()
    plt.show()
