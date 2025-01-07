""" Contrastive Learning for Grid Embeddings """

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class GridDataset(Dataset):
    """Torch Dataset to create dataloaders"""

    def __init__(self, df):
        columns = [c for c in df.columns if c != "label"]
        self.data = df.loc[:, columns].values.astype(np.float32)
        self.data /= self.data.max(0)
        self.data = torch.tensor(self.data)
        self.labels = df.loc[:, "label"].values

        self.label_mapping = {
            label: i for i, label in enumerate(np.unique(self.labels))
        }
        self.labels = torch.tensor([self.label_mapping[label] for label in self.labels])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ContrastiveLoss(nn.Module):
    """Original Contrastive Loss using euclidean distance"""

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, labels):
        """Compute the contrastive loss"""
        label = (labels[None, :] == labels[:, None]).float().flatten()

        # Compute Euclidean distance
        euclidean_distance = nn.functional.pairwise_distance(
            output[None, :, :], output[:, None, :]
        ).flatten()
        # Loss for positive pairs
        positive_loss = label * torch.pow(euclidean_distance, 2)
        # Loss for negative pairs
        negative_loss = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        # Combine losses
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""

    def __init__(self, temperature=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, output, labels):
        """Compute the supervised contrastive loss"""
        label = (labels[None, :] == labels[:, None]).float()  # <batch_size, batch_size>
        label -= torch.eye(label.size(0))

        # Compute cosine similarity
        similarity = nn.functional.cosine_similarity(
            output[None, :, :], output[:, None, :], dim=-1
        )  # <batch_size, batch_size>
        dividend = torch.exp(
            label * similarity / self.temperature
        )  # for each sample keep similarity with positive samples
        divisor = torch.exp(
            (similarity - torch.eye(similarity.size(0))) / self.temperature
        ).sum(
            1
        )  # for each sample sum similarity with all samples

        loss = -torch.log(dividend / divisor).sum() / label.sum()

        return loss


class Encoder(nn.Module):
    """Encoder Network"""

    def __init__(self, input_dim, embedding_dim, layers=(128, 128, 128)):
        super(Encoder, self).__init__()

        if len(layers) == 0:
            self.network = nn.Sequential(nn.Linear(input_dim, embedding_dim))
        elif len(layers) == 1:
            self.network = nn.Sequential(
                nn.Linear(input_dim, layers[0]),
                nn.ReLU(),
                nn.Linear(layers[0], embedding_dim),
            )
        else:
            modules = [nn.Linear(input_dim, layers[0]), nn.ReLU()]
            for i in range(1, len(layers)):
                modules.append(nn.Linear(layers[i - 1], layers[i]))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(layers[-1], embedding_dim))
            self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


def tsne_analysis(data_loader, encoder, label_mapping):
    """Visualize t-SNE embeddings"""

    tsne = TSNE(n_components=2, random_state=42)

    with torch.no_grad():
        embeddings = torch.tensor([])
        labels = []
        for batch_data, batch_labels in data_loader:
            embeddings = torch.cat((embeddings, encoder(batch_data)))
            labels += batch_labels
        labels = torch.tensor(labels)

    embeddings_2d = tsne.fit_transform(embeddings.detach().numpy())

    # Plot the t-SNE results
    fig, ax = plt.subplots(figsize=(10, 10))
    for l in labels.unique():
        ax.scatter(
            embeddings_2d[labels == l, 0],
            embeddings_2d[labels == l, 1],
            label=list(label_mapping)[l],
            alpha=0.75,
        )
    ax.set_title("t-SNE on Embeddings")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend()

    plt.show()


def confusion_matrix(predictions, labels, label_mapping):
    """Confusion Matrix Heatmap of the SVM predictions"""

    conf_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))
    for i, j in zip(labels.numpy(), predictions):
        conf_matrix[i, j] += 1

    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix / conf_matrix.sum(), cmap="YlOrRd")
    fig.colorbar(cax)
    plt.xticks(ticks=np.arange(len(label_mapping)), labels=label_mapping, rotation=90)
    plt.yticks(ticks=np.arange(len(label_mapping)), labels=label_mapping)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix Heatmap")
    plt.show()


def similarities(embeddings, labels, label_mapping):
    """City similarity analysis"""

    emb_df = pd.DataFrame(embeddings)
    emb_df["label"] = labels
    avg_emb = emb_df.groupby("label").mean()

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(avg_emb)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="coolwarm", alpha=0.8)
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(avg_emb.index)), labels=label_mapping, rotation=90)
    plt.yticks(ticks=np.arange(len(avg_emb.index)), labels=label_mapping)
    plt.title("Pairwise Similarities of average embeddings")
    plt.show()


def train_encoder(
    train_loader,
    val_loader,
    loss_fn,
    input_dim,
    embedding_dim=32,
    encoder_layers=(128, 128, 128),
    n_epochs=200,
    learning_rate=1e-3,
    verbose=True,
):
    """Train the encoder network"""

    encoder = Encoder(
        input_dim=input_dim, embedding_dim=embedding_dim, layers=encoder_layers
    )

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    if verbose:
        writer = SummaryWriter()
        pbar = tqdm(total=n_epochs)
    try:
        for epoch in range(n_epochs):
            encoder.train()
            epoch_loss = 0.0

            for batch_data, batch_labels in train_loader:

                output = encoder(batch_data)

                # Compute loss
                loss = loss_fn(output, batch_labels)
                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose:
                # val loss
                encoder.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_data, batch_labels in val_loader:
                        output = encoder(batch_data)
                        val_loss += loss_fn(output, batch_labels).item()
                writer.add_scalar(
                    "encoder/Loss/train", epoch_loss / len(train_loader), epoch
                )
                writer.add_scalar("encoder/Loss/val", val_loss / len(val_loader), epoch)
                pbar.update(1)
                pbar.set_description(f"Loss: {round(epoch_loss / len(train_loader),3)}")
    finally:
        if verbose:
            pbar.close()
            writer.close()

    return encoder


def parameter_search(train_data, val_data, loss_fn, input_dim):
    """Hyperparameter search"""
    batch_sizes = [64, 128, 256]
    embedding_dims = [16, 32, 64]
    network_structures = [(128, 128, 128), (128, 128), (256, 128, 64), (128, 64), (256, 128)]
    n_epochs = 500
    learning_rates = [1e-2, 1e-3, 1e-4]

    def eval_params(batch_size, embedding_dim, encoder_layers, learning_rate):
        print(
            f"Training with batch_size={batch_size}, embedding_dim={embedding_dim}, encoder_layers={encoder_layers}, learning_rate={learning_rate}"
        )
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        encoder = train_encoder(
            train_loader,
            val_loader,
            loss_fn,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            encoder_layers=encoder_layers,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            verbose=False,
        )

        with torch.no_grad():
            train_embeddings = torch.tensor([])
            train_labels = torch.tensor([])
            val_embeddings = torch.tensor([])
            val_labels = torch.tensor([])
            for batch_data, batch_labels in train_loader:
                train_embeddings = torch.cat(
                    (train_embeddings, encoder(batch_data))
                )
                train_labels = torch.cat((train_labels, batch_labels))
            for batch_data, batch_labels in val_loader:
                val_embeddings = torch.cat((val_embeddings, encoder(batch_data)))
                val_labels = torch.cat((val_labels, batch_labels))

        # SVM classifier
        svm_classifier = LinearSVC(class_weight="balanced")
        svm_classifier.fit(train_embeddings.detach().numpy(), train_labels.numpy())

        predictions = svm_classifier.predict(val_embeddings.detach().numpy())
        score = f1_score(val_labels.numpy(), predictions, average="macro")

        return {
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "encoder_layers": encoder_layers,
            "learning_rate": learning_rate,
            "score": score,
        }

    results = Parallel(n_jobs=-1)(
        delayed(eval_params)(batch_size, embedding_dim, encoder_layers, learning_rate)
        for batch_size in batch_sizes
        for embedding_dim in embedding_dims
        for encoder_layers in network_structures
        for learning_rate in learning_rates
    )
    results = pd.DataFrame(results).sort_values("score", ascending=False)
    print(results)


def main(categories, square_size, check_ins, neighbors, analysis, optimize):
    """Main function"""

    batch_size = 128
    embedding_dim = 32
    encoder_layers = (128, 128, 128)
    n_epochs = 500
    temperature = 1.0
    learning_rate = 1e-3

    # loss_fn = ContrastiveLoss(margin=1.0)
    loss_fn = SupConLoss(temperature=temperature)

    # Load data
    data = pd.read_csv(
        f"data/squares_{categories}_cats_{square_size}m{'_neighbors' if neighbors else ''}{'_checkins' if check_ins else ''}{'_'+cities if cities != 'DEFAULT' else ''}.csv"
    )
    data = data.loc[:, data.columns[6:]]

    dataset = GridDataset(data)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [0.7, 0.15, 0.15]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if optimize:
        parameter_search(train_data, val_data, loss_fn, input_dim=len(data.columns) - 1)
        return

    # train the encoder
    encoder = train_encoder(
        train_loader,
        val_loader,
        loss_fn,
        input_dim=len(data.columns) - 1,
        embedding_dim=embedding_dim,
        encoder_layers=encoder_layers,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
    )

    with torch.no_grad():
        embeddings = torch.tensor([])
        labels = []
        for batch_data, batch_labels in test_loader:
            embeddings = torch.cat((embeddings, encoder(batch_data)))
            labels += batch_labels
        labels = torch.tensor(labels)

    # SVM classifier
    svm_classifier = LinearSVC(class_weight="balanced")
    svm_classifier.fit(embeddings.detach().numpy(), labels.numpy())
    predictions = svm_classifier.predict(embeddings.detach().numpy())

    print(classification_report(labels.numpy(), predictions))
    print(f1_score(labels.numpy(), predictions, average="macro"))

    if analysis:
        label_mapping = full_loader.dataset.label_mapping.keys()
        tsne_analysis(test_loader, encoder, label_mapping)
        confusion_matrix(predictions, labels, label_mapping)
        similarities(embeddings, labels, label_mapping)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Contrastive Learning for Grid Embeddings")
    parser.add_argument("--categories", "-c", type=str, default="handpicked", help="Categories for the dataset")
    parser.add_argument("--square_size", "-s", type=int, default=500, help="Square size for the dataset")
    parser.add_argument("--check_ins", "-ci", action="store_true", help="Include check-ins in the dataset")
    parser.add_argument("--cities", "-cty", type=str, default="DEFAULT", help="Cities for the dataset")
    parser.add_argument("--neighbors", "-n", action="store_true", help="Include neighbors in the dataset")
    parser.add_argument("--analysis", "-a", action="store_true", help="Perform analysis after training")
    parser.add_argument("--optimize", "-o", action="store_true", help="Perform hyperparameter optimization")
    args = parser.parse_args()

    main(args.categories, args.square_size, args.check_ins, args.cities, args.neighbors, args.analysis, args.optimize)
