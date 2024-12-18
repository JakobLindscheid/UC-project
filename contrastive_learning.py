import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

CATEGORIES = "lvl2"
SQUARE_SIZE = 1000
CHECK_INS = True
NEIGHBORS = False

ANALYSIS = True

""" Torch Dataset to create dataloaders"""
class GridDataset(Dataset):
    def __init__(self, df):
        columns = [c for c in df.columns if c != "label"]
        self.data = df.loc[:, columns].values.astype(np.float32)
        self.data /= self.data.max(0)
        self.data = torch.tensor(self.data)
        self.labels = df.loc[:, "label"].values

        self.label_mapping = {label: i for i, label in enumerate(np.unique(self.labels))}
        self.labels = np.array([self.label_mapping[label] for label in self.labels])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

""" Original Contrastive Loss using euclidean distance """
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, labels):
        label = (labels[None, :] == labels[:, None]).float().flatten()

        # Compute Euclidean distance
        euclidean_distance = nn.functional.pairwise_distance(output[None,:,:], output[:,None,:]).flatten()
        # Loss for positive pairs
        positive_loss = label * torch.pow(euclidean_distance, 2)
        # Loss for negative pairs
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        # Combine losses
        loss = torch.mean(positive_loss + negative_loss)
        return loss

""" Supervised Contrastive Loss"""
class SupConLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, output, labels):
        label = (labels[None, :] == labels[:, None]).float() # <batch_size, batch_size>
        label -= torch.eye(label.size(0))

        # Compute cosine similarity
        similarity = nn.functional.cosine_similarity(output[None,:,:], output[:,None,:], dim=-1) # <batch_size, batch_size>
        dividend = torch.exp(label * similarity / self.temperature) # for each sample keep similarity with positive samples
        divisor = torch.exp((similarity - torch.eye(similarity.size(0))) / self.temperature).sum(1) # for each sample sum similarity with all samples

        loss = -torch.log(dividend / divisor).sum() / label.sum()

        return loss

""" Encoder Network"""
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, layers=(128,128,128)):
        super(Encoder, self).__init__()

        if len(layers) == 0:
            self.network = nn.Sequential(
                nn.Linear(input_dim, embedding_dim)
            )
        elif len(layers) == 1:
            self.network = nn.Sequential(
                nn.Linear(input_dim, layers[0]),
                nn.ReLU(),
                nn.Linear(layers[0], embedding_dim)
            )
        else:
            modules = [nn.Linear(input_dim, layers[0]), nn.ReLU()]
            for i in range(1, len(layers)):
                modules.append(nn.Linear(layers[i-1], layers[i]))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(layers[-1], embedding_dim))
            self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)

""" Visualize t-SNE embeddings"""
def tsne_analysis(data_loader, encoder, label_mapping):
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
        ax.scatter(embeddings_2d[labels == l, 0], embeddings_2d[labels == l, 1], label=list(label_mapping)[l], alpha=0.75)
    ax.set_title('t-SNE on Embeddings')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend()

    plt.show()

""" Confusion Matrix Heatmap of the SVM predictions"""
def confusion_matrix(predictions, labels, label_mapping):
    conf_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))
    for i, j in zip(labels.numpy(), predictions):
        conf_matrix[i, j] += 1

    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix/conf_matrix.sum(), cmap='YlOrRd')
    fig.colorbar(cax)
    plt.xticks(ticks=np.arange(4), labels=label_mapping)
    plt.yticks(ticks=np.arange(4), labels=label_mapping)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix Heatmap')
    plt.show()

""" City similarity analysis"""
def similarities(embeddings, labels, label_mapping):
    emb_df = pd.DataFrame(embeddings)
    emb_df["label"] = labels
    avg_emb = emb_df.groupby("label").mean()

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(avg_emb)
    # Compute the Euclidean distance matrix
    # similarity_matrix = 1-euclidean_distances(avg_emb)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', alpha=0.8)
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(avg_emb.index)), labels=label_mapping)
    plt.yticks(ticks=np.arange(len(avg_emb.index)), labels=label_mapping)
    plt.title('Pairwise Similarities of average embeddings')
    plt.show()

def train_encoder(train_loader, val_loader, loss_fn, input_dim, embedding_dim=32, encoder_layers=(128, 128, 128), n_epochs=500, learning_rate=1e-3):
    
    encoder = Encoder(input_dim=input_dim, embedding_dim=embedding_dim, layers=encoder_layers)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

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

            # val loss
            encoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    output = encoder(batch_data)
                    val_loss += loss_fn(output, batch_labels).item()
            writer.add_scalar("encoder/Loss/train", epoch_loss / len(train_loader), epoch)
            writer.add_scalar("encoder/Loss/val", val_loss / len(val_loader), epoch)
            pbar.update(1)
            pbar.set_description(f"Loss: {round(epoch_loss / len(train_loader),3)}")
    finally:
        pbar.close()
        writer.close()

    return encoder

def main():

    batch_size = 128
    embedding_dim = 32
    encoder_layers = (128, 128, 128)
    n_epochs = 500
    temperature = 1.0
    learning_rate = 1e-3
        
    # loss_fn = ContrastiveLoss(margin=1.0)
    loss_fn = SupConLoss(temperature=temperature)

    # Load data
    data = pd.read_csv(f"squares_{CATEGORIES}_cats_{SQUARE_SIZE}m{'_neighbors' if NEIGHBORS else ''}{'_checkins' if CHECK_INS else ''}.csv")
    data = data.loc[:, data.columns[6:]]

    dataset = GridDataset(data)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    

    # train the encoder
    encoder = train_encoder(
        train_loader, 
        val_loader,
        loss_fn, 
        input_dim=len(data.columns)-1,
        embedding_dim=embedding_dim,
        encoder_layers=encoder_layers,
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )

    with torch.no_grad():
        embeddings = torch.tensor([])
        labels = []
        for batch_data, batch_labels in test_loader:
            embeddings = torch.cat((embeddings, encoder(batch_data)))
            labels += batch_labels
        labels = torch.tensor(labels)

    # SVM classifier
    svm_classifier = LinearSVC(class_weight='balanced')
    svm_classifier.fit(embeddings.detach().numpy(), labels.numpy())
    predictions = svm_classifier.predict(embeddings.detach().numpy())

    print(classification_report(labels.numpy(), predictions))
    print(f1_score(labels.numpy(), predictions, average='macro'))

    if ANALYSIS:
        label_mapping = full_loader.dataset.label_mapping.keys()
        tsne_analysis(test_loader, encoder, label_mapping)
        confusion_matrix(predictions, labels, label_mapping)
        similarities(embeddings, labels, label_mapping)

if __name__ == "__main__":
    main()

    