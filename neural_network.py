from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, accuracy_score

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx] 
    

class NeuralNetwork(nn.Module):


    
    def __init__(self, hidden_layers_sizes, cat_cardinalities, embedding_dims, num_numerical_features, num_target_classes):
        super().__init__()


        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_cardinality, embedding_dim)
            for cat_cardinality, embedding_dim in zip(cat_cardinalities, embedding_dims)
        ])


        total_embedding_size = sum(embedding_dims)
        input_size = total_embedding_size + num_numerical_features               #  Number of neurons in the input layer
        layers_dims = [input_size] + hidden_layers_sizes + [num_target_classes]  #  Number of neurons in each layer
        layers = []


        for i in range(len(layers_dims) - 1):
            layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))

            # Output layer without ReLU, BatchNorm and Dropout #
            if i < len(layers_dims) - 2:      
                layers.append(nn.BatchNorm1d(layers_dims[i+1]))                 
                layers.append(nn.Dropout(0.2))
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)




    def forward(self, x_num, x_cat):
        embedded_cat = []

        for i, emb in enumerate(self.embeddings):
            column_embedded = emb(x_cat[:,i])                                    #  Gets i-th column in batch and applies embedding
            embedded_cat.append(column_embedded)


        x_cat = torch.cat(embedded_cat, dim=1)
        x = torch.cat([x_num,x_cat], dim=1)

        return self.network(x)


    def fit(self, train_dataloader, valid_dataloader, device, epochs=20, loss_fn=nn.CrossEntropyLoss(), lr=1e-3 ,optimizer=None):
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        for epoch in range(epochs):
            self.network.train()

            for x_num, x_cat, y in train_dataloader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                
                preds = self(x_num, x_cat)
                loss = loss_fn(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            score, acc = self.evaluate(valid_dataloader, device)
            print(f"--- Epoch: {epoch}  |  Loss: {loss.item():.4f}|  F1 Score: {score:.4f}  |  Accuracy: {acc:.4f} ---")


    def evaluate(self, dataloader, device):

        self.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_num, x_cat, y in dataloader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                outputs = self(x_num, x_cat)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        return f1_score(all_labels, all_preds, average='macro'), accuracy_score(all_preds,all_labels)
    

    
    def predict(self, dataloader, device):
        self.eval()
        all_preds = []

        with torch.no_grad():
            for x_num, x_cat, _ in dataloader:  
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                outputs = self(x_num, x_cat)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds)



    def save(self, path, config=None):
        torch.save({
            'state_dict': self.state_dict(),
            'config': config 
        }, path)


    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

