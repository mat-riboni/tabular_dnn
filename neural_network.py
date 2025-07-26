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


    def fit(self, train_dataloader, valid_dataloader, device, epochs=20, loss_fn=nn.CrossEntropyLoss(), lr=1e-3 ,optimizer=None, lr_scheduler=None):
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_f1, best_loss = -float('inf'), float('inf')

        for epoch in range(epochs):
            self.train()

            for x_num, x_cat, y in train_dataloader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                
                preds = self(x_num, x_cat)
                loss = loss_fn(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
            f1, acc = self.evaluate(valid_dataloader, device, loss_fn)
            loss_val = loss.item()

            improved = (f1 > best_f1) or (f1 == best_f1 and loss_val < best_loss)
            acceptable = f1 > 0.9 and loss_val < 0.1

            if improved:
                best_f1 = f1
                best_loss = loss_val

            if improved and acceptable:
                torch.save({'epoch': epoch,
                    'model_state': self.state_dict()},
                    'best_model.pt')

            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(f1)
            else:
                    lr_scheduler.step() 

            print(f"--- Epoch: {epoch}  |  Loss: {loss_val:.4f}  |  F1 Score: {f1:.4f}  |  Accuracy: {acc:.4f} ---")


    def evaluate(self, dataloader, device, loss_fn):

        self.eval()
        
        all_preds = []
        all_labels = []

        running_loss, n = 0, 0

        with torch.no_grad():
            for x_num, x_cat, y in dataloader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                outputs = self(x_num, x_cat)
                preds = outputs.argmax(dim=1)
                batch_size = y.size(0)

                running_loss += loss_fn(outputs, y).item() * batch_size
                n += batch_size


                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        loss_val = running_loss / n
        f1  = f1_score(torch.cat(all_labels), torch.cat(all_preds), average='macro')
        acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))

        return f1, acc, loss_val

    
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

