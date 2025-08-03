import torch.nn as nn
import torch
from sklearn.metrics import f1_score, accuracy_score



class MaskGenerator(nn.Module):
    def __init__(self, cat_cardinalities, embedding_dims, num_numerical_features, mask_sizes, share_embeddings=None):
        super().__init__()

        if share_embeddings is not None:
            self.embeddings = share_embeddings
            self.shared_embeddings = True
        else:
            self.embeddings = nn.ModuleList([
                nn.Embedding(cat_cardinality, embedding_dim)
                for cat_cardinality, embedding_dim in zip(cat_cardinalities, embedding_dims)
            ])
            self.shared_embeddings = False

        input_dim = sum(embedding_dims) + num_numerical_features
        self.block = 4
        self.block_sizes = [s // self.block for s in mask_sizes]
        self.controller = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, sum(self.block_sizes)),
        )

        self.config = {
            'cat_cardinalities': cat_cardinalities,
            'embedding_dims': embedding_dims,
            'num_numerical_features': num_numerical_features,
            'mask_sizes': mask_sizes,
            'shared_embeddings': self.shared_embeddings
        }

        self.mask_sizes = mask_sizes

    def _embed_input(self, x_num, x_cat):
        """Metodo helper per processare gli input (coerente con NeuralNetwork)"""
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embedded, dim=1)
        x = torch.cat([x_num, x_cat], dim=1)
        return x

    def forward(self, x_num, x_cat):
        x = self._embed_input(x_num, x_cat)          

        logits_reduced = self.controller(x)       

        masks = []
        start = 0
        for orig, blocks in zip(self.mask_sizes, self.block_sizes):
            end = start + blocks
            layer_mask = logits_reduced[:, start:end]\
                            .repeat_interleave(self.block, dim=1)[:, :orig]
            masks.append(torch.sigmoid(layer_mask)) 
            start = end

        return masks  

    def fit(self, model, train_dataloader, valid_dataloader, device, threshold, 
            alpha=1e-4, epochs=20, lr=1e-3, optimizer=None, lr_scheduler=None, 
            warmup_epochs=5):
        
        self.to(device)
        model.to(device)
        model.eval()  

        for p in model.parameters():
            p.requires_grad = False

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)

        loss_fn = nn.CrossEntropyLoss()
        best_f1, best_loss, best_sparsity = -float('inf'), float('inf'), -float('inf')
        
        # Scheduling dinamico di alpha per incoraggiare sparsità gradualmente
        initial_alpha = alpha

        for epoch in range(epochs):
            self.train()
            
            # Alpha scheduling: aumenta gradualmente dopo warmup
            if epoch < warmup_epochs:
                current_alpha = initial_alpha * 0.1  # Meno penalizzazione all'inizio
            else:
                current_alpha = initial_alpha * (1 + (epoch - warmup_epochs) * 0.1)

            epoch_loss = 0
            epoch_sparsity = 0
            num_batches = 0

            for x_num, x_cat, y in train_dataloader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                
                optimizer.zero_grad()
                masks = self(x_num, x_cat)
                
                preds = model.forward_with_masks(x_num, x_cat, masks)

                loss, sparsity = self._compute_masked_loss(preds, y, loss_fn, masks, current_alpha, threshold)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_sparsity += sparsity
                num_batches += 1

            f1, acc, valid_loss, valid_sparsity = self.evaluate(model, loss_fn, current_alpha, 
                                                              valid_dataloader, device, threshold)

            improved = (f1 > best_f1 - 0.05) and valid_sparsity > best_sparsity 
            accettable = best_sparsity > 0.5

            if improved:
                best_f1 = f1
                best_loss = valid_loss
                best_sparsity = valid_sparsity
            
            if improved and accettable:
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.state_dict(),
                    'config': self.config,
                    'threshold': threshold
                }, 'best_mask_model.pt')

            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(f1)
            else:
                lr_scheduler.step()
            
            print(f"Epoch {epoch:3d} | Valid Loss: {valid_loss:.4f} | Valid Sparsity: {valid_sparsity:.3f} | F1: {f1:.4f} | Acc: {acc:.4f} | Alpha: {current_alpha:.6f}")

    def _compute_masked_loss(self, outputs, targets, loss_fn, masks, alpha, threshold):
        """Compute loss con regolarizzazione sparsità"""
        base_loss = loss_fn(outputs, targets)
        sparsity_loss = alpha * sum([m.mean() for m in masks])  # Penalizza valori alti (neuroni attivi)
        total_loss = base_loss + sparsity_loss
        
        # Calcola sparsità media (% neuroni disattivati con soglia 0.5)
        avg_sparsity = sum([(m < threshold).float().mean().item() for m in masks]) / len(masks)
        
        return total_loss, avg_sparsity

    def evaluate(self, model, loss_fn, alpha, dataloader, device, threshold):
        self.eval()
        model.eval()
        
        all_preds = []
        all_labels = []
        running_loss, n = 0, 0
        total_sparsity = 0
        num_batches = 0

        with torch.no_grad():
            for x_num, x_cat, y in dataloader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                
                masks = self(x_num, x_cat)
                outputs = model.forward_with_masks(x_num, x_cat, masks)
                
                loss, sparsity = self._compute_masked_loss(outputs, y, loss_fn, masks, alpha, threshold)
                batch_size = y.size(0)

                running_loss += loss.item() * batch_size
                total_sparsity += sparsity
                n += batch_size
                num_batches += 1

                preds = outputs.argmax(dim=1)  
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        loss_val = running_loss / n
        avg_sparsity = total_sparsity / num_batches
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)

        return f1, acc, loss_val, avg_sparsity

    def analyze_sparsity(self, dataloader, device, threshold=0.5):
        """Analizza pattern di sparsità per diversi input"""
        self.eval()
        sparsity_stats = []
        
        with torch.no_grad():
            for i, (x_num, x_cat, y) in enumerate(dataloader):
                if i >= 10:  # Analizza solo i primi 10 batch
                    break
                    
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                masks = self(x_num, x_cat)
                
                batch_sparsity = []
                for j, mask in enumerate(masks):
                    layer_sparsity = (mask < threshold).float().mean().item()
                    batch_sparsity.append(layer_sparsity)
                
                sparsity_stats.append(batch_sparsity)
        
        return sparsity_stats

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config 
        }, path)

    @classmethod
    def load(cls, path, device='cuda'):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA non disponibile. Caricamento su CPU.")
            device = 'cpu'

        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model


def benchmark_inference_speed(model, mask_generator, dataloader, device, threshold, num_runs=5):
    """Confronta velocità di inferenza con e senza maschere"""
    import time
    
    model.eval()
    mask_generator.eval()
    
    # Warm up
    with torch.no_grad():
        for x_num, x_cat, _ in dataloader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            _ = model(x_num, x_cat)
            break
    
    times_normal = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            for x_num, x_cat, _ in dataloader:
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                _ = model(x_num, x_cat)
        times_normal.append(time.time() - start_time)
    
    times_masked = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            for x_num, x_cat, _ in dataloader:
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                masks = mask_generator(x_num, x_cat)
                binary_masks = [(m > threshold).float() for m in masks]
                _ = model.forward_with_masks(x_num, x_cat, binary_masks)
        times_masked.append(time.time() - start_time)
    
    avg_normal = sum(times_normal) / len(times_normal)
    avg_masked = sum(times_masked) / len(times_masked)
    
    print(f"Inference normale:     {avg_normal:.4f}s")
    print(f"Inference con maschere: {avg_masked:.4f}s")
    print(f"Overhead relativo:     {((avg_masked/avg_normal - 1) * 100):.1f}%")
    
    return avg_normal, avg_masked