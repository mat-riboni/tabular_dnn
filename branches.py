from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

 
class RawEmbedder(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        cat_cardinalities: List[int],
        embed_dims: List[int],
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.emb_layers = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, embed_dims)
        ])
        self.out_dim = numeric_dim + sum(embed_dims)
        self._d = nn.Dropout(dropout)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        cat_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        return self._d(torch.cat([x_num] + cat_emb, dim=1))
    

class Node(nn.Module):
    """Un nodo con routing basato sulla **predizione** (non sulla label).

    Args
    ----
    in_dim      : dim vettore input (embedding precedente)
    hidden      : unità del Linear locale
    n_classes   : # classi che questo nodo distingue
    tau         : confidenza minima per *early‑exit*.
    """

    def __init__(self, in_dim: int, hidden: int, n_classes: int, tau: List[int]):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_classes)
        self.children = nn.ModuleDict()  
        self.tau = tau
        self.hidden_dim = hidden

    # ---------- building ----------
    def add_child(self, class_idx: int, child: Node):
        self.children[str(class_idx)] = child

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    # ---------- forward ----------
    @torch.no_grad()
    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Routing multilivello con soglie specifiche per classe.
        Ritorna (logits_finali, embedding_finale, depth)."""
        h = self.dense(x)
        logits = self.head(h)
        probs  = logits.softmax(1)

        # Ordina le classi per confidenza decrescente
        sorted_classes = torch.argsort(probs, descending=True)[0]

        for class_idx in sorted_classes:
            idx = str(class_idx.item())
            conf = probs[0, class_idx].item()
            if idx in self.children and conf >= self.tau.get(int(idx), 1.1):
                return self.children[idx].route(h)

        # Nessuna classe supera la soglia o non ha figlio: ritorna i logits attuali
        return logits, h, 0 

    def forward(self, x: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Training/inference unificato.
        • train=True  → disattiva routing (resta qui)
        • train=False → chiama .route()"""
        if train or not self.children:
            h = self.dense(x)
            return self.head(h), h, 0
        else:
            return self.route(x)
        
    def fit(self, train_loader, val_loader, epochs,
            optimizer: Optional[torch.optim.Optimizer] = None, 
            loss_fn: Optional[nn.Module] = None,
            device: str = "cuda", lr=1e-3) -> None:
        """Train this node.

        Parameters
        ----------
        train_loader : DataLoader yielding (x_in, y_local) for *training* samples
        val_loader   : optional DataLoader yielding (x_in, y_local) for validation;
                       if provided, loss / accuracy / macro‑F1 are logged each epoch.
        optimizer    : pre‑created optimizer (default AdamW)
        criterion    : loss function (default CrossEntropy)
        epochs       : number of epochs
        device       : cuda / cpu
        """

        self.to(device).train()
        opt  = optimizer  or torch.optim.AdamW(self.parameters(), lr)
        loss_fn = loss_fn  or nn.CrossEntropyLoss()

        for ep in range(epochs):
            # ----- training phase -----
            self.train()
            tot, correct, loss_sum = 0, 0, 0.0
            for x_in, y_local in train_loader:
                x_in, y_local = x_in.to(device), y_local.to(device)

                logits, _, _ = self(x_in, train=True)
                loss = loss_fn(logits, y_local)

                opt.zero_grad()
                loss.backward()
                opt.step()

                        # ----- validation phase -----
            metrics = self.evaluate(val_loader, loss_fn=loss_fn, device=device)        
            self.eval()
            print(f"[Node {id(self):x}] ep{ep:02d} | val_loss={metrics['loss']:.4f} | val_acc={metrics['acc']:.3f} val_f1={metrics['macro_f1']:.3f}")

    # ---------- evaluate ----------
    @torch.no_grad()
    def evaluate(self, loader, loss_fn: Optional[nn.Module] = None,device: str = "cuda"):
        """Return loss, accuracy, macro‑F1 on `loader`. Routing disabled."""
        self.to(device).eval()
        
        crit = loss_fn or nn.CrossEntropyLoss(reduction="sum")
        total, correct, loss_sum = 0, 0, 0.0
        y_true, y_pred = [], []
        for x_in, y_local in loader:
            x_in, y_local = x_in.to(device), y_local.to(device)
            logits, _, _ = self(x_in, train=True)            # routing OFF
            loss_sum += crit(logits, y_local).item()
            preds = logits.argmax(1)
            correct += (preds==y_local).sum().item()
            total   += x_in.size(0)
            y_true += y_local.cpu().tolist()
            y_pred += preds.cpu().tolist()
        acc = correct/total
        f1  = f1_score(y_true, y_pred, average="macro")
        return {"loss": loss_sum/total, "acc": acc, "macro_f1": f1}
