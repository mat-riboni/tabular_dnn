from __future__ import annotations
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
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
    """Nodo con early-exit routing, che unifica fit/evaluate su tuple (x_num,x_cat)."""

    def __init__(
        self,
        in_dim:    int,                   # dimensione dell'input flat
        hidden:    int,                   # unità del layer nascosto
        n_classes: int,                   # numero di classi locali
        tau:       dict[int, float],      # soglie per early-exit
        embedder:  Optional[RawEmbedder] = None,  
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.dense    = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        self.head     = nn.Linear(hidden, n_classes)
        self.routing_children = nn.ModuleDict()
        self.tau      = tau

    def add_child(self, class_idx: int, child: Node) -> None:
        self.routing_children[str(class_idx)] = child

    def freeze(self, include_embedder: bool = False) -> None:
        for p in self.dense.parameters():   p.requires_grad = False
        for p in self.head.parameters():    p.requires_grad = False
        if include_embedder and self.embedder:
            for p in self.embedder.parameters(): p.requires_grad = False

    def _flat(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.embedder:
            return self.embedder(x_num, x_cat)
        # se non ho embedder, assumo x_num sia già flat
        return x_num

    @torch.no_grad()
    def route(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        h_all      = self.dense(x_flat)
        logits_all = self.head(h_all)
        probs_all  = logits_all.softmax(dim=1)

        sorted_idxs = torch.argsort(probs_all, dim=1, descending=True)
        out_logits, out_h, depths = [], [], []

        for i in range(x_flat.size(0)):
            h_i      = h_all[i : i+1]
            logits_i = logits_all[i : i+1]

            for class_idx in sorted_idxs[i]:
                idx = str(int(class_idx))
                conf = float(probs_all[i, class_idx])
                if idx in self.routing_children and conf >= self.tau.get(int(idx), 1.1):
                    l_c, h_c, d_c = self.routing_children[idx].route(h_i)
                    out_logits.append(l_c)
                    out_h.append(h_c)
                    depths.append(d_c[0] + 1)
                    break
            else:
                out_logits.append(logits_i)
                out_h.append(h_i)
                depths.append(0)

        return (
            torch.cat(out_logits, dim=0),
            torch.cat(out_h,      dim=0),
            depths
        )

    def forward(
        self,
        inputs: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        train: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if isinstance(inputs, tuple):
            x_num, x_cat = inputs
            x_flat = self._flat(x_num, x_cat)
        else:
            x_flat = inputs

        if train or not self.routing_children:
            h = self.dense(x_flat)
            bs = x_flat.size(0)
            return self.head(h), h, [0] * bs
        return self.route(x_flat)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs:  int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn:   Optional[nn.Module]           = None,
        device:    str = "cuda",
        lr:        float = 1e-3
    ) -> None:
        print("🚀 ENTERED fit(), epochs=", epochs)
        self.to(device).train()
        opt       = optimizer or torch.optim.AdamW(self.parameters(), lr)
        criterion = loss_fn   or nn.CrossEntropyLoss()

        for ep in range(epochs):
            self.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                logits, _, _ = self(x, train=True)

                loss = criterion(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            metrics = self.evaluate(val_loader, loss_fn=criterion, device=device)
            print(
                f"[Node {id(self):x}] ep{ep:02d} | "
                f"val_loss={metrics['loss']:.4f} | "
                f"val_acc={metrics['acc']:.3f} | "
                f"val_f1={metrics['macro_f1']:.3f}"
            )
        self.eval()

    @torch.no_grad()
    def evaluate(self, loader, loss_fn=None, device="cuda") -> dict:
        self.to(device).eval()
        crit = loss_fn or nn.CrossEntropyLoss(reduction="sum")

        total=correct=0
        loss_sum=0.0
        y_true, y_pred = [], []

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _, _ = self(x, train=True)

            loss_sum += crit(logits, y).item()
            preds    = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            y_true  += y.cpu().tolist()
            y_pred  += preds.cpu().tolist()

        return {
            "loss":     loss_sum / total,
            "acc":      correct / total,
            "macro_f1": f1_score(y_true, y_pred, average="macro")
        }