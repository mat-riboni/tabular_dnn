import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score



class DenseBranchyNet(nn.Module):
    """Implementazione semplice di branchynet, sempre e solo 3 layer hidden per semplicità."""

    def __init__(self, hidden_layers_sizes, taus, alphas, cat_cardinalities, embedding_dims, num_numerical, num_target_classes):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_cardinality, embedding_dim)
            for cat_cardinality, embedding_dim in zip(cat_cardinalities, embedding_dims)
        ])

        self.taus = taus
        self.alphas = alphas 

        total_emb_size = sum(embedding_dims)
        input_size = total_emb_size + num_numerical

        self.fc1 = nn.Linear(input_size, hidden_layers_sizes[0])
        self.head1 = nn.Linear(hidden_layers_sizes[0], num_target_classes)

        self.fc2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
        self.head2 = nn.Linear(hidden_layers_sizes[1], num_target_classes)

        self.fc3 = nn.Linear(hidden_layers_sizes[1], hidden_layers_sizes[2])
        self.head3 = nn.Linear(hidden_layers_sizes[2], num_target_classes)



    def forward(self, x_num, x_cat):
        x = self._embed_input(x_num, x_cat)

        h1 = F.relu(self.fc1(x))
        out1 = self.head1(h1)

        h2 = F.relu(self.fc2(h1))
        out2 = self.head2(h2)

        h3 = F.relu(self.fc3(h2))
        out3 = self.head3(h3)
    

        return out1, out2, out3
    
    def inference(self, x_num, x_cat):
        assert x_num.size(0) == 1, "inference() supporta solo batch=1"
        x = self._embed_input(x_num, x_cat)
        h1 = F.relu(self.fc1(x)); p1 = F.softmax(self.head1(h1), dim=1)
        H1 = self._entropy(p1)
        if H1.item() < self.taus[0]:
            return p1
        h2 = F.relu(self.fc2(h1)); p2 = F.softmax(self.head2(h2), dim=1)
        H2 = self._entropy(p2)
        if H2.item() < self.taus[1]:
            return p2
        h3 = F.relu(self.fc3(h2)); return F.softmax(self.head3(h3), dim=1)



    
    def fit(self, train_loader, valid_loader, optimizer, device,
            epochs=10, criterion=None, grad_clip=None, scheduler=None,
            eval_exit_stats=True, verbose=True):
        """
        Training per 'epochs' epoche.
        - A fine epoca esegue evaluate() sul validation set e inserisce in history accuracy e F1.
        Ritorna: list[dict] con metriche per epoca (train loss e val metrics).
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
        )

        history = []
        for ep in range(1, epochs+1):
            self.train()
            running = {"loss": 0.0, "loss1": 0.0, "loss2": 0.0, "loss3": 0.0}
            n_batches = 0

            for x_num, x_cat, y in train_loader:
                x_num = x_num.to(device)
                x_cat = x_cat.to(device).long()
                y     = y.to(device).long()

                optimizer.zero_grad()

                # logits delle tre teste
                out1, out2, out3 = self(x_num, x_cat)

                loss1 = criterion(out1, y)
                loss2 = criterion(out2, y)
                loss3 = criterion(out3, y)
                loss  = self.alphas[0]*loss1 + self.alphas[1]*loss2 + self.alphas[2]*loss3

                loss.backward()
                #if grad_clip is not None:
                #    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

                running["loss"]  += loss.item()
                running["loss1"] += loss1.item()
                running["loss2"] += loss2.item()
                running["loss3"] += loss3.item()
                n_batches += 1

            # medie epoca
            for k in running:
                running[k] /= max(n_batches, 1)


            # valutazione su validation set (accuracy + F1 + exit stats)
            val_metrics = self.evaluate(valid_loader, device, return_exit_stats=eval_exit_stats)
            scheduler.step(metrics=val_metrics['f1_weighted'])

            log = {"epoch": ep, **running,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"]}
            if eval_exit_stats:
                log["val_exit_counts"] = val_metrics["val_exit_counts"]
                log["val_exit_rate"]   = val_metrics["val_exit_rate"]

            history.append(log)

            if verbose:
                print(f"[Ep {ep:03d}] "
                    f"train_loss={running['loss']:.4f} "
                    f"(l1={running['loss1']:.4f} l2={running['loss2']:.4f} l3={running['loss3']:.4f})  "
                    f"| val_acc={log['val_accuracy']:.4f}  val_f1={log['val_f1_weighted']:.4f}")

        return history
    
    
    @torch.no_grad()
    def evaluate(self, dataloader, device, return_exit_stats=True, use_margin=True):
        self.eval()
        all_preds, all_labels = [], []
        exit_counts = {1: 0, 2: 0, 3: 0}

        for x_num, x_cat, y in dataloader:
            x_num = x_num.to(device); x_cat = x_cat.to(device).long(); y = y.to(device).long()

            x  = self._embed_input(x_num, x_cat)
            h1 = F.relu(self.fc1(x)); l1 = self.head1(h1)

            if use_margin:
                conf1 = self._logit_margin(l1)
                exit1_mask = conf1 >= self.taus[0]
            else:
                logp1 = F.log_softmax(l1, 1); p1 = logp1.exp()
                H1 = -(p1 * logp1).sum(1)
                exit1_mask = H1 <= self.taus[0]

            keep_mask = ~exit1_mask
            batch_preds = torch.empty(x_num.size(0), dtype=torch.long, device=device)

            if exit1_mask.any():
                batch_preds[exit1_mask] = l1[exit1_mask].argmax(1)
                exit_counts[1] += int(exit1_mask.sum().item())

            if keep_mask.any():
                idx_keep = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
                h2_sel = F.relu(self.fc2(h1[idx_keep])); l2_sel = self.head2(h2_sel)

                if use_margin:
                    conf2 = self._logit_margin(l2_sel)
                    exit2_local = conf2 >= self.taus[1]
                else:
                    logp2 = F.log_softmax(l2_sel, 1); p2 = logp2.exp()
                    H2 = -(p2 * logp2).sum(1)
                    exit2_local = H2 <= self.taus[1]

                if exit2_local.any():
                    idx_exit2 = idx_keep[exit2_local]
                    batch_preds[idx_exit2] = l2_sel[exit2_local].argmax(1)
                    exit_counts[2] += int(exit2_local.sum().item())

                keep2_local = ~exit2_local
                if keep2_local.any():
                    idx_keep2 = idx_keep[keep2_local]
                    h3_sel = F.relu(self.fc3(h2_sel[keep2_local])); l3_sel = self.head3(h3_sel)
                    batch_preds[idx_keep2] = l3_sel.argmax(1)
                    exit_counts[3] += int(keep2_local.sum().item())

            all_preds.extend(batch_preds.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="weighted")
        out = {"accuracy": acc, "f1_weighted": f1}
        if return_exit_stats:
            total = sum(exit_counts.values()) or 1
            out["val_exit_counts"] = exit_counts
            out["val_exit_rate"]   = {k: v/total for k, v in exit_counts.items()}
        return out

    
    @torch.inference_mode()
    def predict(self, dataloader, device, early_exit=True, use_margin=True):
        self.eval()
        preds_out = []

        for x_num, x_cat, _ in dataloader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True).long()

            x  = self._embed_input(x_num, x_cat)
            h1 = F.relu(self.fc1(x)); l1 = self.head1(h1)
            if early_exit and use_margin:
                conf1 = self._logit_margin(l1)
                exit1 = conf1 >= self.taus[0]
                # se meno del 30% uscirebbe al ramo1, fai fast path (tuning: 0.2–0.4)
                if exit1.float().mean().item() < 0.30:
                    # forward fino in fondo in una volta sola
                    h2 = F.relu(self.fc2(h1)); h3 = F.relu(self.fc3(h2)); l3 = self.head3(h3)
                    preds_out.append(l3.argmax(1).cpu())
                    continue

            B = x.size(0)
            batch_preds = torch.empty(B, dtype=torch.long, device=device)

            if early_exit:
                if use_margin:
                    conf1 = self._logit_margin(l1)
                    exit1 = conf1 >= self.taus[0]      # margine: alto = meglio
                else:
                    logp1 = F.log_softmax(l1, dim=1); p1 = logp1.exp()
                    H1 = -(p1 * logp1).sum(dim=1)
                    exit1 = H1 <= self.taus[0]         # entropia: basso = meglio
            else:
                exit1 = torch.zeros(B, dtype=torch.bool, device=device)

            if exit1.any():
                batch_preds[exit1] = l1[exit1].argmax(dim=1)

            keep1 = ~exit1
            if keep1.any():
                h2 = F.relu(self.fc2(h1[keep1])); l2 = self.head2(h2)

                if early_exit:
                    if use_margin:
                        conf2 = self._logit_margin(l2)
                        exit2 = conf2 >= self.taus[1]
                    else:
                        logp2 = F.log_softmax(l2, dim=1); p2 = logp2.exp()
                        H2 = -(p2 * logp2).sum(dim=1)
                        exit2 = H2 <= self.taus[1]
                else:
                    exit2 = torch.zeros(l2.size(0), dtype=torch.bool, device=device)

                idx1 = keep1.nonzero(as_tuple=False).squeeze(1)
                if exit2.any():
                    batch_preds[idx1[exit2]] = l2[exit2].argmax(dim=1)

                keep2 = ~exit2
                if keep2.any():
                    h3 = F.relu(self.fc3(h2[keep2])); l3 = self.head3(h3)
                    batch_preds[idx1[keep2]] = l3.argmax(dim=1)

            preds_out.append(batch_preds.cpu())

        return torch.cat(preds_out, dim=0)


    @staticmethod
    def _logit_margin(logits):
        top2 = logits.topk(2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    
    def set_stage(self, stage: str):
        """
        stage in {"stage0_trunk_final", "stage1_heads_only", "stage2_finetune_all"}
        - stage0: allena tronco (fc1, fc2, fc3) + head3, congela head1/head2
        - stage1: allena solo head1/head2, congela tronco + head3
        - stage2: allena tutto
        """
        for p in self.parameters():
            p.requires_grad = True

        heads12 = list(self.head1.parameters()) + list(self.head2.parameters())
        head3   = list(self.head3.parameters())
        trunk   = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters())

        if stage == "stage0_trunk_final":
            for p in heads12: p.requires_grad = False
        elif stage == "stage1_heads_only":
            for p in trunk + head3: p.requires_grad = False
        elif stage == "stage2_finetune_all":
            pass
        else:
            raise ValueError("stage non valido")
            

    def _embed_input(self, x_num, x_cat):
        embedded_cat = []
        for i, emb in enumerate(self.embeddings):
            column_embedded = emb(x_cat[:, i])
            embedded_cat.append(column_embedded)

        x_cat = torch.cat(embedded_cat, dim = 1)
        x = torch.cat([x_num, x_cat], dim = 1)
        return x
    
    @staticmethod
    def _entropy(p):
        H = -torch.sum(p * torch.log(p + 1e-12), dim = 1)
        return H
    



    @torch.no_grad()
    def _collect_val_stats(self, loader, device, use_margin=True):
        self.eval()
        conf1_list, conf2_list = [], []
        pred1_list, pred2_list, pred3_list = [], [], []
        y_list = []

        for x_num, x_cat, y in loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True).long()
            y     = y.to(device, non_blocking=True).long()

            # full path: h1->l1, h2->l2, h3->l3
            x   = self._embed_input(x_num, x_cat)
            h1  = F.relu(self.fc1(x)); l1 = self.head1(h1)
            h2  = F.relu(self.fc2(h1)); l2 = self.head2(h2)
            h3  = F.relu(self.fc3(h2)); l3 = self.head3(h3)

            # confidenze (margine logit di default)
            if use_margin:
                conf1 = self._logit_margin(l1)
                conf2 = self._logit_margin(l2)
            else:
                logp1 = F.log_softmax(l1, 1); p1 = logp1.exp()
                conf1 = -(p1 * logp1).sum(1)             # entropia
                logp2 = F.log_softmax(l2, 1); p2 = logp2.exp()
                conf2 = -(p2 * logp2).sum(1)

            conf1_list.append(conf1.detach().cpu())
            conf2_list.append(conf2.detach().cpu())
            pred1_list.append(l1.argmax(1).detach().cpu())
            pred2_list.append(l2.argmax(1).detach().cpu())
            pred3_list.append(l3.argmax(1).detach().cpu())
            y_list.append(y.detach().cpu())

        stats = {
            "conf1": torch.cat(conf1_list),
            "conf2": torch.cat(conf2_list),
            "pred1": torch.cat(pred1_list),
            "pred2": torch.cat(pred2_list),
            "pred3": torch.cat(pred3_list),
            "y":     torch.cat(y_list),
        }
        return stats
    @staticmethod
    def _simulate_exits(stats, t1, t2, use_margin=True):
        conf1, conf2 = stats["conf1"], stats["conf2"]
        pred1, pred2, pred3 = stats["pred1"], stats["pred2"], stats["pred3"]
        y = stats["y"].numpy()

        # regola: se use_margin True → più alto è meglio; se entropia → più basso è meglio
        if use_margin:
            exit1 = conf1 >= t1
            # attenzione: conf2 si valuta SOLO su chi non è uscito al ramo1
            exit2 = (~exit1) & (conf2 >= t2)
        else:
            exit1 = conf1 <= t1
            exit2 = (~exit1) & (conf2 <= t2)

        # costruisci le predizioni finali
        preds = pred3.clone()
        preds[exit2] = pred2[exit2]
        preds[exit1] = pred1[exit1]

        f1 = f1_score(y, preds.numpy(), average="weighted")
        # statistiche di uscita
        r1 = exit1.float().mean().item()
        r2 = exit2.float().mean().item()
        r3 = 1.0 - r1 - r2
        return f1, {"r1": r1, "r2": r2, "r3": r3}
    


    def calibrate_taus(self, loader, device, use_margin=True,
                    n_grid=15, mode="max_f1",
                    f1_min=None, lam=0.0, cost=None):
        """
        mode:
        - "max_f1": massimizza F1
        - "tradeoff": massimizza (F1 - lam * normalized_cost) se passi 'cost'
        - "min_cost_at_f1": minimizza costo con vincolo F1 >= f1_min
        cost: dict opzionale coi costi incrementali per livello, es: {"c1":1.0,"c2":0.6,"c3":0.6}
            (costi relativi: c1 fino a head1, c2 da head1→head2, c3 da head2→head3)
        """
        stats = self._collect_val_stats(loader, device, use_margin=use_margin)

        # grid per τ1: percentili alti se margine, bassi se entropia
        q = torch.linspace(0.50, 0.995, n_grid) if use_margin else torch.linspace(0.005, 0.50, n_grid)
        t1_grid = torch.quantile(stats["conf1"], q)

        best = {"t1": None, "t2": None, "f1": -1.0, "rates": None, "obj": -1e9, "cost": None}

        # baseline cost & normalizzazione
        def expected_cost(rates, cost):
            # E[costo] = r1*c1 + r2*(c1+c2) + r3*(c1+c2+c3)
            c1, c2, c3 = cost["c1"], cost["c2"], cost["c3"]
            return rates["r1"]*c1 + rates["r2"]*(c1+c2) + rates["r3"]*(c1+c2+c3)

        base_norm = None  # per normalizzare il costo rispetto al full forward
        if cost is not None:
            full = expected_cost({"r1":0.0,"r2":0.0,"r3":1.0}, cost)
            base_norm = full if full > 0 else 1.0

        for t1 in t1_grid:
            # condiziona t2 ai non-usciti con t1
            if use_margin:
                keep_mask = stats["conf1"] < t1
            else:
                keep_mask = stats["conf1"] > t1
            conf2_pool = stats["conf2"][keep_mask]
            if conf2_pool.numel() == 0:  # tutti usciti a ramo1
                # simula con qualsiasi t2 (non influisce)
                f1, rates = self._simulate_exits(stats, t1.item(), torch.tensor(0.).item(), use_margin)
                if cost is None:
                    obj = f1
                    cval = None
                else:
                    cval = expected_cost(rates, cost)/base_norm
                    if mode == "max_f1":
                        obj = f1
                    elif mode == "tradeoff":
                        obj = f1 - lam * cval
                    elif mode == "min_cost_at_f1":
                        obj = -cval if (f1_min is not None and f1 >= f1_min) else -1e9
                if obj > best["obj"]:
                    best.update({"t1": t1.item(), "t2": None, "f1": f1, "rates": rates, "obj": obj, "cost": cval})
                continue

            q2 = torch.linspace(0.50, 0.995, n_grid) if use_margin else torch.linspace(0.005, 0.50, n_grid)
            t2_grid = torch.quantile(conf2_pool, q2)

            for t2 in t2_grid:
                f1, rates = self._simulate_exits(stats, t1.item(), t2.item(), use_margin)
                if cost is None:
                    obj = f1
                    cval = None
                else:
                    cval = expected_cost(rates, cost)/base_norm
                    if mode == "max_f1":
                        obj = f1
                    elif mode == "tradeoff":
                        obj = f1 - lam * cval
                    elif mode == "min_cost_at_f1":
                        obj = -cval if (f1_min is not None and f1 >= f1_min) else -1e9
                if obj > best["obj"]:
                    best.update({"t1": t1.item(), "t2": t2.item(), "f1": f1, "rates": rates, "obj": obj, "cost": cval})

        # se t2 è None (tutti escono a ramo1), metti una t2 di comodo
        if best["t2"] is None:
            best["t2"] = float("inf") if use_margin else 0.0
        return best
    



