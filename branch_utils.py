from typing import List, Optional, Tuple
from branches import Node, RawEmbedder
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.utils           import compute_class_weight
from torch.utils.data        import DataLoader, Dataset

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y     = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return (self.X_num[i], self.X_cat[i]), self.y[i]

def get_weights(y: torch.Tensor) -> torch.Tensor:
    w = compute_class_weight("balanced",
                             classes=np.unique(y.numpy()),
                             y=y.numpy())
    return torch.tensor(w, dtype=torch.float32)

def prepare_dl_nb15_branching(
    data: pd.DataFrame,
    target_col: str,
    numerical_cols:   list[str],
    categorical_cols: list[str],
    groups: Optional[list[list]] = None,    # e.g. [["dos","u2r"], ["r2l","probe"]]
    remove_classes: Optional[List[str]] = None,
    include_classes: Optional[list[str]] = None,
    batch_size:  int   = 512,
    test_size:   float = 0.2,
    valid_size:  float = 0.1,
    random_state: int   = 42,
):
    # — 0) filtro iniziale su include / remove
    df0 = data
    if include_classes is not None:
        df0 = df0[df0[target_col].isin(include_classes)]
    if remove_classes is not None:
        df0 = df0[~df0[target_col].isin(remove_classes)]

    # — 1) selezione colonne e pulizia
    df = df0[numerical_cols + categorical_cols + [target_col]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)


    # 2) Costruisco mappatura e class_names
    if groups:
        all_classes = sorted(df[target_col].unique())
        grouped_set = {c for g in groups for c in g}

        mapping = {}
        class_names: list[str] = []
        code = 0

        # 2a) Etichette “easy” rimangono individuali, uscite al primo nodo
        for cls in all_classes:
            if cls not in grouped_set:
                mapping[cls] = code
                class_names.append(cls)
                code += 1

        # 2b) Gruppi aggregati in “group_i”
        for idx, group in enumerate(groups):
            grp_name = f"group_{idx}"
            for cls in group:
                mapping[cls] = code
            class_names.append(grp_name)
            code += 1

        # Applico il mapping al target
        df[target_col] = df[target_col].map(mapping).astype(int)

    else:
        # Se non ci sono gruppi, codifico come categoria standard
        df[target_col] = df[target_col].astype("category")
        class_names = list(df[target_col].cat.categories)
        mapping = {cat: i for i, cat in enumerate(class_names)}
        df[target_col] = df[target_col].map(mapping).astype(int)

    # 3) Split stratificato
    tmp_size = test_size + valid_size
    train_df, tmp_df = train_test_split(
        df, test_size=tmp_size,
        stratify=df[target_col],
        random_state=random_state
    )
    rel_valid = valid_size / tmp_size
    valid_df, test_df = train_test_split(
        tmp_df, test_size=rel_valid,
        stratify=tmp_df[target_col],
        random_state=random_state
    )

    # 4) Encoding categorico + scaling numerico
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[categorical_cols])

    def encode_cat(df_):
        return enc.transform(df_[categorical_cols]).astype(int) + 1

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numerical_cols])
    X_valid_num = scaler.transform(valid_df[numerical_cols])
    X_test_num  = scaler.transform(test_df[numerical_cols])

    # 5) Trasformo in tensori
    to_t = lambda arr, dt: torch.tensor(arr, dtype=dt)
    X_tn = to_t(X_train_num, torch.float32)
    X_vn = to_t(X_valid_num, torch.float32)
    X_te_n = to_t(X_test_num, torch.float32)

    X_tc = to_t(encode_cat(train_df), torch.long)
    X_vc = to_t(encode_cat(valid_df), torch.long)
    X_te_c = to_t(encode_cat(test_df), torch.long)

    y_t  = to_t(train_df[target_col].values, torch.long)
    y_v  = to_t(valid_df[target_col].values, torch.long)
    y_te = to_t(test_df[target_col].values, torch.long)

    # 6) Dataset e DataLoader
    train_ds = TabularDataset(X_tn, X_tc, y_t)
    valid_ds = TabularDataset(X_vn, X_vc, y_v)
    test_ds  = TabularDataset(X_te_n, X_te_c, y_te)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)

    # 7) Cardinalità embedding e pesi di classe
    cat_cardinalities = [len(cats) + 1 for cats in enc.categories_]
    cw = get_weights(y_t)



    return (
        train_dl, valid_dl, test_dl,
        cat_cardinalities,
        cw,
        class_names,
        enc, scaler
    )

def apply_chain(
    batch: Tuple[torch.Tensor, torch.Tensor],
    chain: List[Node],
    device: str = "cuda"
) -> torch.Tensor:
    """
    Prende (x_num, x_cat) e una lista di nodi [n0, n1, ..., nk]
    e restituisce h_k = nk.dense(...n1.dense(n0.embed+dense(x))...).
    """
    x_num, x_cat = batch
    x_num, x_cat = x_num.to(device), x_cat.to(device)
    x = chain[0].embedder(x_num, x_cat)      # root.embedder
    x = chain[0].dense(x)                    # root.dense
    for node in chain[1:]:
        x = node.dense(x)                    # solo dense per i figli
    return x        
    
class FlatDataset(Dataset):
    """Dataset che già contiene coppie (x_flat, y)."""
    def __init__(
        self,
        dataloader: DataLoader,
        chain_node: List[Node],
        device:     str = "cuda",
    ):
        super().__init__()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        # Accumuliamo on-the-fly, batch‐wise
        for (x_num, x_cat), y in dataloader:
            # apply_chain gestisce embedder + dense di tutta la chain
            x_flat = apply_chain((x_num, x_cat), chain_node, device)
            # sposto su CPU per evitare di tenere tutto in GPU
            x_flat_cpu = x_flat.cpu()
            y_cpu      = y.cpu()
            # salvo sample‐wise
            for xi, yi in zip(x_flat_cpu, y_cpu):
                self.samples.append((xi, yi))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

def flatten_dl(
    dataloader: DataLoader,
    chain_node: List[Node],
    embedder: RawEmbedder = None,
    batch_size: int = 512,
    device:     str = "cuda",
) -> DataLoader:
    """
    Restituisce un DataLoader che emette (x_flat, y), dove x_flat
    è il risultato di apply_chain su ogni (x_num,x_cat). Metti embedder solo se root.
    """
    if embedder is not None and len(chain_node) == 1:
        root_flat_train_ds = EmbedDataset(dataloader, embedder, device)
        return DataLoader(root_flat_train_ds, batch_size=batch_size, shuffle=True)
    else:
        flat_ds = FlatDataset(dataloader, chain_node, device)
        return DataLoader(flat_ds, batch_size=batch_size, shuffle=True)

class EmbedDataset(Dataset):
    """Da TabularDataset ((x_num,x_cat),y) a (z,y), con z = embedder(x)."""
    def __init__(
        self,
        tab_loader: DataLoader,
        embedder:   RawEmbedder,
        device:     str = "cuda"
    ):
        embedder.to(device).eval()
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for (x_num, x_cat), y in tab_loader:
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                z = embedder(x_num, x_cat).cpu()    # [B, embed_dim]
                y = y.cpu()
                for zi, yi in zip(z, y):
                    self.samples.append((zi, yi))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]