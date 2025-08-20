import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from neural_network import TabularDataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import numpy as np
from collections import OrderedDict

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        it = pd.read_csv(file_path,chunksize=100000, low_memory=True)
        data = pd.concat(it, ignore_index=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def split_data(data, target_col, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    data (pd.DataFrame): The DataFrame to split.
    target_col: Name of target column.
    
    Returns:
    tuple: A tuple containing the training, valid and testing DataFrames.
    """
    train_df, temp_df = train_test_split(data, test_size=0.3, random_state=random_state, stratify=data[target_col])
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df[target_col])   
    
    return train_df, valid_df, test_df

def remove_rows(data, values_to_remove):
    
    for col, values in values_to_remove.items():
        for val in values:
            data = data[data[col] != val]

    return data

def keep_columns(data, columns_to_keep):
    """
    Remove unnecessary columns from the DataFrame.
    
    Parameters:
    data (pd.DataFrame): The DataFrame from which to remove columns.
    columns_to_keep (list): List of column names to keep.
    
    Returns:
    pd.DataFrame: The DataFrame with specified columns removed.
    """
    return data[columns_to_keep]

def get_X_num(data, numerical_cols):
    """
    Get a DataFrame containing only the specified numeric columns.
    
    Parameters:
    data (pd.DataFrame): The original DataFrame.
    numerical_cols (list): List of numeric column names to include.
    
    Returns:
    pd.DataFrame: A DataFrame with only the specified numeric columns.
    """
    return data[numerical_cols]

def get_X_cat(data, categorical_cols):
    """
    Get a DataFrame containing only the specified categorical columns.
    
    Parameters:
    data (pd.DataFrame): The original DataFrame.
    categorical_cols (list): List of categorical column names to include.
    
    Returns:
    pd.DataFrame: A DataFrame with only the specified categorical columns.
    """
    return data[categorical_cols]


def get_weights(y):
    class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y.numpy()),
                                     y=y.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights



def load_and_prepare_data(file_path, target_col, numerical_cols, categorical_cols, rows_to_remove, batch_size=512):
    """
    Load and prepare the data for modeling.
    
    Parameters:
    file_path (str): The path to the CSV file.
    target_col: Name of target column.
    numerical_cols (list): List of numeric column names.
    categorical_cols (list): List of categorical column names.
    columns_to_keep (list): List of columns to keep in the final DataFrame.
    """

    data = load_data(file_path)
    if data is None:
        return None, None, None
    
    data = keep_columns(data, numerical_cols + categorical_cols + [target_col])
    data = remove_rows(data, rows_to_remove)

    data[target_col] = data[target_col].astype('category')
    class_names = data[target_col].cat.categories.tolist()

    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    data[target_col] = data[target_col].cat.codes

    cat_cardinalities = [data[col].nunique() for col in categorical_cols]

    train_df, valid_df, test_df = split_data(data, target_col)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numerical_cols])
    X_valid_num = scaler.transform(valid_df[numerical_cols])
    X_test_num  = scaler.transform(test_df[numerical_cols])
    
    X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    X_valid_num = torch.tensor(X_valid_num, dtype=torch.float32)
    X_test_num = torch.tensor(X_test_num, dtype=torch.float32)
    
    X_train_cat = torch.tensor(get_X_cat(train_df, categorical_cols).values, dtype=torch.long)
    X_valid_cat = torch.tensor(get_X_cat(valid_df, categorical_cols).values, dtype=torch.long)
    X_test_cat = torch.tensor(get_X_cat(test_df, categorical_cols).values, dtype=torch.long)

    y_train = torch.tensor(train_df[target_col].values, dtype=torch.long)
    y_valid = torch.tensor(valid_df[target_col].values, dtype=torch.long)
    y_test = torch.tensor(test_df[target_col].values, dtype=torch.long)

    train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
    valid_dataset = TabularDataset(X_valid_num, X_valid_cat, y_valid)
    test_dataset = TabularDataset(X_test_num, X_test_cat, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    cw = get_weights(y_train)


    
    return train_dataloader, valid_dataloader, test_dataloader, cat_cardinalities, cw, class_names


def load_and_prepare_nb15(file_path, target_col, numerical_cols, categorical_cols, batch_size=512):
    """
    Load and prepare the data for modeling.
    
    Parameters:
    file_path (str): The path to the CSV file.
    target_col: Name of target column.
    numerical_cols (list): List of numeric column names.
    categorical_cols (list): List of categorical column names.
    columns_to_keep (list): List of columns to keep in the final DataFrame.
    """
    data = load_data(file_path)
    if data is None:
        return None, None, None
    
    data = keep_columns(data, numerical_cols + categorical_cols + [target_col])
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    data[target_col] = data[target_col].astype('category')
    class_names = data[target_col].cat.categories.tolist()

    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    data[target_col] = data[target_col].cat.codes

    cat_cardinalities = [data[col].nunique() for col in categorical_cols]

    train_df, valid_df, test_df = split_data(data, target_col)



    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numerical_cols])
    X_valid_num = scaler.transform(valid_df[numerical_cols])
    X_test_num  = scaler.transform(test_df[numerical_cols])
    
    X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    X_valid_num = torch.tensor(X_valid_num, dtype=torch.float32)
    X_test_num = torch.tensor(X_test_num, dtype=torch.float32)
    
    X_train_cat = torch.tensor(get_X_cat(train_df, categorical_cols).values, dtype=torch.long)
    X_valid_cat = torch.tensor(get_X_cat(valid_df, categorical_cols).values, dtype=torch.long)
    X_test_cat = torch.tensor(get_X_cat(test_df, categorical_cols).values, dtype=torch.long)

    y_train = torch.tensor(train_df[target_col].values, dtype=torch.long)
    y_valid = torch.tensor(valid_df[target_col].values, dtype=torch.long)
    y_test = torch.tensor(test_df[target_col].values, dtype=torch.long)

    train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
    valid_dataset = TabularDataset(X_valid_num, X_valid_cat, y_valid)
    test_dataset = TabularDataset(X_test_num, X_test_cat, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    cw = get_weights(y_train)


    
    return train_dataloader, valid_dataloader, test_dataloader, cat_cardinalities, cw, class_names, train_dataset, y_train


def _stratified_subset_indices(y, fraction: float, random_state: int = 42):
    """
    Restituisce indici di un sottoinsieme stratificato di dimensione 'fraction'
    rispetto alle etichette y (torch.Tensor o array-like).
    """
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction deve essere in (0, 1], ricevuto: {fraction}")

    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    n = len(y_np)
    if n == 0:
        raise ValueError("Vettore etichette vuoto.")

    # Evita che qualche classe resti con 0 campioni
    _, counts = np.unique(y_np, return_counts=True)
    if (counts * fraction).astype(int).min() < 1:
        raise ValueError(
            f"La frazione {fraction} è troppo piccola: alcune classi avrebbero 0 campioni."
        )

    sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=random_state)
    idx = np.arange(n)
    sub_idx, _ = next(sss.split(idx, y_np))
    return sub_idx

def make_stratified_subset_dataloaders(train_dataset,
                                       y_train: torch.Tensor,
                                       fractions,
                                       batch_size: int = 512,
                                       random_state: int = 42,
                                       shuffle: bool = True,
                                       num_workers: int = 0):
    """
    Crea DataLoader stratificati per ciascuna frazione richiesta.

    Parametri:
      - train_dataset: TabularDataset completo (già creato da load_and_prepare_data)
      - y_train: tensor delle etichette del train (stesso ordine del dataset)
      - fractions: iterabile di float (es. [1.0, 0.75, 0.50, 0.25, 0.10, 0.05])
      - batch_size, random_state, shuffle, num_workers: come in DataLoader

    Ritorna:
      - OrderedDict { frazione(float) : DataLoader }

    Note:
      - Ogni frazione è campionata indipendentemente (non garantisce subset annidati).
      - Se una frazione è impossibile (classe troppo rara), solleva ValueError.
    """
    loaders = OrderedDict()
    for f in fractions:
        sub_idx = _stratified_subset_indices(y_train, float(f), random_state=random_state)
        subset = Subset(train_dataset, sub_idx)
        loaders[float(f)] = DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
    return loaders