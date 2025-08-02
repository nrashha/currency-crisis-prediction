#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import openpyxl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import PatchTSTConfig, PatchTSTForClassification
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification
import random
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
    set_seed
)
from torch.utils.data import Dataset
import accelerate
from transformers import PatchTSTForClassification, PatchTSTConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import set_seed
from sklearn.metrics import accuracy_score, f1_score
import itertools
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from itertools import product


# In[104]:


config = {
    "seed": 42,
    "device": "cpu",

    # 11 macro indicators + 24 country one-hot
    "input_dim": 35,     
    "seq_len": 5,
    "batch_size": 32,

    "d_model": 64,         
    "nhead": 2,
    "num_layers": 4,    
    "dropout": 0.1,

    "learning_rate": 1e-3,
    "num_epochs": 20
}


# In[105]:


# set seed for reproducibility

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(config["seed"])


# In[106]:


# loading raw data on economic indicators of 24 different countries 

file_path = file_path = "/Users/nahianrashha/Downloads/31b601f5-342d-440c-8e63-04a635edc50b_Data.csv"
df = pd.read_csv(file_path)
df.head()


# In[107]:


# data processing
# input missing macroeconomic indicator values

# clean year column names
year_cols = [col for col in df.columns if "[YR" in col]
cleaned_year_cols = {col: col.split(" ")[0] for col in year_cols}
df.rename(columns=cleaned_year_cols, inplace=True)

# take into long format
df_long = df.melt(
    id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
    var_name="Year",
    value_name="Value"
)
df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long.dropna(subset=["Year"], inplace=True)

# pivot to wide format (Country-Year × Indicators)
df_pivot = df_long.pivot_table(
    index=["Country Name", "Year"],
    columns="Series Name",
    values="Value"
).reset_index()


# In[108]:


# impute missing indicator values using column-wise mean 
# use forward fill and backward fill to fill-in missing indicator values
indicator_columns = df_pivot.columns.difference(["Country Name", "Year"])
df_pivot[indicator_columns] = df_pivot[indicator_columns].fillna(df_pivot[indicator_columns].mean())

# one-hot encode countries
df_encoded = pd.get_dummies(df_pivot, columns=["Country Name"])

df_encoded_with_country = df_encoded.copy()
df_encoded_with_country["Country Name"] = df_pivot["Country Name"]


# In[109]:


# load and clean crisis label data
crisis_excel_path = "/Users/nahianrashha/Downloads/20160923_global_crisis_data (1).xlsx"
crisis_data = pd.read_excel(crisis_excel_path, sheet_name="Sheet1")
crisis_clean = crisis_data[["Country", "Year", "Currency Crises"]].copy()
crisis_clean.dropna(subset=["Country", "Year", "Currency Crises"], inplace=True)
crisis_clean["Year"] = crisis_clean["Year"].astype(int)
crisis_clean["Currency Crises"] = pd.to_numeric(
    crisis_clean["Currency Crises"], errors="coerce"
).fillna(0).astype(int)
crisis_clean.rename(columns={"Country": "Country Name"}, inplace=True)

# merge with crisis labels
merged_imputed_df = df_encoded_with_country.merge(
    crisis_clean,
    on=["Country Name", "Year"],
    how="left"
)
merged_imputed_df["Currency Crises"] = merged_imputed_df["Currency Crises"].fillna(0).astype(int)


# In[110]:


# save the new enriched dataset
imputed_path = "/Users/nahianrashha/Downloads/merged_dataset_imputed.csv"
merged_imputed_df.to_csv(imputed_path, index=False)

# return path for download
{
    "download_path": imputed_path,
    "num_rows": merged_imputed_df.shape[0],
    "num_features": merged_imputed_df.shape[1],
    "num_country_columns": sum(col.startswith("Country Name_") for col in merged_imputed_df.columns),
    "num_crisis_cases": merged_imputed_df["Currency Crises"].sum(),
    "crisis_ratio": merged_imputed_df["Currency Crises"].mean(),
    "year_range": (merged_imputed_df["Year"].min(), merged_imputed_df["Year"].max())
}


# In[111]:


merged_imputed_df.head()


# In[112]:


indicator_cols = [
    "Central government debt, total (% of GDP)",
    "Current account balance (% of GDP)",
    "Exports of goods and services (% of GDP)",
    "External debt stocks (% of GNI)",
    "GDP growth (annual %)",
    "Imports of goods and services (% of GDP)",
    "Inflation, consumer prices (annual %)",
    "Interest rate spread (lending rate minus deposit rate, %)",
    "Official exchange rate (LCU per US$, period average)",
    "Total reserves (includes gold, current US$)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)",
]

country_onehot_cols = [col for col in merged_imputed_df.columns if col.startswith("Country Name_")]


# In[113]:


# function to create samples
# input sequences and output labels for binary classification task
# 820 samples: each is one 5-year window for 1 country
# each X[i] is a matrix of 5 consecutive years (5 year matrix) of 11 macronomic indicators + 24 country one-hot features
# each row is the data for a single year for a single country. Shape is (5, 35)
# Y[i] is a crisis label: 1 for yes (in the immediately following year) and 0 for no

def create_training_samples_with_years(df, indicator_cols, country_onehot_cols, label_col="Currency Crises", window_size=5):
    X, Y, years = [], [], []
    grouped = df.groupby("Country Name")
    
    for country, group in grouped:
        group_sorted = group.sort_values("Year")
        if len(group_sorted) < window_size + 1:
            continue
        for i in range(len(group_sorted) - window_size):
            window = group_sorted.iloc[i : i + window_size]
            target = group_sorted.iloc[i + window_size]
            X_window = window[indicator_cols].values
            country_onehot = window[country_onehot_cols].values[0]
            X_combined = np.hstack([X_window, np.tile(country_onehot, (window_size, 1))])
            X.append(X_combined)
            Y.append(target[label_col])
            years.append(target["Year"])
    return np.array(X), np.array(Y), np.array(years)


# In[114]:


# create raw samples
X, Y, years = create_training_samples_with_years(merged_imputed_df, indicator_cols, country_onehot_cols)

# split macro and one-hot features
X_macro = X[:, :, :len(indicator_cols)]
X_onehot = X[:, :, len(indicator_cols):]


# In[115]:


# normalize
mean = X_macro.mean(axis=(0, 1), keepdims=True)
std = X_macro.std(axis=(0, 1), keepdims=True)
X_macro_norm = (X_macro - mean) / (std + 1e-8)


# In[116]:


# recombine and convert to tensors
X_processed = np.concatenate([X_macro_norm, X_onehot], axis=-1)

X_tensor = torch.tensor(X_processed, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# filter our all non-binary labels
mask = (Y_tensor == 0) | (Y_tensor == 1)
X_tensor = X_tensor[mask]
Y_tensor = Y_tensor[mask]


# In[117]:


# train/val/test split (70/15/15)
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_tensor, Y_tensor, test_size=0.15, stratify=Y_tensor, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.1765, stratify=Y_temp, random_state=42
)


# In[118]:


# dataLoaders
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32)


# In[119]:


# inspect: summary
{
    "X_tensor_shape": X_tensor.shape,
    "Y_tensor_shape": Y_tensor.shape,
    "num_crisis_labels": int(Y_tensor.sum().item()),
    "crisis_ratio": float(Y_tensor.mean().item()),
    "train_size": len(train_loader.dataset),
    "val_size": len(val_loader.dataset),
    "test_size": len(test_loader.dataset)
}


# In[120]:


X_macro = X_tensor[:, :, :11]  # first 11 features are macro indicators

# inspect mean and sd over all samples and time steps
macro_mean = X_macro.mean(dim=(0, 1))
macro_std = X_macro.std(dim=(0, 1))

print("Means (should be ~0):", macro_mean)
print("Stds  (should be ~1):", macro_std)


# In[121]:


# M1: transformer-based binary classifier
# hardcoded hyperparamters from config, no tuning
# look at sequences of macroeconomic data to predict whether a currency crisis will occur in the future
# input sequence: 5 years of 11 macro indicators + 24 countries (one-hot); shape: (batch, 5, 35)
# using only the CLS token (first timestep) as the summary representation of the sequence for classification

class BinaryTransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, nhead, num_layers, dropout):
        super().__init__()
        
        # linearly project each 35-dimensional year input onto a 64-dimensional embedding 
    
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # positional embedding, 0430: self.pos_embedding is a learnable tensor that tells the model which year is which in the 5-year sequence; 
        # added to projected input; so inject position to preserve sequence order

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # define 1st layer: multi-head attention + feedforward network (mlp)
        # stack 4 of them (num_layers = 4)
        # nhead = 2 parallel attention heads
        # so we get cross-year attention: each year attends to each other
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # feedforward classifier: takes output from first time-step (CLS token) after transformer processing
        # goes through hidden layers' activation: MLP
        # outputs logit for classification (1-dim)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    # transformer encoder processes the entire sequence with self-attention across years
    # cache the first time step's output
    # feed the summary vector into the classifier head and outputs a logit (raw score) for each sample.
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = self.encoder(x)
        x_cls = x[:, 0, :]
        logits = self.cls_head(x_cls).squeeze(-1)
        return logits


# In[122]:


# evaluation function (standard from M1–M3)
# convert logits (raw scores) to probabilities between 0 and 1 using the sigmoid function.
# threshold probailities 
# compute metrics: 
# Accuracy = % of correct predictions
# F1 score = harmonic mean of precision and recall (better for imbalanced data)

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            y_pred = (probs > 0.5).long().cpu().numpy()
            preds.extend(y_pred)
            labels.extend(y_batch.numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds)


# In[123]:


# M1: Hardcoded config model, trained for 20 epochs
# dataloader: shuffle batches of training and validation samples
# initialize model with the current config

def train_M1(X_train, Y_train, X_val, Y_val, config):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=config["batch_size"])

    model = BinaryTransformerClassifier(
        input_dim=config["input_dim"],
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(config["device"])

    # BCEWithLogitsLoss with pos_weight
    # pos_weight > 1: assigns more loss penalty to false negatives to deal with imbalanced classification
    # Adam optimizer
    
    pos_weight = (Y_train == 0).sum() / (Y_train == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config["device"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # in each epoch: for each batch:
    # compute logits (raw crisis score)
    # compute loss (weighted BCE)
    # backprop then update weights (optimizer.step())
    # track total loss
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config["device"]), y_batch.float().to(config["device"])
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # apply sigmoid and compute accuracy, f1
        acc, f1 = evaluate_model(model, val_loader, config["device"])
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Acc = {acc:.4f}, F1 = {f1:.4f}")

    return model


# In[124]:


# M2: Grid Search tuning on val set (hyperparameter tuning)
# for each combo of hp (16): initialize model, train it on training data for 15 epochs, 
# eval it on the valid set; track valid f1 score and pick the best one

def train_M2(X_train, Y_train, X_val, Y_val, param_grid, config):
    best_f1 = -1
    best_model = None
    best_config = None
    results = []

    # shuffling samples
    train_loader = lambda X, Y: DataLoader(TensorDataset(X, Y), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=config["batch_size"])

    # grid search
    for lr, d_model, dropout, num_layers in product(*param_grid.values()):
        print(f"Training with lr={lr}, d_model={d_model}, dropout={dropout}, layers={num_layers}")

        # new transformer of 4 layers with specified h-p combo
        model = BinaryTransformerClassifier(
            input_dim=config["input_dim"],
            seq_len=config["seq_len"],
            d_model=d_model,
            nhead=config["nhead"],
            num_layers=num_layers,
            dropout=dropout
        ).to(config["device"])

        # capped pos_weight to avoid undertraining minority class: crisis = 1 by giving it more weight
        # adam: optimizes lr for each param
        pos_weight = torch.tensor(2.0).to(config["device"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # trains for 15 epochs
        # batch to forward pass to loss to backward pass to optimizer step
        # no total_loss or early stopping 
        for epoch in range(15):  
            model.train()
            for X_batch, y_batch in train_loader(X_train, Y_train):
                X_batch, y_batch = X_batch.to(config["device"]), y_batch.float().to(config["device"])
                loss = criterion(model(X_batch), y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # after training, eval current model’s accuracy and f1 on valid set
        # track best (highest) f1
        acc, f1 = evaluate_model(model, val_loader, config["device"])
        print(f" → Val Acc: {acc:.4f}, F1: {f1:.4f}")
        results.append((lr, d_model, dropout, num_layers, acc, f1))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_config = (lr, d_model, dropout, num_layers)

    return best_model, best_config, results


# In[125]:


# M3: retrain best model config from M2 on train+val and test on test set

def train_M3(X_trainval, Y_trainval, X_test, Y_test, best_config, config):
    torch.manual_seed(config["seed"]) 
    # shuffling the train set
    train_loader = DataLoader(TensorDataset(X_trainval, Y_trainval), batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=config["batch_size"])

    lr, d_model, dropout, num_layers = best_config

    # construct transformer using best hyperparameter set 
    model = BinaryTransformerClassifier(
        input_dim=config["input_dim"],
        seq_len=config["seq_len"],
        d_model=d_model,
        nhead=config["nhead"],
        num_layers=num_layers,
        dropout=dropout
    ).to(config["device"])

    # handling class imbalance
    pos_weight = torch.tensor(1.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config["device"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Restore stability

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config["device"]), y_batch.float().to(config["device"])
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # after training in each epoch, eval on test set
        acc, f1 = evaluate_model(model, test_loader, config["device"])
        print(f"Epoch {epoch+1}: Test Acc = {acc:.4f}, F1 = {f1:.4f}")

    return model


# In[126]:


# hyperparameter grid for grid search

param_grid = {
    "learning_rate": [1e-3, 5e-4],
    "d_model": [64, 128],
    "dropout": [0.1, 0.2],
    "num_layers": [2, 3, 4]
}


# In[127]:


# train m1

model_m1 = train_M1(X_train, Y_train, X_val, Y_val, config)


# In[128]:


# train m2 

best_model_m2, best_config_m2, results_m2 = train_M2(
    X_train, Y_train, X_val, Y_val, param_grid, config
)
print("✅ Best config for M2:", best_config_m2)


# In[ ]:


# train m3 

X_trainval = torch.cat([X_train, X_val], dim=0)
Y_trainval = torch.cat([Y_train, Y_val], dim=0)

model_m3 = train_M3(X_trainval, Y_trainval, X_test, Y_test, best_config_m2, config)


# In[98]:


# linear probing for targeted feature study
# extract the first time step from each transformer layer and store: [input_proj_output, layer1_output] 
# track summary of the 5-year sequence" evolves as it goes through the model: token acquires context through self-attention as it goes through the layers
# on each time step, train a logistic regression classifier
# evaluate how well this probing performs
# if accuracy/F1 is high at a layer, it means the CLS token at that layer contains a strong crisis signal.
# probing analysis on trained Transformer (M3) to assess whether its internal activations(CLS token) 
# contain info about target label (either crisis occurrence or high-risk country status)


def run_probing(model, X_tensor, Y_tensor, risk_labels=None):

    # iterate through the M3's num_layers(2 from grid search).
    # load data in mini-batches for efficient inference
    
    model.eval()
    num_layers = len(model.encoder.layers)
    batch_size = 32
    loader = DataLoader(X_tensor, batch_size=batch_size)

    # cls token output at each layer: index 0 then CLS token after input projection (x = input_proj(x) + pos_embedding)
    # index 1…n then CLS token after each encoder layer 
    # forward pass collects the token’s state after each M3 layer
    
    cls_outputs_by_layer = [[] for _ in range(num_layers + 1)]  
    with torch.no_grad():
        for batch_x in loader:
            x = model.input_proj(batch_x)
            x += model.pos_embedding
            cls_outputs_by_layer[0].append(x[:, 0, :].cpu())  
            for i, layer in enumerate(model.encoder.layers):
                x = layer(x)
                cls_outputs_by_layer[i + 1].append(x[:, 0, :].cpu())

    
    layerwise_X = [torch.cat(outputs, dim=0).numpy() for outputs in cls_outputs_by_layer]
    # classification of probing problem based on risk_labels 
    y_numpy = risk_labels.numpy() if risk_labels is not None else Y_tensor.numpy()

    # each layer i, train linear classifier (logistic regression) on the CLS activations to predict label
    # get accuracy score and f1
    probe_results = []
    for i, X_layer in enumerate(layerwise_X):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_layer, y_numpy)
        preds = clf.predict(X_layer)
        acc = accuracy_score(y_numpy, preds)
        f1 = f1_score(y_numpy, preds)
        print(f"Layer {i}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
        probe_results.append((i, acc, f1))

    # table
    df = pd.DataFrame(probe_results, columns=["Layer", "Accuracy", "F1 Score"])
    fig, ax = plt.subplots(figsize=(6, len(df) * 0.5 + 1))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.savefig("probe_results_table.png", bbox_inches='tight', dpi=300)
    plt.close()

    # bar plot (if we want)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Layer", y="F1 Score", hue="Layer", data=df, palette="Blues_d", legend=False)
    plt.title("Layerwise F1 Score from Probing CLS Token")
    plt.xlabel("Transformer Layer")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("probe_barplot.png", dpi=300)
    plt.close()

    return df


# In[ ]:


# extracts the 1st year’s country one-hot for each sample
# converts one-hot to country index to assigns binary label if country is in a pre-defined high-risk list
# so this is like targetted feature study

def construct_risk_labels(X_tensor, high_risk_indices):
    country_onehots = X_tensor[:, 0, 5:29]  
    country_index = country_onehots.argmax(dim=1)
    return torch.tensor([1 if i in high_risk_indices else 0 for i in country_index], dtype=torch.float32)


# In[100]:


# H2: Does this Transformer activation (e.g., CLS token at layer l) encode the is_high_risk_country feature?

risk_labels = construct_risk_labels(X_tensor, high_risk_indices=[2, 6, 7])  # Brazil, South Africa, Turkey
df_risk = run_probing(model_m3, X_tensor, Y_tensor, risk_labels=risk_labels)


# In[101]:


# H1: Does this Transformer activation (e.g., CLS token at layer l) encode the is_crisis_year_next feature?

df_crisis = run_probing(model_m3, X_tensor, Y_tensor)


# In[102]:


get_ipython().system('jupyter nbconvert --to script DMLV2.ipynb')


# In[ ]:




