import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from TRPCA.TRPCA import utils, trpca

# Positional encoding module used in the transformer models.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Transformer model for regression tasks.
class TransformerRegressionModel(nn.Module):
    def __init__(self, feature_size, num_transformer_layers=6, nhead=8, dim_feedforward=2048, dropout=0.4, hidden_layer_size=1024, fast_transformer=True):
        super(TransformerRegressionModel, self).__init__()
        self.d_model = feature_size
        self.positional_encoding = PositionalEncoding(d_model=feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_transformer_layers)
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=nhead)
        self.regressor = nn.Sequential(
            nn.Linear(feature_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

    def forward(self, src, return_embeddings=True, return_attention=True):
        src = self._process_input(src)
        transformer_output = self.transformer_encoder(src)
        attention_output, attention_weights = self.attention(transformer_output, transformer_output, transformer_output)
        output = attention_output.squeeze(0)
        regression_output = self.regressor(output)
        outputs = {'regression_output': regression_output}
        if return_embeddings:
            outputs['embeddings'] = transformer_output
        if return_attention:
            outputs['attention_weights'] = attention_weights
        return outputs

    def _process_input(self, src):
        src = src.unsqueeze(1) * math.sqrt(self.d_model)
        src = src.transpose(0, 1)
        return self.positional_encoding(src)

# Transformer model for classification tasks.
class TransformerClassificationModel(TransformerRegressionModel):
    def __init__(self, feature_size, num_classes, **kwargs):
        super(TransformerClassificationModel, self).__init__(feature_size, **kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, kwargs['hidden_layer_size']),
            nn.ReLU(),
            nn.Linear(kwargs['hidden_layer_size'], num_classes)
        )

    def forward(self, src, return_embeddings=False, return_attention=False):
        src = self._process_input(src)
        transformer_output = self.transformer_encoder(src)
        attention_output, attention_weights = self.attention(transformer_output, transformer_output, transformer_output)
        output = attention_output.squeeze(0)
        class_output = self.classifier(output)
        outputs = {'class_output': class_output}
        if return_embeddings:
            outputs['embeddings'] = transformer_output
        if return_attention:
            outputs['attention_weights'] = attention_weights
        return outputs

# Function to perform regression analysis using the Transformer model.
def trpca_regress(table, metadata, MetadataColumn, test_size=0.2, n_dimensions=128, feature_frequency=5, num_transformer_layers=1, nhead=8, dim_feedforward=2048, epochs=1000):
    df1 = preprocess_data(table, metadata, MetadataColumn, feature_frequency)
    df1, X_pca_tensor = pca_and_tensor(df1, n_dimensions)
    y = df1[MetadataColumn].astype(float)
    train_loader, test_loader = prepare_data_loaders(X_pca_tensor, y, test_size)
    regression_model = fit_transformer_regression(train_loader, test_loader, n_dimensions, num_transformer_layers, nhead, dim_feedforward, epochs)
    plot_losses(regression_model)
    evaluate_regression_model(train_loader, test_loader, regression_model)

# Function to perform classification analysis using the Transformer model.
def trpca_classify(table, metadata, MetadataColumn, test_size=0.2, n_dimensions=128, feature_frequency=5, num_transformer_layers=1, nhead=8, dim_feedforward=2048, epochs=1000):
    df2 = preprocess_data(table, metadata, MetadataColumn, feature_frequency)
    df2, X_pca_tensor = pca_and_tensor(df2, n_dimensions)
    y = df2[MetadataColumn].astype('category').cat.codes
    train_loader, test_loader = prepare_data_loaders(X_pca_tensor, y, test_size, classification=True)
    classification_model = fit_transformer_classification(train_loader, test_loader, n_dimensions, len(y.unique()), num_transformer_layers, nhead, dim_feedforward, epochs)
    plot_losses(classification_model)
    evaluate_classification_model(train_loader, test_loader, classification_model)

# Additional helper functions can be added here to handle data preprocessing, model training, loss plotting, and evaluation to keep the main functions clean and concise.

def preprocess_data(table, metadata, MetadataColumn, feature_frequency):
    # Drop columns with less than 'feature_frequency' non-zero entries
    columns_to_drop = table.columns[table.sum() < feature_frequency]
    df = table.drop(columns=columns_to_drop)
    df = utils.clr_transformation(df)
    # Merge metadata and filter rows with missing metadata
    df[MetadataColumn] = metadata.loc[metadata.index.isin(df.index)][MetadataColumn]
    df = df.loc[df[MetadataColumn].notna()]
    return df

def pca_and_tensor(df, n_dimensions):
    # Apply PCA and convert to DataFrame
    X_reduced, pca = utils.apply_pca(df.drop(columns=MetadataColumn), n_dimensions)
    df_pca = pd.DataFrame(X_reduced, index=df.index)
    # Convert to tensor
    X_pca_tensor = torch.tensor(df_pca.to_numpy(), dtype=torch.float32)
    return df_pca, X_pca_tensor

def prepare_data_loaders(X_pca_tensor, y, test_size, classification=False):
    # Binning if regression and splitting the dataset
    if not classification:
        num_bins = 6
        y_binned = pd.qcut(y, q=num_bins, labels=False, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(X_pca_tensor, y, test_size=test_size, stratify=y_binned)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_pca_tensor, y, test_size=test_size, stratify=y)
    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Converting to PyTorch tensors
    train_features = torch.tensor(X_train_scaled, dtype=torch.float32)
    test_features = torch.tensor(X_test_scaled, dtype=torch.float32)
    if classification:
        train_targets = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        test_targets = torch.tensor(y_test.to_numpy(), dtype=torch.long)
    else:
        train_targets = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
        test_targets = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)
    # DataLoader
    batch_size = 512
    train_dataset = TensorDataset(train_features, train_targets)
    test_dataset = TensorDataset(test_features, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def fit_transformer_regression(train_loader, test_loader, feature_size, num_transformer_layers, nhead, dim_feedforward, epochs):
    model = TransformerRegressionModel(feature_size=feature_size, num_transformer_layers=num_transformer_layers, nhead=nhead, dim_feedforward=dim_feedforward)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs['regression_output'], targets)
            loss.backward()
            optimizer.step()
    return model

def fit_transformer_classification(train_loader, test_loader, feature_size, num_classes, num_transformer_layers, nhead, dim_feedforward, epochs):
    model = TransformerClassificationModel(feature_size=feature_size, num_classes=num_classes, num_transformer_layers=num_transformer_layers, nhead=nhead, dim_feedforward=dim_feedforward)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs['class_output'], targets)
            loss.backward()
            optimizer.step()
    return model

def plot_losses(model):
    # Plotting logic assuming you have collected loss data during training
    plt.figure()
    plt.plot(model.train_losses, label='Training Loss')
    plt.plot(model.valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

def evaluate_regression_model(train_loader, test_loader, model):
    model.eval()
    # Collect predictions and targets to calculate performance metrics
    predictions, targets = [], []
    for inputs, actuals in test_loader:
        with torch.no_grad():
            outputs = model(inputs)
            predictions.extend(outputs['regression_output'].numpy())
            targets.extend(actuals.numpy())
    # Calculate metrics such as R^2 and MAE
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    print(f'R^2: {r2}, MAE: {mae}')

def evaluate_classification_model(train_loader, test_loader, model):
    model.eval()
    # Collect predictions and targets to calculate performance metrics
    predictions, targets = [], []
    for inputs, actuals in test_loader:
        with torch.no_grad():
            outputs = model(inputs)
            pred_labels = outputs['class_output'].argmax(dim=1)
            predictions.extend(pred_labels.numpy())
            targets.extend(actuals.numpy())
    # Calculate accuracy
    accuracy = accuracy_score(targets, predictions)
    print(f'Accuracy: {accuracy}')

