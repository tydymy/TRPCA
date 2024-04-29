import math
import torch
import torch.nn as nn
from TRPCA.TRPCA import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from TRPCA.TRPCA import trpca
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough 'position' tensor
        position = torch.arange(max_len).unsqueeze(1)
        # Use div term to create a series of waves
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register pe as a persistent buffer that is not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input tensor
        x = x + self.pe[:x.size(0), :]
        return x
  
class TransformerRegressionModel(nn.Module):
    def __init__(self, feature_size, num_transformer_layers=6, nhead=8, dim_feedforward=2048, dropout=0.4, hidden_layer_size=1024, fast_transformer=True):
        super(TransformerRegressionModel, self).__init__()
        self.d_model = feature_size
        self.fast_transformer = fast_transformer
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=feature_size)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_transformer_layers)
        
        # Multi-head Attention
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=nhead)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(feature_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

    def forward(self, src, return_embeddings=True, return_attention=True):
        if self.fast_transformer:
            # First forward path for fast=True
            # src shape is expected to be [batch_size, feature_size]
            src = src.unsqueeze(1)  # Add a sequence length dimension
            d_model = src.size(-1)  # Assuming the last dimension is the embedding dimension
            src = src * math.sqrt(d_model)
            
            # Transpose to match the transformer's expected input shape [seq_len, batch_size, feature_size]
            src = src.transpose(0, 1)
            
            # Add positional encoding
            src = self.positional_encoding(src)
            
            # Pass through Transformer Encoder
            transformer_output = self.transformer_encoder(src)
            
            # Attention layer
            attention_output, attention_weights = self.attention(transformer_output, transformer_output, transformer_output)
            
            # Squeeze the sequence length dimension and pass through the regression head
            output = attention_output.squeeze(0)
        else:
            # Second forward path for fast=False
            # Reshape the input to add a feature dimension
            src = src.unsqueeze(-1)  # Shape becomes [batch_size, seq_length, 1]

            # Scale the embeddings according to the embedding dimension
            src = src * math.sqrt(self.d_model)

            # Transpose to match the transformer's expected input shape [seq_len, batch_size, feature_size]
            src = src.transpose(0, 1)

            # Add positional encoding
            src = self.positional_encoding(src)

            # Pass through Transformer Encoder
            transformer_output = self.transformer_encoder(src)

            # Attention layer
            attention_output, attention_weights = self.attention(transformer_output, transformer_output, transformer_output)

            # Squeeze the sequence length dimension and pass through the regression head
            output = attention_output[-1]

        regression_output = self.regressor(output)

        # Prepare outputs based on the flags
        outputs = {'regression_output': regression_output}
        if return_embeddings:
            outputs['embeddings'] = transformer_output
        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs

    
class TransformerClassificationModel(nn.Module):
    def __init__(self, feature_size, num_classes, num_transformer_layers=6, nhead=8, dim_feedforward=2048, dropout=0.4, hidden_layer_size=1024, fast_transformer=True):
        super(TransformerClassificationModel, self).__init__()
        self.d_model = feature_size
        self.fast_transformer = fast_transformer
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=feature_size)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_transformer_layers)
        
        # Multi-head Attention
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=nhead)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes)
        )

    def forward(self, src, return_embeddings=False, return_attention=False):
        if self.fast_transformer:
            # Fast path (Assume src shape is [batch_size, feature_size])
            src = src.unsqueeze(1)  # Add a sequence length dimension
            d_model = src.size(-1)  # Assuming the last dimension is the embedding dimension
            src = src * math.sqrt(d_model)
            
            # Transpose to match the transformer's expected input shape [seq_len, batch_size, feature_size]
            src = src.transpose(0, 1)
            
            # Add positional encoding
            src = self.positional_encoding(src)
            
            # Pass through Transformer Encoder
            transformer_output = self.transformer_encoder(src)
            
            # Attention layer - using only for embedding purposes here
            attention_output, attention_weights = self.attention(transformer_output, transformer_output, transformer_output)
            
            # Squeeze the sequence length dimension and pass through the classification head
            output = attention_output.squeeze(0)
        else:
            # Detailed path (Assume src shape is [batch_size, seq_length])
            src = src.unsqueeze(-1)  # Add a feature dimension
            
            # Scale the embeddings according to the embedding dimension
            src = src * math.sqrt(self.d_model)
            
            # Transpose to match the transformer's expected input shape [seq_len, batch_size, feature_size]
            src = src.transpose(0, 1)
            
            # Add positional encoding
            src = self.positional_encoding(src)
            
            # Pass through Transformer Encoder
            transformer_output = self.transformer_encoder(src)
            
            # Attention layer
            attention_output, attention_weights = self.attention(transformer_output, transformer_output, transformer_output)
            
            # Use the last output of the sequence to pass to the classifier
            output = attention_output[-1]

        class_output = self.classifier(output)

        # Prepare outputs based on the flags
        outputs = {'class_output': class_output}
        if return_embeddings:
            outputs['embeddings'] = transformer_output
        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs
    
def trpca_regress(table, metadata, MetadataColumn, test_size=0.2, n_dimensions=128, feature_frequency=5, num_transformer_layers=1, nhead=8, dim_feedforward=2048, epochs=1000):
    columns_to_drop = table.columns[table.sum() < feature_frequency] #drop columns with low prev
    df1 = table.drop(columns=columns_to_drop)
    df1 = utils.clr_transformation(df1)
    print('CLR Transformed.')
    n_dimensions = n_dimensions
    # # # # # # Preprocess with PCA (Re-using the PCA application code from earlier)
    X1_reduced, pca1 = utils.apply_pca(df1, n_dimensions) 
    df = pd.DataFrame(X1_reduced, index=df1.index)
    df[MetadataColumn] = metadata.loc[metadata.index.isin(df.index)][MetadataColumn]
    df = df.loc[df[MetadataColumn].notna()]

    y = df[MetadataColumn].astype(float)

    X_pca_tensor = torch.tensor(df.drop(columns=MetadataColumn).to_numpy(), dtype=torch.float32)

    num_bins = 6

    # Binning the targets
    age_bins = pd.qcut(y, q=num_bins, labels=False, duplicates='drop')

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_pca_tensor, y, test_size=test_size, random_state=42, stratify=age_bins)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Converting to PyTorch tensors
    train_features = torch.tensor(X_train_scaled, dtype=torch.float32)
    test_features = torch.tensor(X_test_scaled, dtype=torch.float32)
    train_targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    test_targets = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Creating DataLoader instances
    train_dataset = TensorDataset(train_features, train_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    batch_size = 512 # You can adjust the batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    regression_model = trpca.TransformerRegressionModel(feature_size=n_dimensions, 
                                                        num_transformer_layers=num_transformer_layers, 
                                                        nhead=nhead, 
                                                        dim_feedforward=dim_feedforward, 
                                                        dropout=0.2, 
                                                        fast_transformer=True)
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in regression_model.parameters())
    trainable_params = sum(p.numel() for p in regression_model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    # Loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = torch.optim.SGD(regression_model.parameters(), lr=8e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=-1)

    epochs = epochs
    best_valid_loss = float('inf')  # Initialize the best validation loss
    train_losses = []
    valid_losses = []
    pbar = tqdm(total=epochs, desc="Training Progress")
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    for epoch in tqdm(range(epochs), total=epochs):
        regression_model.to(device)
        regression_model.train()
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = regression_model(batch_features)
            loss = criterion(outputs['regression_output'], batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.size(0)
        
        # Average train loss for the epoch
        train_loss_epoch = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss_epoch)
        
        # Validation phase
        regression_model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = regression_model(batch_features)
                loss = criterion(outputs['regression_output'], batch_labels)
                valid_loss += loss.item() * batch_features.size(0)
        
        # Average valid loss for the epoch
        valid_loss_epoch = valid_loss / len(test_loader.dataset)
        valid_losses.append(valid_loss_epoch)

        scheduler.step()
        # Update the progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Epoch': epoch + 1,
            'Train Loss': f'{train_loss_epoch:.4f}',
            'Validation Loss': f'{valid_loss_epoch:.4f}'
        })
        
        # Check for best validation loss
        if valid_loss_epoch < best_valid_loss:
            best_valid_loss = valid_loss_epoch
            # Save the model
            torch.save(regression_model.state_dict(), 'best_designated_skin_age_regression_model.pth')
            # print(f'Epoch {epoch+1}: Validation loss improved, model saved.')

    # Close the progress bar after all epochs are done
    pbar.close()
    regression_model.load_state_dict(torch.load('best_designated_skin_age_regression_model.pth'))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

    # After training, evaluate the model on both the training and testing sets
    regression_model.eval()  # Set the model to evaluation mode

    # Collect predictions and actual values for the training set
    train_preds, train_actuals = [], []
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        batch_preds = regression_model(batch_features)['regression_output'].detach()
        train_preds.extend(batch_preds.view(-1).tolist())
        train_actuals.extend(batch_labels.view(-1).tolist())

    # Collect predictions and actual values for the test set
    test_preds, test_actuals = [], []
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        batch_preds = regression_model(batch_features)['regression_output'].detach()
        test_preds.extend(batch_preds.view(-1).tolist())
        test_actuals.extend(batch_labels.view(-1).tolist())

    # Calculate MAE and R^2 for training and testing sets
    train_mae = mean_absolute_error(train_actuals, train_preds)
    test_mae = mean_absolute_error(test_actuals, test_preds)
    train_r2 = r2_score(train_actuals, train_preds)
    test_r2 = r2_score(test_actuals, test_preds)

    # Plotting
    plt.figure(figsize=(10, 8))

    # Train predictions vs actuals
    plt.scatter(train_actuals, train_preds, color='blue', alpha=0.5, label='Train')
    # Test predictions vs actuals
    plt.scatter(test_actuals, test_preds, color='red', alpha=0.5, label='Test')

    # Perfect predictions line
    plt.plot([min(train_actuals+test_actuals), max(train_actuals+test_actuals)], 
            [min(train_actuals+test_actuals), max(train_actuals+test_actuals)], 
            color='darkgreen', linestyle='--')

    plt.title(f'Model Predictions vs Actuals\nTrain MAE: {train_mae:.2f}, Train R2: {train_r2:.2f}\nTest MAE: {test_mae:.2f}, Test R2: {test_r2:.2f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.show()

def trpca_classify(table, metadata, MetadataColumn, test_size=0.2, n_dimensions=128, feature_frequency=5, num_transformer_layers=1, nhead=8, dim_feedforward=2048, epochs=1000):
    columns_to_drop = table.columns[table.sum() < feature_frequency] #drop columns with low prev
    df2 = table.drop(columns=columns_to_drop)
    df2 = utils.clr_transformation(df2)
    print('CLR Transformed.')

    n_dimensions = n_dimensions
    # # # # # # Preprocess with PCA (Re-using the PCA application code from earlier)
    X2_reduced, pca1 = utils.apply_pca(df2, n_dimensions) 
    df = pd.DataFrame(X2_reduced, index=df2.index)
    df[MetadataColumn] = metadata[MetadataColumn]
    df.dropna(subset=[MetadataColumn], inplace=True)

    # Use seaborn to make the scatter plot
    sns.scatterplot(data=df, x=0, y=1, hue=MetadataColumn)

    # Add titles and labels (optional)
    plt.title('PC1 vs PC2 colored by Qiita Host Sex')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # Show legend and plot
    plt.legend(title='Qiita Host Sex')
    plt.show()

    # Assuming 'df' is your DataFrame and 'sex' is the binary classification target
    y = df[MetadataColumn].astype('category').cat.codes

    X_pca_tensor = torch.tensor(df.drop(columns=[MetadataColumn]).to_numpy(), dtype=torch.float32)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_pca_tensor, y, test_size=test_size, random_state=42, stratify=y)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Converting to PyTorch tensors
    train_features = torch.tensor(X_train_scaled, dtype=torch.float32)
    test_features = torch.tensor(X_test_scaled, dtype=torch.float32)
    train_targets = torch.tensor(y_train.values, dtype=torch.long)  # Use torch.long for classification targets
    test_targets = torch.tensor(y_test.values, dtype=torch.long)

    # Creating DataLoader instances
    train_dataset = TensorDataset(train_features, train_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    batch_size = 512  # You can adjust the batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the classification model (assuming two classes, binary classification)
    classification_model = trpca.TransformerClassificationModel(feature_size=n_dimensions, 
                                                                num_classes=len(y.unique()), 
                                                                num_transformer_layers=num_transformer_layers, 
                                                                nhead=nhead, 
                                                                dim_feedforward=dim_feedforward, 
                                                                dropout=0.2, 
                                                                fast_transformer=True)
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in classification_model.parameters())
    trainable_params = sum(p.numel() for p in classification_model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    # Assuming classification_model has been initialized correctly elsewhere in your code
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    classification_model.to(device)
    classification_model.train()

    # Set up loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classification_model.parameters(), lr=8e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=-1)

    epochs = epochs
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []

    pbar = tqdm(total=epochs, desc="Training Progress")

    for epoch in range(epochs):
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = classification_model(batch_features)
            loss = criterion(outputs['class_output'], batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.size(0)

        # Average train loss for the epoch
        train_loss_epoch = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss_epoch)
        
        # Validation phase
        classification_model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = classification_model(batch_features)
                loss = criterion(outputs['class_output'], batch_labels)
                valid_loss += loss.item() * batch_features.size(0)
        
        # Average valid loss for the epoch
        valid_loss_epoch = valid_loss / len(test_loader.dataset)
        valid_losses.append(valid_loss_epoch)

        scheduler.step()
        # Update the progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Epoch': epoch + 1,
            'Train Loss': f'{train_loss_epoch:.4f}',
            'Validation Loss': f'{valid_loss_epoch:.4f}'
        })
        
        # Check for best validation loss
        if valid_loss_epoch < best_valid_loss:
            best_valid_loss = valid_loss_epoch
            # Save the model
            torch.save(classification_model.state_dict(), 'best_classification_model.pth')
            # Uncomment below to see the message when model improves
            # print(f'Epoch {epoch+1}: Validation loss improved, model saved.')

    # Close the progress bar after all epochs are done
    pbar.close()

    classification_model.load_state_dict(torch.load('best_classification_model.pth'))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

    # Assuming classification_model has been defined and trained
    classification_model.eval()  # Set the model to evaluation mode

    # Collect predictions and actual values for the training set
    train_preds, train_actuals = [], []
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        batch_preds = torch.argmax(classification_model(batch_features)['class_output'].detach(), dim=1)
        train_preds.extend(batch_preds.tolist())
        train_actuals.extend(batch_labels.tolist())

    # Collect predictions and actual values for the test set
    test_preds, test_actuals = [], []
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        batch_preds = torch.argmax(classification_model(batch_features)['class_output'].detach(), dim=1)
        test_preds.extend(batch_preds.tolist())
        test_actuals.extend(batch_labels.tolist())

    # Calculate accuracy for training and testing sets
    train_accuracy = accuracy_score(train_actuals, train_preds)
    test_accuracy = accuracy_score(test_actuals, test_preds)

    # Generate confusion matrices for training and testing sets
    train_cm = confusion_matrix(train_actuals, train_preds)
    test_cm = confusion_matrix(test_actuals, test_preds)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Training confusion matrix
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title(f'Train Confusion Matrix\nAccuracy: {train_accuracy:.2f}')
    ax[0].set_xlabel('Predicted Labels')
    ax[0].set_ylabel('True Labels')

    # Test confusion matrix
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Reds', ax=ax[1])
    ax[1].set_title(f'Test Confusion Matrix\nAccuracy: {test_accuracy:.2f}')
    ax[1].set_xlabel('Predicted Labels')
    ax[1].set_ylabel('True Labels')

    plt.tight_layout()
    plt.show()

