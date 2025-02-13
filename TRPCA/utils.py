import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from sklearn.model_selection import train_test_split
import optuna
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
from .trpca import NormalizedTransformer, MTLNormalizedTransformer
from .losses import compute_mtl_loss
import shap
import seaborn as sns
from textwrap import wrap
from gemelli.preprocessing import matrix_rclr
from tqdm.auto import tqdm

class IndexedDataset(torch.utils.data.Dataset):
    """Dataset class that preserves original DataFrame indices"""
    def __init__(self, features, targets, original_indices):
        self.features = features
        self.targets = targets
        self.original_indices = original_indices  # Store the original DataFrame indices
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    @property
    def indices(self):
        return self.original_indices 

def clr_transform(data):
    """
    Apply Centered Log Ratio transformation to the data
    """
    # Add small constant to handle zeros
    data = data + 1
    # Calculate geometric mean for each sample
    geometric_mean = np.exp(np.mean(np.log(data), axis=1))
    # Apply CLR transformation
    clr_data = np.log(data) - np.log(geometric_mean[:, np.newaxis])
    return clr_data

def preprocess(df, series, stratify_col=None, group_col=None, num_pcs=256, test_split=0.2, batch_size=32):
    """
    Preprocess data with support for both grouping and stratification
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with samples as rows and features as columns
    series : pandas.Series
        Target variable for regression
    stratify_col : str or Series
        Column name or Series for stratification (e.g., study names)
    group_col : str or Series
        Column name or Series for grouping (e.g., subject IDs)
    num_pcs : int
        Number of principal components to keep (default: 256)
    test_split : float
        Proportion of data to use for testing (default: 0.2)
    """
    # Convert data to numpy arrays
    X = df.values
    y = series.values
    
    # Store original DataFrame indices
    original_indices = df.index.values
    
    # Apply CLR transformation
    X_clr = matrix_rclr(X+1)#clr_transform(X)#
    
    # Initialize and fit PCA
    pca = PCA(n_components=num_pcs, whiten=False)#min(num_pcs, X_clr.shape[1]))
    
    # Handle stratification and grouping variables
    if stratify_col is not None and isinstance(stratify_col, pd.Series):
        stratify = stratify_col.values
    elif stratify_col is not None and isinstance(df, pd.DataFrame):
        stratify = df[stratify_col].values
    else:
        stratify = None
        
    if group_col is not None and isinstance(group_col, pd.Series):
        groups = group_col.values
    elif group_col is not None and isinstance(df, pd.DataFrame):
        groups = df[group_col].values
    else:
        groups = None
    
    # Split data while tracking original indices
    if groups is not None:
        unique_groups = np.unique(groups)
        print(f"Total number of unique groups (subjects): {len(unique_groups)}")
        
        if stratify is not None:
            # Create a DataFrame with group-level information
            group_info = pd.DataFrame({
                'group': groups,
                'stratify': stratify
            })
            
            # Get the majority stratification value for each group
            group_strat = group_info.groupby('group')['stratify'].agg(
                lambda x: pd.Series.mode(x)[0]
            ).reset_index()
            
            print("\nStratification distribution before splitting:")
            print(pd.Series(stratify).value_counts(normalize=True))
            
            # Split groups while maintaining stratification proportions
            train_groups, test_groups = train_test_split(
                group_strat['group'],
                test_size=test_split,
                stratify=group_strat['stratify'],
                random_state=42
            )
            
            # Create masks for train and test splits
            train_mask = np.isin(groups, train_groups)
            test_mask = np.isin(groups, test_groups)
            
            # Print stratification distribution for each split
            print("\nTrain split stratification distribution:")
            print(pd.Series(stratify[train_mask]).value_counts(normalize=True))
            print("\nTest split stratification distribution:")
            print(pd.Series(stratify[test_mask]).value_counts(normalize=True))
            
        else:
            # Regular group-based splitting without stratification
            train_groups, test_groups = train_test_split(
                unique_groups,
                test_size=test_split,
                random_state=42
            )
            train_mask = np.isin(groups, train_groups)
            test_mask = np.isin(groups, test_groups)
        
        # Verify that every sample is assigned to exactly one split
        assert np.all(train_mask | test_mask), "Some samples are not assigned to any split"
        assert not np.any(train_mask & test_mask), "Some samples are assigned to both splits"
        
        # Split the data
        X_train, y_train = X_clr[train_mask], y[train_mask]
        X_test, y_test = X_clr[test_mask], y[test_mask]
        train_indices = original_indices[train_mask]
        test_indices = original_indices[test_mask]
        
        # Validate group integrity
        train_groups_actual = np.unique(groups[train_mask])
        test_groups_actual = np.unique(groups[test_mask])
        group_overlap = np.intersect1d(train_groups_actual, test_groups_actual)
        assert len(group_overlap) == 0, f"Found groups in both train and test: {group_overlap}"
        
        print(f"\nSplit Summary:")
        print(f"Groups in train: {len(train_groups_actual)}")
        print(f"Groups in test: {len(test_groups_actual)}")
        print(f"Samples in train: {len(X_train)}")
        print(f"Samples in test: {len(X_test)}")
        print("✓ Group integrity verified: No subject's samples are split between train and test")
        
    else:
        # Regular or stratified splitting without grouping
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X_clr, 
            y,
            original_indices,
            test_size=test_split, 
            stratify=stratify,
            random_state=42
        )
    
    # Fit PCA on training data and transform both sets
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_pca)
    X_test_tensor = torch.FloatTensor(X_test_pca)
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets with original indices
    train_dataset = IndexedDataset(X_train_tensor, y_train_tensor, train_indices)
    test_dataset = IndexedDataset(X_test_tensor, y_test_tensor, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, pca

def train_model(
    train_loader: DataLoader,
    model_params: Dict,
    num_epochs: int = 20,
    device: str = 'cuda',
    val_split: float = 0.2,
    use_optuna: bool = False,
    n_trials: int = 100
) -> Tuple[nn.Module, Dict, plt.Figure]:
    """
    Train the model with optional Optuna hyperparameter tuning
    """
    # Split training data into train and validation sets
    torch.manual_seed(42)
    dataset = train_loader.dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader_split = DataLoader(
        train_dataset, 
        batch_size=train_loader.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_loader.batch_size,
        shuffle=False
    )
    
    def train_with_params(lr, weight_decay, optimizer_name):
        # Initialize model
        model = NormalizedTransformer(**model_params).to(device)
        
        # Initialize optimizer
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=2,
            eta_min=0.00005
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_model_state = None
        history = {'train_loss': [], 'val_loss': []}
        
        # Create single progress bar for epochs
        pbar = tqdm(range(num_epochs), desc='Training Progress', ascii=True)
        
        for epoch in pbar:
            # Training phase
            model.train()
            train_loss = 0.0
            
            for x_batch, y_batch in train_loader_split:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                y_batch = y_batch.view(-1, 1)
                loss = criterion(outputs['regression_output'], y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    y_batch = y_batch.view(-1, 1)
                    loss = criterion(outputs['regression_output'], y_batch)
                    val_loss += loss.item()
            
            # Calculate average losses
            train_loss /= len(train_loader_split)
            val_loss /= len(val_loader)
            
            # Store losses in history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'train': f'{train_loss:.4f}',
                'val': f'{val_loss:.4f}',
                'best': f'{best_val_loss:.4f}'
            })
            
            # Update best model if needed
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        # Load best model state
        model.load_state_dict(best_model_state)
        return model, history, best_val_loss
    
    if use_optuna:
        def objective(trial):
            lr = trial.suggest_float('lr', 5e-5, 5e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay',  5e-6, 1, log=True)
            optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
            
            _, _, best_val_loss = train_with_params(lr, weight_decay, optimizer_name)
            return best_val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train final model with best parameters
        best_params = study.best_params
        final_model, final_history, _ = train_with_params(
            best_params['lr'],
            best_params['weight_decay'],
            best_params['optimizer']
        )
    else:
        final_model, final_history, _ = train_with_params(1e-3, 0.1, 'AdamW')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot losses with enhanced styling
    epochs = range(1, len(final_history['train_loss']) + 1)
    
    ax.plot(epochs, final_history['train_loss'], 
            color='cornflowerblue', 
            label='Training Loss',
            linewidth=2)
    ax.plot(epochs, final_history['val_loss'], 
            color='crimson', 
            label='Validation Loss',
            linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=12, labelpad=10)
    ax.set_ylabel('Loss', fontsize=12, labelpad=10)
    ax.set_title('Training and Validation Loss', 
                fontsize=14, 
                pad=20)
    
    # Enhance legend
    ax.legend(frameon=True, 
             fancybox=True, 
             shadow=True, 
             fontsize=10)
    
    # Add grid with lower opacity
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Make spines less prominent
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Set y-axis to log scale for better visualization of loss changes
    ax.set_yscale('log')
    
    # Add light horizontal lines at decade intervals
    y_major = plt.LogLocator(base=10)
    ax.yaxis.set_major_locator(y_major)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Adjust layout
    plt.tight_layout()
    return final_model, final_history, fig

def predict_and_evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], plt.Figure]:
    """
    Evaluate model on test data and generate predictions with performance metrics
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model to evaluate
    test_loader : DataLoader
        DataLoader containing the test data
    device : str
        Device to run predictions on ('cuda' or 'cpu')
    
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary containing evaluation metrics
    predictions : Dict[str, np.ndarray]
        Dictionary containing true and predicted values
    fig : plt.Figure
        Scatter plot of predictions vs true values
    """
    model.eval()
    y_true = []
    y_pred = []
    
    # Generate predictions
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            y_batch = y_batch.view(-1, 1)
            predictions = outputs['regression_output']
            
            y_true.append(y_batch.numpy())
            y_pred.append(predictions.cpu().numpy())
    
    # Concatenate batches
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Store predictions
    predictions = {
        'true_values': y_true,
        'predicted_values': y_pred
    }

    # Set the style
    plt.style.use('default')
    
    # Create figure and axis with a square aspect ratio
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create scatter plot with enhanced styling
    scatter = ax.scatter(y_true, y_pred, 
                        alpha=0.6, 
                        c='cornflowerblue',
                        edgecolor='white',
                        s=100)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    line = ax.plot([min_val, max_val], [min_val, max_val], 
                  '--', 
                  color='crimson', 
                  label='Perfect Prediction',
                  linewidth=2)
    
    # Add metrics text in a nice box
    metrics_text = (f'$R^2$ = {metrics["r2"]:.3f}\n'
                   f'MAE = {metrics["mae"]:.3f}\n'
                   f'RMSE = {metrics["rmse"]:.3f}')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props)
    
    # Customize the plot
    ax.set_xlabel('True Values', fontsize=12, labelpad=10)
    ax.set_ylabel('Predicted Values', fontsize=12, labelpad=10)
    ax.set_title('Prediction vs True Values', 
                fontsize=14, 
                pad=20)
    
    # Add grid with lower opacity
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Make spines less prominent
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Equal aspect ratio to make the plot square
    ax.set_aspect('equal', adjustable='box')
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    print("Test Set Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.3f}")

    return metrics, predictions, fig

def extract_feature_importance(model, train_loader, test_loader, pca, original_features, device='cuda', batch_size=32):
    """Extract and analyze feature importance using SHAP values"""
    # Extract data from loaders
    X_train, y_train = get_data_from_loader(train_loader)
    X_test, y_test = get_data_from_loader(test_loader)
    
    def batched_model_wrapper(x, batch_size=batch_size):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        n_samples = x.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        outputs = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch = x[start_idx:end_idx]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                batch_output = model(batch_tensor)
            
            batch_numpy = batch_output['regression_output'].cpu().numpy()
            outputs.append(batch_numpy)
        
        return np.concatenate(outputs, axis=0)
    
    # Combine train and test data
    X_combined = np.concatenate([X_train, X_test], axis=0)
    y_combined = np.concatenate([y_train, y_test])
    
    # Create SHAP explainer
    explainer = shap.Explainer(batched_model_wrapper, X_train)
    
    # Calculate SHAP values
    shap_values = explainer(X_combined, max_evals=2*X_combined.shape[1]+1)
    
    # Transform SHAP values back to original feature space
    original_feature_importance = np.dot(shap_values.values, pca.components_)
    
    # Calculate importance metrics
    mean_abs_importance = np.mean(np.abs(original_feature_importance), axis=0)
    mean_importance = np.mean(original_feature_importance, axis=0)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': original_features,
        'Mean_Absolute_Importance': mean_abs_importance,
        'Mean_Importance': mean_importance,
        'Std_Importance': np.std(original_feature_importance, axis=0)
    })
    
    return importance_df, original_feature_importance, shap_values, y_combined

def truncate_feature_name(name, length=20):
    """Truncate feature name to last n characters"""
    return '...' + str(name)[-length:] if len(str(name)) > length else str(name)

def plot_feature_importance(importance_df, n_features=20):
    """Create bar plot of top positive and negative features"""
    plt.style.use('default')
    
    # Handle positive features
    positive_features = importance_df[importance_df['Mean_Importance'] > 0].copy()
    positive_features = positive_features.nlargest(n_features//2, 'Mean_Importance')
    positive_features = positive_features.sort_values('Mean_Importance', ascending=True)
    
    # Handle negative features
    negative_features = importance_df[importance_df['Mean_Importance'] < 0].copy()
    negative_features['Abs_Mean_Importance'] = negative_features['Mean_Importance'].abs()
    negative_features = negative_features.nlargest(n_features//2, 'Abs_Mean_Importance')
    negative_features = negative_features.sort_values('Mean_Importance', ascending=True)
    negative_features = negative_features.drop('Abs_Mean_Importance', axis=1)
    
    # Combine features
    top_features = pd.concat([negative_features, positive_features])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    
    # Truncate feature names
    feature_names = [truncate_feature_name(name) for name in top_features['Feature']]
    
    # Create bars
    colors = np.where(top_features['Mean_Importance'] > 0, '#DB4437', '#4285F4')
    bars = ax.barh(range(len(top_features)), 
                  top_features['Mean_Importance'],
                  color=colors)
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel('Mean SHAP Value', fontsize=12)
    ax.set_title('Top Feature Importance', fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_importance_heatmap(feature_importance_matrix, feature_names, target_values, n_top_features=50):
    """Create clustered heatmap of feature importance"""
    plt.style.use('default')
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame(feature_importance_matrix, columns=feature_names)
    
    # Sort by target values
    sorted_indices = np.argsort(target_values)
    importance_df = importance_df.iloc[sorted_indices]
    
    # Select top features by mean absolute importance
    mean_importance = importance_df.abs().mean()
    top_features = mean_importance.nlargest(n_top_features).index
    
    # Filter and scale data
    filtered_data = importance_df[top_features]
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(filtered_data),
        columns=[truncate_feature_name(name) for name in filtered_data.columns]
    )
    
    # Create clustered heatmap
    g = sns.clustermap(
        scaled_data,
        cmap='RdBu_r',
        center=0,
        figsize=(15, 12),
        xticklabels=True,
        yticklabels=False,
        dendrogram_ratio=(.1, .1),
        cbar_pos=(1.0, .2, .03, .4),
        cbar_kws={'label': 'Standardized Importance'},
        row_cluster=False,
        col_cluster=True
    )
    
    plt.setp(g.ax_heatmap.get_xticklabels(), 
            rotation=45, 
            ha='right', 
            fontsize=8)
    
    g.fig.suptitle('Feature Importance Clustered Heatmap', 
                   y=1.02, 
                   fontsize=14)
    
    return g.figure

def get_data_from_loader(loader):
    """Extract all data and original indices from a DataLoader"""
    all_x = []
    all_y = []
    all_indices = []
    
    # Get all original indices from the dataset
    original_indices = loader.dataset.indices
    
    for i, (x_batch, y_batch) in enumerate(loader):
        start_idx = i * loader.batch_size
        end_idx = min((i + 1) * loader.batch_size, len(loader.dataset))
        # Use the original indices for this batch
        batch_indices = original_indices[start_idx:end_idx]
        
        all_x.append(x_batch.cpu().numpy())
        all_y.append(y_batch.cpu().numpy())
        all_indices.append(batch_indices)
    
    return (np.vstack(all_x), 
            np.concatenate(all_y), 
            np.concatenate(all_indices))

def analyze_features(model, train_loader, test_loader, pca, original_features, device='cuda'):
    """Analyze features while maintaining original DataFrame indices"""
    # Extract data and indices
    X_train, y_train, train_indices = get_data_from_loader(train_loader)
    X_test, y_test, test_indices = get_data_from_loader(test_loader)
    
    # Combine data and indices
    X_combined = np.concatenate([X_train, X_test], axis=0)
    y_combined = np.concatenate([y_train, y_test])
    combined_indices = np.concatenate([train_indices, test_indices])
    
    # Sort data by original indices
    sort_idx = np.argsort(combined_indices)
    X_combined = X_combined[sort_idx]
    y_combined = y_combined[sort_idx]
    combined_indices = combined_indices[sort_idx]
    
    # Create batched model wrapper
    def batched_model_wrapper(x, batch_size=32):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        n_samples = x.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        outputs = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch = x[start_idx:end_idx]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                batch_output = model(batch_tensor)
            
            batch_numpy = batch_output['regression_output'].cpu().numpy()
            outputs.append(batch_numpy)
        
        return np.concatenate(outputs, axis=0)
    
    # Create SHAP explainer and calculate values
    explainer = shap.Explainer(batched_model_wrapper, X_train)
    shap_values = explainer(X_combined, max_evals=2*X_combined.shape[1]+1)
    
    # Transform SHAP values back to original feature space
    original_feature_importance = np.dot(shap_values.values, pca.components_)
    
    # Calculate importance metrics
    mean_abs_importance = np.mean(np.abs(original_feature_importance), axis=0)
    mean_importance = np.mean(original_feature_importance, axis=0)
    
    # Create overall importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': original_features,
        'Mean_Absolute_Importance': mean_abs_importance,
        'Mean_Importance': mean_importance,
        'Std_Importance': np.std(original_feature_importance, axis=0)
    })
    
    # Create sample-level feature importance DataFrame with original indices
    sample_importance_df = pd.DataFrame(
        original_feature_importance,
        columns=original_features,
        index=combined_indices  # Use original DataFrame indices
    )
    
    # Create visualizations
    importance_plot = plot_feature_importance(importance_df)
    heatmap_plot = create_importance_heatmap(
        original_feature_importance, 
        original_features,
        y_combined
    )
    
    return {
        'feature_importance_df': importance_df,
        'sample_importance_df': sample_importance_df,
        'shap_values': shap_values,
        'target_values': y_combined,
        'importance_plot': importance_plot,
        'heatmap_plot': heatmap_plot
    }

# Multi-task learning functions for preprcessing, training and evaluation. 
def preprocess_mtl(df, reg_series, cls_series, stratify_col=None, group_col=None, 
                   num_pcs=256, test_split=0.2, batch_size=32):
    """
    Preprocess data for multi-task learning with proper group-based stratification
    """
    # Convert data to numpy arrays
    X = df.values
    X = matrix_rclr(X+1)
    y_reg = reg_series.values.reshape(-1, 1)
    y_cls = cls_series.values
    
    # Create label encoder for classification target
    label_encoder = LabelEncoder()
    y_cls = label_encoder.fit_transform(y_cls)
    
    # Handle stratification and grouping
    if group_col is not None:
        # Create a DataFrame with all relevant information
        split_df = pd.DataFrame({
            'group': group_col,
            'strata': stratify_col if stratify_col is not None else None
        })
        
        # Get unique groups and their most common strata
        group_strata = split_df.groupby('group')['strata'].agg(
            lambda x: pd.Series.mode(x)[0] if stratify_col is not None else None
        )
        
        # Split groups while preserving strata distribution
        train_groups, test_groups = train_test_split(
            group_strata.index,
            test_size=test_split,
            stratify=group_strata.values if stratify_col is not None else None,
            random_state=42
        )
        
        # Create masks for the original data based on groups
        train_mask = np.isin(group_col, train_groups)
        test_mask = np.isin(group_col, test_groups)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
        y_cls_train, y_cls_test = y_cls[train_mask], y_cls[test_mask]
    else:
        # If no grouping, perform regular stratified split
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
            X, y_reg, y_cls,
            test_size=test_split,
            stratify=stratify_col if stratify_col is not None else None,
            random_state=42
        )
    
    # Apply PCA
    pca = PCA(n_components=num_pcs)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_pca),
        torch.FloatTensor(y_reg_train),
        torch.LongTensor(y_cls_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_pca),
        torch.FloatTensor(y_reg_test),
        torch.LongTensor(y_cls_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, pca, label_encoder

def train_model_mtl(
    train_loader: DataLoader,
    model_params: Dict,
    num_epochs: int = 20,
    device: str = 'cuda',
    val_split: float = 0.2,
    use_optuna: bool = False,
    n_trials: int = 100
) -> Tuple[nn.Module, Dict, plt.Figure]:
    """
    Train the MTL model with optional Optuna hyperparameter tuning
    """
    # Split training data into train and validation
    dataset = train_loader.dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader_split = DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)

    # Calculate class weights from the training data
    all_labels = []
    for _, _, y_cls in train_loader.dataset:
        all_labels.append(y_cls.item())
    
    # Compute class weights
    class_counts = torch.bincount(torch.tensor(all_labels))
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()  # normalize
    class_weights = class_weights.to(device)
    
    def train_with_params_mtl(lr, weight_decay):
        model = MTLNormalizedTransformer(**model_params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        regression_criterion = nn.L1Loss()
        classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        best_val_loss = float('inf')
        best_model_state = None
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_accuracy': []}
        
        pbar = tqdm(range(num_epochs), desc='Training')
        for epoch in pbar:
            # Training phase
            model.train()
            train_loss = 0.0
            
            for x_batch, y_reg_batch, y_cls_batch in train_loader_split:
                x_batch = x_batch.to(device)
                y_reg_batch = y_reg_batch.to(device)
                y_cls_batch = y_cls_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                total_loss, _, _ = compute_mtl_loss(
                    outputs, y_reg_batch, y_cls_batch,
                    regression_criterion, classification_criterion
                )
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds_reg = []
            val_true_reg = []
            val_preds_cls = []
            val_true_cls = []
            
            with torch.no_grad():
                for x_batch, y_reg_batch, y_cls_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_reg_batch = y_reg_batch.to(device)
                    y_cls_batch = y_cls_batch.to(device)
                    
                    outputs = model(x_batch)
                    total_loss, _, _ = compute_mtl_loss(
                        outputs, y_reg_batch, y_cls_batch,
                        regression_criterion, classification_criterion
                    )
                    val_loss += total_loss.item()
                    
                    val_preds_reg.extend(outputs['regression_output'].cpu().numpy())
                    val_true_reg.extend(y_reg_batch.cpu().numpy())
                    val_preds_cls.extend(outputs['classification_output'].argmax(1).cpu().numpy())
                    val_true_cls.extend(y_cls_batch.cpu().numpy())
            
            # Calculate metrics
            val_mae = mean_absolute_error(val_true_reg, val_preds_reg)
            val_accuracy = accuracy_score(val_true_cls, val_preds_cls)
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader_split))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_mae'].append(val_mae)
            history['val_accuracy'].append(val_accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                'val_mae': f'{val_mae:.4f}',
                'val_acc': f'{val_accuracy:.4f}'
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        model.load_state_dict(best_model_state)
        return model, history, best_val_loss
    
    if use_optuna:
        def objective(trial):
            lr = trial.suggest_float('lr', 1e-3, 5e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1, log=True)
            # optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
            _, _, best_val_loss = train_with_params_mtl(lr, weight_decay)
            return best_val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        final_model, final_history, _ = train_with_params_mtl(
            best_params['lr'],
            best_params['weight_decay']
        )
    else:
        final_model, final_history, _ = train_with_params_mtl(1e-3, 1e-4)
    
    # Create training plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Loss plot
    ax1.plot(final_history['train_loss'], label='Train Loss')
    ax1.plot(final_history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics plot
    ax2.plot(final_history['val_mae'], label='MAE')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(final_history['val_accuracy'], label='Accuracy', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2_twin.set_ylabel('Accuracy')
    ax2.set_title('Validation Metrics')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return final_model, final_history, fig

def predict_and_evaluate_mtl(
    model: nn.Module,
    test_loader: DataLoader,
    label_encoder: LabelEncoder,
    device: str = 'cuda'
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], plt.Figure]:
    """
    Evaluate model on test data and generate predictions with performance metrics
    Uses original class labels for visualization
    """
    model.eval()
    y_reg_true = []
    y_reg_pred = []
    y_cls_true = []
    y_cls_pred = []
    
    with torch.no_grad():
        for x_batch, y_reg_batch, y_cls_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            
            y_reg_true.extend(y_reg_batch.numpy())
            y_reg_pred.extend(outputs['regression_output'].cpu().numpy())
            y_cls_true.extend(y_cls_batch.numpy())
            y_cls_pred.extend(outputs['classification_output'].argmax(1).cpu().numpy())
    
    # Convert to arrays
    y_reg_true = np.array(y_reg_true)
    y_reg_pred = np.array(y_reg_pred)
    y_cls_true = np.array(y_cls_true)
    y_cls_pred = np.array(y_cls_pred)
    
    # Convert numeric classes back to original labels
    y_cls_true_labels = label_encoder.inverse_transform(y_cls_true)
    y_cls_pred_labels = label_encoder.inverse_transform(y_cls_pred)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_reg_true, y_reg_pred),
        'r2': r2_score(y_reg_true, y_reg_pred),
        'accuracy': accuracy_score(y_cls_true, y_cls_pred)
    }
    
    predictions = {
        'reg_true': y_reg_true,
        'reg_pred': y_reg_pred,
        'cls_true': y_cls_true_labels,  # Store original labels
        'cls_pred': y_cls_pred_labels,  # Store original labels
        'cls_true_encoded': y_cls_true, # Also store encoded versions if needed
        'cls_pred_encoded': y_cls_pred
    }
    
    # Create evaluation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Regression scatter plot
    ax1.scatter(y_reg_true, y_reg_pred, alpha=0.5)
    ax1.plot([y_reg_true.min(), y_reg_true.max()], 
             [y_reg_true.min(), y_reg_true.max()], 
             'r--', label='Perfect Prediction')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Regression Results\nMAE: {metrics["mae"]:.3f}, R²: {metrics["r2"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Classification confusion matrix with original labels
    cm = pd.crosstab(
        pd.Series(y_cls_true_labels, name='True'),
        pd.Series(y_cls_pred_labels, name='Predicted')
    )
    
    # Create heatmap with better formatting
    sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap='Blues',
                cbar_kws={'label': 'Count'})
    
    # Rotate labels for better readability
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    ax2.set_xlabel('Predicted Class')
    ax2.set_ylabel('True Class')
    ax2.set_title(f'Classification Results\nAccuracy: {metrics["accuracy"]:.3f}')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return metrics, predictions, fig
