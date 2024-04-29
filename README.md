# Readme: TRPCA

## Run in Google Colab
Access the notebook [here](https://drive.google.com/file/d/1HoLVNs7WaQwUKwb-KLntuEz1Nq9gCYEg/view?usp=sharing).

## Input/Output
**Input:** `pd.DataFrame`, `pd.Series`

**Output:**
- Loss curves for train/test
- Regression of predictions vs. actuals or classification confusion matrix
- Best trained model checkpoint

## Usage Instructions
1. Load `.biom`/`.qza` and metadata with the same index.
2. Remove unwanted samples (i.e., samples with NA metadata).
3. Apply RPCA or CLR+PCA transformation.
4. Convert sample loadings or PCA results to a DataFrame.
5. Add metadata to the DataFrame.
6. Load DataFrame and metadata into train/test dataloaders.

## Dependencies
- biom-format
- pytorch
- pandas
- numpy
- sklearn
- matplotlib
- seaborn
- tqdm

## Note
- Compatible with qiime2 environment with Gemelli installed for RPCA.
- Apple Silicon Macs: MPS compatible PyTorch install may break qiime2 environment. This issue might not occur with CPU-only or CUDA-enabled devices.

## Parameters
### Descriptions and Recommended Values
| Parameter             | Description                                      | Recommended Values                               |
|-----------------------|--------------------------------------------------|--------------------------------------------------|
| `n_dimensions`        | Number of principal components; must be a power of 2 | 8, 16, 32, 64, 128, 256, 512                     |
| `epochs`              | Number of times the model iterates through the training data | 1000+ (observe loss curves for diagnosis)        |
| `learning_rate`       | Size of step the model takes to adjust based on what it has learned | 1e-03, 1e-04, 1e-05                              |
| `batch_size`          | Number of samples per batch                     | 32, 64, 128, 256, 512                             |
| `num_transformer_layers` | Number of transformer encoder layers          | 1, 3, 6, 12                                       |
| `nhead`               | Number of attention heads                       | 4, 8, 16                                          |
| `dim_feedforward`     | Layer size for feedforward layer                | 1024, 2048                                        |

### Parameters Based on Sample Size
- **<100 samples**
  - `{'n_dimensions': 8-32, 'learning_rate': 1e-05, 'batch_size': 16, 'num_transformer_layers': 1-3, 'nhead': 4-8, 'dim_feedforward': 1024}`
- **100-1000 samples**
  - `{'n_dimensions': 8-256, 'learning_rate': 1e-03, 'batch_size': 128, 'num_transformer_layers': 1-3, 'nhead': 8-16, 'dim_feedforward': 1024-2048}`
- **1000+ samples**
  - `{'n_dimensions': 8-512, 'learning_rate': 1e-03, 'batch_size': 128-512, 'num_transformer_layers': 3-12, 'nhead': 8-16, 'dim_feedforward': 1024-2048}`
