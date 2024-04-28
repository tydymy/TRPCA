import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA, IncrementalPCA

def apply_pca(df, n_components):
    pca = IncrementalPCA(n_components=n_components)
    # pca = PCA(n_components=n_components)
    return pca.fit_transform(df), pca

def rarefy(df, depth):
    # Ensuring all data is numeric; replacing non-numeric with NaN
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    rarefied_df = pd.DataFrame(0, index=df.index, columns=df.columns)
    
    for idx, row in df.iterrows():
        # Non-zero counts
        non_zero_counts = row[row > 0]
        
        if non_zero_counts.sum() >= depth:
            # Map non-zero indices to the column positions
            col_positions = [df.columns.get_loc(col) for col in non_zero_counts.index]
            
            # Sample based on the non-zero counts, using integer positions
            chosen_positions = np.random.choice(col_positions, size=depth, replace=True, 
                                                p=non_zero_counts.values / non_zero_counts.sum())
            
            # Generate the counts for chosen positions
            counts = np.bincount(chosen_positions, minlength=len(df.columns))
            
            # Assign the counts back to the DataFrame structure
            rarefied_df.loc[idx] = counts
        else:
            # Use original counts if sum is less than depth
            rarefied_df.loc[idx] = row.astype(int)
    
    return rarefied_df.astype(int)

def clr_transformation(df):
    # Replace zeros with a very small number to avoid issues with log(0)
    df_replaced_zeros = df.replace(0, 1e-6)
    
    # Calculate the geometric mean for each row
    geometric_mean = df_replaced_zeros.apply(lambda x: np.exp(np.mean(np.log(x))), axis=1)
    
    # Apply the CLR transformation
    clr_transformed = df_replaced_zeros.apply(lambda x: np.log(x / geometric_mean[x.name]), axis=1)
    
    return clr_transformed

def clr_transformation_optimized(df):
    df_replaced_zeros = df.replace(0, 1e-6)
    geometric_mean = np.exp(np.mean(np.log(df_replaced_zeros), axis=1))
    clr_transformed = np.log(df_replaced_zeros.div(geometric_mean, axis='index'))
    return clr_transformed
