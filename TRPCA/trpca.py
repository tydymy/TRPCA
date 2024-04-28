import math
import torch
import torch.nn as nn

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
