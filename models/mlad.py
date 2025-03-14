import torch
import torch.nn as nn
import torch.nn.functional as F

from models.alpha_entmax import SparseAttention
from models.feed_forward import FeedForward
from models.gmm import GaussianMixtureModel

class SparseTransformerLayer(nn.Module):
    """
    Transformer layer with sparse attention
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, alpha=1.5):
        super(SparseTransformerLayer, self).__init__()
        
        # Multi-head sparse attention
        self.self_attn = MultiHeadSparseAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of Transformer layer
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            
        Returns:
            output: Transformed tensor
        """
        # Self-attention with residual connection and normalization
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward with residual connection and normalization
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x

class MultiHeadSparseAttention(nn.Module):
    """
    Multi-head attention with sparse attention mechanism
    """
    def __init__(self, d_model, n_heads, dropout=0.1, alpha=1.5):
        super(MultiHeadSparseAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
        # Sparse attention for each head
        self.sparse_attention = SparseAttention(
            d_model=self.d_k,
            alpha=alpha,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Attention mask
            
        Returns:
            output: Attention output
            attn_weights: Attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply sparse attention to each head
        heads = []
        attn_weights = []
        
        for i in range(self.n_heads):
            head_output, head_attn = self.sparse_attention(
                q[:, i], k[:, i], v[:, i], mask
            )
            heads.append(head_output)
            attn_weights.append(head_attn)
        
        # Concatenate heads and apply final linear projection
        heads = torch.cat([h.unsqueeze(1) for h in heads], dim=1)
        heads = heads.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.output_linear(heads)
        
        # Average attention weights across heads
        attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1)
        
        return output, attn_weights

class MLAD(nn.Module):
    """
    Multi-system Log Anomaly Detection (MLAD) model
    
    This model combines a Transformer with a Gaussian Mixture Model (GMM)
    for anomaly detection in system logs.
    """
    def __init__(self, d_model=100, n_heads=4, n_layers=2, d_ff=256, 
                 dropout=0.1, alpha=1.5, n_components=5):
        super(MLAD, self).__init__()
        
        self.d_model = d_model
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            SparseTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                alpha=alpha
            )
            for _ in range(n_layers)
        ])
        
        # Gaussian Mixture Model for anomaly detection
        self.gmm = GaussianMixtureModel(
            n_components=n_components,
            n_features=d_model
        )
        
        # Classification head
        self.classifier = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, update_gmm=True):
        """
        Forward pass of MLAD model
        
        Args:
            x: Input tensor [batch_size, d_model]
            update_gmm: Whether to update GMM parameters
            
        Returns:
            predictions: Anomaly predictions
            energy: Sample energy (higher indicates anomaly)
            hidden: Hidden vectors from Transformer
        """
        batch_size = x.size(0)
        
        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Pass through Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Get hidden vectors (use mean if sequence input)
        if x.size(1) > 1:
            hidden = torch.mean(x, dim=1)
        else:
            hidden = x.squeeze(1)
        
        # Pass through GMM
        y_pred, energy = self.gmm(hidden)
        
        # Update GMM parameters during training
        if self.training and update_gmm:
            self.gmm.update_parameters(hidden, y_pred)
        
        # Classification prediction
        logits = self.classifier(hidden)
        predictions = self.sigmoid(logits)
        
        return predictions, energy, hidden
    
    def compute_loss(self, predictions, labels, hidden, energy=None):
        """
        Compute combined loss for training
        
        Args:
            predictions: Model predictions
            labels: True labels
            hidden: Hidden vectors
            energy: Sample energy (if None, computed from hidden)
            
        Returns:
            loss: Combined loss value
        """
        # Classification loss
        classification_loss = F.binary_cross_entropy(
            predictions.squeeze(-1), labels
        )
        
        # GMM losses
        energy_loss, cov_loss = self.gmm.compute_loss(hidden)
        
        # Combined loss
        loss = classification_loss + energy_loss + cov_loss
        
        return loss
    
    def predict(self, x, threshold=None):
        """
        Predict anomalies based on energy threshold
        
        Args:
            x: Input tensor
            threshold: Energy threshold (if None, use classifier)
            
        Returns:
            predictions: Binary anomaly predictions
            scores: Anomaly scores
        """
        self.eval()
        with torch.no_grad():
            _, energy, _ = self.forward(x, update_gmm=False)
            
            if threshold is None:
                # Use classifier predictions
                predictions, _, _ = self.forward(x, update_gmm=False)
                predictions = (predictions > 0.5).float()
            else:
                # Use energy threshold
                predictions = (energy > threshold).float()
            
            return predictions, energy 