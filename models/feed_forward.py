import torch
import torch.nn as nn
import torch.nn.functional as F

class CeLU(nn.Module):
    """
    Continuously Differentiable Exponential Linear Unit (CeLU) activation function
    as described in the paper.
    """
    def __init__(self, alpha=1.0):
        super(CeLU, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        """
        CeLU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor after applying CeLU activation
        """
        return torch.max(torch.zeros_like(x), x) + torch.min(
            torch.zeros_like(x),
            self.alpha * (torch.exp(x / self.alpha) - 1)
        )

class FeedForward(nn.Module):
    """
    Feed-Forward Network with CeLU activation as described in the paper
    """
    def __init__(self, d_model, d_ff, dropout=0.1, alpha=1.0):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.celu = CeLU(alpha=alpha)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass of Feed-Forward Network
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor after passing through FFN
        """
        # Apply residual connection
        residual = x
        
        # First linear layer
        x = self.linear1(x)
        
        # Apply CeLU activation
        x = self.celu(x)
        
        # Apply dropout
        x = self.dropout1(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        # Apply dropout
        x = self.dropout2(x)
        
        # Add residual connection and normalize
        x = self.norm(x + residual)
        
        return x 