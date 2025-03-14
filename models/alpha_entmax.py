import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaEntmax(nn.Module):
    """
    Implementation of alpha-entmax as described in the paper:
    "Sparse Sequence-to-Sequence Models" (Peters et al., 2019)
    
    Alpha-entmax is a generalization of softmax that can produce sparse probability distributions.
    When alpha=1, it is equivalent to softmax. When alpha=2, it is sparsemax.
    """
    def __init__(self, alpha=1.5, dim=-1, n_iter=50, eps=1e-8):
        super(AlphaEntmax, self).__init__()
        self.alpha = alpha
        self.dim = dim
        self.n_iter = n_iter
        self.eps = eps
        
    def forward(self, x):
        """
        Forward pass of alpha-entmax
        
        Args:
            x: Input tensor
            
        Returns:
            Sparse probability distribution
        """
        if self.alpha == 1.0:
            # If alpha=1, use standard softmax
            return F.softmax(x, dim=self.dim)
        elif self.alpha == 2.0:
            # If alpha=2, use sparsemax
            return self._sparsemax(x)
        else:
            # Otherwise, use general alpha-entmax
            return self._alpha_entmax(x)
    
    def _sparsemax(self, x):
        """
        Sparsemax function (alpha=2 case)
        """
        dim = self.dim
        if dim == -1:
            dim = len(x.shape) - 1
            
        # Move dim to last position for easier manipulation
        x_sorted, _ = torch.sort(x, dim=dim, descending=True)
        
        # Calculate cumulative sum
        cum_sum = torch.cumsum(x_sorted, dim=dim)
        
        # Calculate threshold indices
        rho = torch.arange(1, x.shape[dim] + 1, device=x.device)
        threshold = x_sorted - (cum_sum - 1) / rho
        
        # Find largest index where threshold is positive
        rho_threshold = (threshold > 0).sum(dim=dim, keepdim=True)
        
        # Get the threshold value
        threshold_value = torch.take_along_dim(
            threshold, rho_threshold - 1, dim=dim
        )
        
        # Apply threshold to get sparse probabilities
        p = torch.clamp(x - threshold_value, min=0)
        
        return p
    
    def _alpha_entmax(self, x):
        """
        General alpha-entmax function for 1 < alpha < 2
        
        This uses the bisection method for finding the optimal threshold
        """
        dim = self.dim
        if dim == -1:
            dim = len(x.shape) - 1
        
        # Move dim to last position for easier manipulation
        x_orig = x
        x = x.transpose(dim, -1)
        
        # Compute max for numerical stability
        max_val, _ = x.max(dim=-1, keepdim=True)
        x = x - max_val
        
        # Compute tau (threshold) via bisection
        tau_left = x.max(dim=-1, keepdim=True)[0]
        tau_right = tau_left - 1.0 / (1.0 - self.alpha)
        
        for _ in range(self.n_iter):
            tau_mid = (tau_left + tau_right) / 2.0
            p = torch.clamp(x - tau_mid, min=0) ** (1.0 / (self.alpha - 1.0))
            sum_p = p.sum(dim=-1, keepdim=True)
            err = sum_p - 1.0
            
            # Update bounds
            tau_left = torch.where(err > 0, tau_mid, tau_left)
            tau_right = torch.where(err <= 0, tau_mid, tau_right)
            
            # Check convergence
            if torch.all(torch.abs(err) < self.eps):
                break
        
        # Compute final probabilities
        p = torch.clamp(x - tau_mid, min=0) ** (1.0 / (self.alpha - 1.0))
        
        # Restore original dimensions
        p = p.transpose(dim, -1)
        
        return p

class SparseAttention(nn.Module):
    """
    Sparse attention mechanism using alpha-entmax instead of softmax
    """
    def __init__(self, d_model, alpha=1.5, dropout=0.1):
        super(SparseAttention, self).__init__()
        
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.alpha_entmax = AlphaEntmax(alpha=alpha)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of sparse attention
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional mask tensor
            
        Returns:
            context vectors
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply alpha-entmax instead of softmax
        attn_weights = self.alpha_entmax(scores)
        attn_weights = self.dropout(attn_weights)
        
        # Get context vectors
        context = torch.matmul(attn_weights, v)
        
        return context, attn_weights 