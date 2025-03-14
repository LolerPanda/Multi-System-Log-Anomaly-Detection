import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianMixtureModel(nn.Module):
    """
    Gaussian Mixture Model component for unsupervised anomaly detection
    as described in the paper.
    """
    def __init__(self, n_components, n_features, lambda_energy=0.1, lambda_cov=-0.005, eps=1e-8):
        super(GaussianMixtureModel, self).__init__()
        
        self.n_components = n_components
        self.n_features = n_features
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.eps = eps
        
        # Initialize GMM parameters
        # Each component has a prior (phi), mean (mu), and covariance matrix (cov)
        self.register_parameter("phi", nn.Parameter(torch.ones(n_components) / n_components))
        self.register_parameter("mu", nn.Parameter(torch.randn(n_components, n_features)))
        
        # Initialize covariance matrices as identity
        eye = torch.eye(n_features).unsqueeze(0).repeat(n_components, 1, 1)
        self.register_parameter("cov", nn.Parameter(eye))
        
        # For numerical stability when inverting covariance matrices
        self.register_buffer("eps_tensor", torch.eye(n_features) * self.eps)
        
    def forward(self, h):
        """
        Forward pass of GMM component
        
        Args:
            h: Hidden vectors from the Transformer layers
            
        Returns:
            y_pred: Component probabilities for each sample
            energy: Sample energy (higher indicates anomaly)
        """
        # Compute component probabilities for each sample
        y_pred = self._compute_probs(h)
        
        # Compute sample energy
        energy = self._compute_energy(h)
        
        return y_pred, energy
    
    def _compute_probs(self, h):
        """
        Compute probability that each sample belongs to each Gaussian component
        
        Args:
            h: Hidden vectors [batch_size, n_features]
            
        Returns:
            probs: Component probabilities [batch_size, n_components]
        """
        batch_size = h.size(0)
        
        # Calculate negative Mahalanobis distance for each component
        # (h - mu)^T * Sigma^-1 * (h - mu)
        probs = torch.zeros(batch_size, self.n_components, device=h.device)
        
        for k in range(self.n_components):
            # Center data
            centered_data = h - self.mu[k]
            
            # Ensure covariance matrix is positive definite
            cov_matrix = self.cov[k] + self.eps_tensor
            
            # Calculate Mahalanobis distance
            inv_cov = torch.inverse(cov_matrix)
            mahalanobis = torch.sum(
                torch.matmul(centered_data, inv_cov) * centered_data,
                dim=1
            )
            
            # Calculate log determinant of covariance
            log_det = torch.logdet(cov_matrix)
            
            # Calculate log probability (unnormalized)
            log_prob = -0.5 * (mahalanobis + log_det + self.n_features * math.log(2 * math.pi))
            
            probs[:, k] = log_prob + torch.log(self.phi[k] + self.eps)
        
        # Normalize log probabilities using log-sum-exp trick
        max_probs, _ = torch.max(probs, dim=1, keepdim=True)
        probs = probs - max_probs
        probs = torch.exp(probs)
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        
        return probs
    
    def update_parameters(self, h, probs):
        """
        Update GMM parameters using EM algorithm
        
        Args:
            h: Hidden vectors [batch_size, n_features]
            probs: Component probabilities [batch_size, n_components]
            
        Returns:
            Updated GMM parameters
        """
        batch_size = h.size(0)
        
        # E-step: Already completed by computing probs
        
        # M-step: Update parameters based on probabilities
        
        # Update component weights (phi)
        component_weights = torch.mean(probs, dim=0)
        self.phi.data = component_weights
        
        # Update means (mu)
        for k in range(self.n_components):
            # Weight samples by their probability of belonging to component k
            weighted_sum = torch.sum(probs[:, k].unsqueeze(1) * h, dim=0)
            self.mu.data[k] = weighted_sum / (torch.sum(probs[:, k]) + self.eps)
        
        # Update covariance matrices (cov)
        for k in range(self.n_components):
            # Center data around component mean
            centered_data = h - self.mu[k]
            
            # Calculate weighted outer products
            weighted_outer_prods = torch.zeros_like(self.cov[k])
            for i in range(batch_size):
                outer = torch.outer(centered_data[i], centered_data[i])
                weighted_outer_prods += probs[i, k] * outer
            
            # Update covariance matrix
            self.cov.data[k] = weighted_outer_prods / (torch.sum(probs[:, k]) + self.eps)
            
            # Add small diagonal term for numerical stability
            self.cov.data[k] += torch.eye(self.n_features, device=h.device) * self.eps
    
    def _compute_energy(self, h):
        """
        Compute sample energy as negative log likelihood
        
        Args:
            h: Hidden vectors [batch_size, n_features]
            
        Returns:
            energy: Sample energy (higher indicates anomaly)
        """
        # Compute component probabilities
        probs = self._compute_probs(h)
        
        # Calculate weighted likelihood
        weighted_likelihood = torch.zeros(h.size(0), device=h.device)
        
        for k in range(self.n_components):
            # Center data
            centered_data = h - self.mu[k]
            
            # Ensure covariance matrix is positive definite
            cov_matrix = self.cov[k] + self.eps_tensor
            
            # Calculate Mahalanobis distance
            inv_cov = torch.inverse(cov_matrix)
            mahalanobis = torch.sum(
                torch.matmul(centered_data, inv_cov) * centered_data,
                dim=1
            )
            
            # Calculate log determinant
            log_det = torch.logdet(cov_matrix)
            
            # Calculate log probability density
            log_prob = -0.5 * (mahalanobis + log_det + self.n_features * math.log(2 * math.pi))
            
            # Add weighted log probability
            weighted_likelihood += self.phi[k] * torch.exp(log_prob)
        
        # Energy is negative log likelihood
        energy = -torch.log(weighted_likelihood + self.eps)
        
        return energy
    
    def compute_loss(self, h, y_true=None):
        """
        Compute GMM loss components
        
        Args:
            h: Hidden vectors
            y_true: True labels (optional)
            
        Returns:
            energy_loss: Energy loss term
            cov_loss: Covariance regularization term
        """
        # Compute energy
        energy = self._compute_energy(h)
        
        # Energy loss
        energy_loss = torch.mean(energy)
        
        # Covariance regularization term
        cov_loss = 0
        for k in range(self.n_components):
            cov_loss += torch.norm(self.cov[k], p='fro')
        cov_loss = cov_loss / self.n_components
        
        return self.lambda_energy * energy_loss, self.lambda_cov * cov_loss
    
    def estimate_threshold(self, train_energy, contamination=0.01):
        """
        Estimate threshold for anomaly detection based on training data energy
        
        Args:
            train_energy: Energy values from training data
            contamination: Expected proportion of anomalies (default 0.01)
            
        Returns:
            threshold: Energy threshold for anomaly detection
        """
        # Sort energy values
        sorted_energy = torch.sort(train_energy)[0]
        
        # Find threshold based on contamination level
        threshold_idx = int((1 - contamination) * len(sorted_energy))
        threshold = sorted_energy[threshold_idx]
        
        return threshold 