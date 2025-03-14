import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.mlad import MLAD
from utils.log_preprocessing import load_and_preprocess
from utils.data_loader import LogDataset, create_data_loaders

def train_model(train_loader, val_loader, model, optimizer, device, args):
    """
    Train the MLAD model
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: MLAD model
        optimizer: Optimizer
        device: Device to train on (cpu or cuda)
        args: Training arguments
        
    Returns:
        trained model
    """
    # Set model to training mode
    model.train()
    
    # Initialize best validation F1 score
    best_f1 = 0.0
    best_model_path = None
    
    print(f"Training MLAD model for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (features, labels, _) in enumerate(train_loader):
            # Move data to device
            features = features.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions, energy, hidden = model(features)
            
            # Compute loss
            loss = model.compute_loss(predictions, labels, hidden, energy)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
        
        # Average training loss
        train_loss /= len(train_loader)
        
        # Validate model
        precision, recall, f1, accuracy, val_loss = validate_model(val_loader, model, device)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            
            # Create directory if it doesn't exist
            os.makedirs(args.save_dir, exist_ok=True)
            
            # Save model
            best_model_path = os.path.join(args.save_dir, f"mlad_{args.dataset}_{args.alpha}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
            }, best_model_path)
            
            print(f"New best model saved! F1: {best_f1:.4f}")
    
    # Load best model
    if best_model_path is not None and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with F1: {checkpoint['best_f1']:.4f}")
    
    return model

def validate_model(val_loader, model, device):
    """
    Validate the MLAD model
    
    Args:
        val_loader: DataLoader for validation data
        model: MLAD model
        device: Device to validate on (cpu or cuda)
        
    Returns:
        precision, recall, f1, accuracy, validation loss
    """
    # Set model to evaluation mode
    model.eval()
    
    all_labels = []
    all_predictions = []
    val_loss = 0.0
    
    with torch.no_grad():
        for features, labels, _ in val_loader:
            # Move data to device
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions, energy, hidden = model(features, update_gmm=False)
            
            # Compute loss
            loss = model.compute_loss(predictions, labels, hidden, energy)
            val_loss += loss.item()
            
            # Binarize predictions
            binary_preds = (predictions.squeeze(-1) > 0.5).float()
            
            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(binary_preds.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Average validation loss
    val_loss /= len(val_loader)
    
    return precision, recall, f1, accuracy, val_loss

def main():
    parser = argparse.ArgumentParser(description='Train MLAD model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='BGL', choices=['BGL', 'HDFS', 'Thunderbird', 'BGL_Thunderbird'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for BGL and Thunderbird datasets')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=100, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=256, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=1.5, help='Alpha parameter for entmax')
    parser.add_argument('--n_components', type=int, default=5, help='Number of Gaussian components')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_gmm', action='store_true', help='Disable GMM component')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print(f"Loading {args.dataset} dataset...")
    train_data, test_data = load_and_preprocess(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        window_size=args.window_size
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data=train_data,
        test_data=test_data,
        batch_size=args.batch_size
    )
    
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    
    # Initialize model
    model = MLAD(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        alpha=args.alpha,
        n_components=args.n_components
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    trained_model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        args=args
    )
    
    print("Training completed!")
    
    # Test model
    precision, recall, f1, accuracy, _ = validate_model(val_loader, trained_model, device)
    print(f"Final model performance - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f"mlad_{args.dataset}_{args.alpha}_final.pt")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'args': vars(args),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }, final_model_path)
    
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    main() 