import os
import argparse
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from models.mlad import MLAD
from utils.log_preprocessing import load_and_preprocess
from utils.data_loader import create_data_loaders

def evaluate_model(model, test_loader, device, threshold=None):
    """
    Evaluate MLAD model on test data
    
    Args:
        model: MLAD model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cpu or cuda)
        threshold: Energy threshold for anomaly detection
        
    Returns:
        precision, recall, f1, accuracy, confusion matrix, energies, true labels
    """
    # Set model to evaluation mode
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_energies = []
    
    with torch.no_grad():
        for features, labels, _ in test_loader:
            # Move data to device
            features = features.to(device)
            
            # Get predictions
            if threshold is None:
                # Use classifier predictions
                predictions, energies, _ = model(features, update_gmm=False)
                binary_preds = (predictions.squeeze(-1) > 0.5).float()
            else:
                # Use energy threshold
                _, energies, _ = model(features, update_gmm=False)
                binary_preds = (energies > threshold).float()
            
            # Collect labels, predictions, and energies
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(binary_preds.cpu().numpy())
            all_energies.extend(energies.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_energies = np.array(all_energies)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return precision, recall, f1, accuracy, cm, all_energies, all_labels

def visualize_results(energies, labels, save_path=None):
    """
    Visualize energy distribution for normal and anomalous samples
    
    Args:
        energies: Sample energies
        labels: True labels
        save_path: Path to save visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Get energies for normal and anomalous samples
    normal_energies = energies[labels == 0]
    anomaly_energies = energies[labels == 1]
    
    # Plot histograms
    plt.hist(normal_energies, bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(anomaly_energies, bins=50, alpha=0.7, label='Anomaly', density=True)
    
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.title('Energy Distribution for Normal and Anomalous Samples')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_embeddings(model, test_loader, device, save_path=None):
    """
    Visualize sample embeddings using t-SNE
    
    Args:
        model: MLAD model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cpu or cuda)
        save_path: Path to save visualization
    """
    # Extract embeddings and labels
    embeddings = []
    labels = []
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for features, true_labels, _ in test_loader:
            # Move data to device
            features = features.to(device)
            
            # Forward pass
            pred_labels, _, hidden = model(features, update_gmm=False)
            
            # Collect embeddings and labels
            embeddings.append(hidden.cpu().numpy())
            labels.append(true_labels.cpu().numpy())
            predictions.append((pred_labels.squeeze(-1) > 0.5).float().cpu().numpy())
    
    # Concatenate batches
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    
    # Subsample if there are too many points (for faster t-SNE)
    max_samples = 2000
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        predictions = predictions[indices]
    
    # Apply t-SNE dimensionality reduction
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot based on true labels
    plt.subplot(2, 1, 1)
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    plt.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1], 
                c='blue', marker='o', label='Normal (True)', alpha=0.7)
    plt.scatter(embeddings_2d[anomaly_mask, 0], embeddings_2d[anomaly_mask, 1], 
                c='red', marker='^', label='Anomaly (True)', alpha=0.7)
    
    plt.title('t-SNE Visualization of Log Embeddings (True Labels)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot based on predicted labels
    plt.subplot(2, 1, 2)
    normal_pred_mask = predictions == 0
    anomaly_pred_mask = predictions == 1
    
    plt.scatter(embeddings_2d[normal_pred_mask, 0], embeddings_2d[normal_pred_mask, 1], 
                c='blue', marker='o', label='Normal (Predicted)', alpha=0.7)
    plt.scatter(embeddings_2d[anomaly_pred_mask, 0], embeddings_2d[anomaly_pred_mask, 1], 
                c='red', marker='^', label='Anomaly (Predicted)', alpha=0.7)
    
    plt.title('t-SNE Visualization of Log Embeddings (Predicted Labels)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        save_path: Path to save visualization
    """
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['Normal', 'Anomaly'])
    plt.yticks([0.5, 1.5], ['Normal', 'Anomaly'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate MLAD model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='BGL', choices=['BGL', 'HDFS', 'Thunderbird', 'BGL_Thunderbird'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for BGL and Thunderbird datasets')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--threshold', type=float, default=None, help='Energy threshold for anomaly detection')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get model parameters from checkpoint
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        model = MLAD(
            d_model=model_args['d_model'],
            n_heads=model_args['n_heads'],
            n_layers=model_args['n_layers'],
            d_ff=model_args['d_ff'],
            dropout=model_args['dropout'],
            alpha=model_args['alpha'],
            n_components=model_args['n_components']
        ).to(device)
    else:
        # Use default parameters
        model = MLAD().to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {args.model_path}")
    
    # Load and preprocess data
    print(f"Loading {args.dataset} dataset...")
    train_data, test_data = load_and_preprocess(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        window_size=args.window_size
    )
    
    # Create data loaders
    _, test_loader = create_data_loaders(
        train_data=train_data,
        test_data=test_data,
        batch_size=args.batch_size
    )
    
    print(f"Dataset loaded: {len(test_loader.dataset)} test samples")
    
    # Evaluate model
    precision, recall, f1, accuracy, cm, energies, labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=args.threshold
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Create output directory if needed
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, f"{args.dataset}_confusion_matrix.png")
        plot_confusion_matrix(cm, save_path=cm_path)
        
        # Visualize energy distribution
        energy_path = os.path.join(args.output_dir, f"{args.dataset}_energy_distribution.png")
        visualize_results(energies, labels, save_path=energy_path)
        
        # Visualize embeddings
        embeddings_path = os.path.join(args.output_dir, f"{args.dataset}_embeddings.png")
        visualize_embeddings(model, test_loader, device, save_path=embeddings_path)

if __name__ == '__main__':
    main() 