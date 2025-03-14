import os
import argparse
import torch
import numpy as np
import random
import subprocess
import shutil

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directories(args):
    """Create necessary directories"""
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories for each dataset
    for dataset in ['BGL', 'HDFS', 'Thunderbird']:
        os.makedirs(os.path.join(args.data_dir, dataset), exist_ok=True)

def download_datasets(args):
    """
    Download datasets if they don't exist
    
    Note: This is a simplified placeholder. In practice, you would need to
    download the datasets from their respective sources, which might require
    authentication or specific procedures.
    """
    # Check if datasets already exist
    bgl_path = os.path.join(args.data_dir, 'BGL', 'BGL.log')
    hdfs_path = os.path.join(args.data_dir, 'HDFS', 'HDFS.log')
    tb_path = os.path.join(args.data_dir, 'Thunderbird', 'Thunderbird.log')
    
    # Print dataset status
    print("\nDataset Status:")
    print(f"BGL: {'✅ Available' if os.path.exists(bgl_path) else '❌ Not available'}")
    print(f"HDFS: {'✅ Available' if os.path.exists(hdfs_path) else '❌ Not available'}")
    print(f"Thunderbird: {'✅ Available' if os.path.exists(tb_path) else '❌ Not available'}")
    
    # Provide download instructions if datasets are missing
    if not all([os.path.exists(p) for p in [bgl_path, hdfs_path, tb_path]]):
        print("\nSome datasets are missing. Please download them manually:")
        print("- BGL and Thunderbird datasets: https://www.usenix.org/cfdr-data")
        print("- HDFS dataset: https://github.com/logpai/loghub")
        print("\nAfter downloading, place the log files in their respective directories.")

def train_model(args, dataset):
    """Train MLAD model on specified dataset"""
    print(f"\n{'='*50}")
    print(f"Training MLAD on {dataset} dataset")
    print(f"{'='*50}")
    
    # Define command
    cmd = [
        "python", "train.py",
        "--dataset", dataset,
        "--data_dir", args.data_dir,
        "--window_size", str(args.window_size),
        "--d_model", str(args.d_model),
        "--n_heads", str(args.n_heads),
        "--n_layers", str(args.n_layers),
        "--d_ff", str(args.d_ff),
        "--dropout", str(args.dropout),
        "--alpha", str(args.alpha),
        "--n_components", str(args.n_components),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--save_dir", args.save_dir,
        "--seed", str(args.seed)
    ]
    
    # Add no_gmm flag if specified
    if args.no_gmm:
        cmd.append("--no_gmm")
    
    # Run command
    process = subprocess.run(cmd, check=True)
    
    # Return model path
    return os.path.join(args.save_dir, f"mlad_{dataset}_{args.alpha}_final.pt")

def evaluate_model(args, dataset, model_path):
    """Evaluate MLAD model on specified dataset"""
    print(f"\n{'='*50}")
    print(f"Evaluating MLAD on {dataset} dataset")
    print(f"{'='*50}")
    
    # Define command
    cmd = [
        "python", "evaluate.py",
        "--dataset", dataset,
        "--data_dir", args.data_dir,
        "--window_size", str(args.window_size),
        "--model_path", model_path,
        "--batch_size", str(args.batch_size),
        "--output_dir", args.output_dir
    ]
    
    # Add visualization flag if specified
    if args.visualize:
        cmd.append("--visualize")
    
    # Add threshold if specified
    if args.threshold is not None:
        cmd.extend(["--threshold", str(args.threshold)])
    
    # Run command
    process = subprocess.run(cmd, check=True)

def run_transfer_learning(args, source_dataset, target_dataset, model_path):
    """Run transfer learning from source dataset to target dataset"""
    print(f"\n{'='*50}")
    print(f"Running transfer learning from {source_dataset} to {target_dataset}")
    print(f"{'='*50}")
    
    # Create transfer learning output directory
    transfer_dir = os.path.join(args.output_dir, f"transfer_{source_dataset}_to_{target_dataset}")
    os.makedirs(transfer_dir, exist_ok=True)
    
    # Define command
    cmd = [
        "python", "evaluate.py",
        "--dataset", target_dataset,
        "--data_dir", args.data_dir,
        "--window_size", str(args.window_size),
        "--model_path", model_path,
        "--batch_size", str(args.batch_size),
        "--output_dir", transfer_dir
    ]
    
    # Add visualization flag if specified
    if args.visualize:
        cmd.append("--visualize")
    
    # Run command
    process = subprocess.run(cmd, check=True)

def run_alpha_ablation(args, dataset):
    """Run ablation study for different alpha values"""
    print(f"\n{'='*50}")
    print(f"Running alpha ablation study on {dataset} dataset")
    print(f"{'='*50}")
    
    # Create ablation output directory
    ablation_dir = os.path.join(args.output_dir, f"ablation_alpha_{dataset}")
    os.makedirs(ablation_dir, exist_ok=True)
    
    # Define alpha values
    alpha_values = [1.0, 1.2, 1.5, 1.8]
    
    results = []
    
    # Train and evaluate model with different alpha values
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}")
        
        # Update alpha value
        args.alpha = alpha
        
        # Train model
        model_path = train_model(args, dataset)
        
        # Define evaluation command
        cmd = [
            "python", "evaluate.py",
            "--dataset", dataset,
            "--data_dir", args.data_dir,
            "--window_size", str(args.window_size),
            "--model_path", model_path,
            "--batch_size", str(args.batch_size),
            "--output_dir", os.path.join(ablation_dir, f"alpha_{alpha}")
        ]
        
        # Add visualization flag if specified
        if args.visualize:
            cmd.append("--visualize")
        
        # Run command
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract results
        output = process.stdout
        precision = float(output.split("Precision: ")[1].split("\n")[0])
        recall = float(output.split("Recall: ")[1].split("\n")[0])
        f1 = float(output.split("F1 Score: ")[1].split("\n")[0])
        accuracy = float(output.split("Accuracy: ")[1].split("\n")[0])
        
        results.append({
            "alpha": alpha,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        })
    
    # Print ablation results
    print("\nAlpha Ablation Results:")
    print(f"{'Alpha':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Accuracy':<10}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['alpha']:<10.1f} {result['precision']:<10.4f} {result['recall']:<10.4f} "
              f"{result['f1']:<10.4f} {result['accuracy']:<10.4f}")

def main():
    parser = argparse.ArgumentParser(description='MLAD: Multi-system Log Anomaly Detection')
    
    # Pipeline control
    parser.add_argument('--download_only', action='store_true', help='Only check/download datasets')
    parser.add_argument('--train_only', action='store_true', help='Only train models, skip evaluation')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate models, skip training')
    parser.add_argument('--transfer_learning', action='store_true', help='Run transfer learning experiments')
    parser.add_argument('--alpha_ablation', action='store_true', help='Run alpha ablation study')
    
    # Dataset parameters
    parser.add_argument('--datasets', nargs='+', default=['BGL', 'HDFS', 'Thunderbird'],
                        choices=['BGL', 'HDFS', 'Thunderbird', 'BGL_Thunderbird'],
                        help='Datasets to use')
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
    parser.add_argument('--no_gmm', action='store_true', help='Disable GMM component')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    
    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=None, help='Energy threshold for anomaly detection')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    create_directories(args)
    
    # Check/download datasets
    download_datasets(args)
    
    # Exit if only checking datasets
    if args.download_only:
        return
    
    # Initialize model paths dictionary
    model_paths = {}
    
    # Train models
    if not args.eval_only:
        for dataset in args.datasets:
            model_paths[dataset] = train_model(args, dataset)
    
    # Evaluate models
    if not args.train_only:
        for dataset in args.datasets:
            # If we didn't train the model, try to find the latest model
            if dataset not in model_paths or not os.path.exists(model_paths[dataset]):
                model_path = os.path.join(args.save_dir, f"mlad_{dataset}_{args.alpha}_final.pt")
                if not os.path.exists(model_path):
                    print(f"No model found for {dataset}. Skipping evaluation.")
                    continue
                model_paths[dataset] = model_path
            
            evaluate_model(args, dataset, model_paths[dataset])
    
    # Run transfer learning experiments
    if args.transfer_learning and 'BGL' in args.datasets and 'Thunderbird' in args.datasets:
        # BGL → Thunderbird
        if 'BGL' in model_paths:
            run_transfer_learning(args, 'BGL', 'Thunderbird', model_paths['BGL'])
        
        # Thunderbird → BGL
        if 'Thunderbird' in model_paths:
            run_transfer_learning(args, 'Thunderbird', 'BGL', model_paths['Thunderbird'])
    
    # Run alpha ablation study
    if args.alpha_ablation and args.datasets:
        run_alpha_ablation(args, args.datasets[0])

if __name__ == '__main__':
    main() 