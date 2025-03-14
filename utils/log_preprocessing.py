import os
import re
import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

class LogPreprocessor:
    def __init__(self, window_size=20, seed=42):
        """
        Initialize log preprocessor
        
        Args:
            window_size: Size of sliding window for BGL and Thunderbird datasets
            seed: Random seed for reproducibility
        """
        self.window_size = window_size
        self.seed = seed
        
        # Initialize template miner with default config
        config = TemplateMinerConfig()
        config.masking_instructions = []
        config.drain_extra_delimiters = [",", "=", ":", ";"]
        config.max_clusters = 2000
        self.template_miner = TemplateMiner(config=config)
        
        # Initialize sentence transformer model
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def parse_logs(self, log_file, dataset_type):
        """
        Parse logs and extract templates
        
        Args:
            log_file: Path to the log file
            dataset_type: Type of dataset ('BGL', 'HDFS', 'Thunderbird')
            
        Returns:
            DataFrame with extracted templates and labels
        """
        print(f"Parsing logs for dataset: {dataset_type}")
        
        if dataset_type == 'HDFS':
            return self._parse_hdfs(log_file)
        elif dataset_type in ['BGL', 'Thunderbird']:
            return self._parse_with_window(log_file, dataset_type)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def _parse_hdfs(self, log_file):
        """Parse HDFS logs using block IDs"""
        logs = []
        block_dict = {}
        
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(r'blk_-?\d+', line)
                if match:
                    blk_id = match.group(0)
                    if blk_id not in block_dict:
                        block_dict[blk_id] = []
                    
                    # Extract template
                    result = self.template_miner.add_log_message(line)
                    template = result["template_mined"]
                    block_dict[blk_id].append(template)
        
        # Convert to sequences
        for blk_id, templates in block_dict.items():
            # Check if anomalous based on specific patterns for HDFS
            # This is simplified - in practice, you'd use the provided labels
            is_anomaly = 'Error' in ' '.join(templates) or 'Exception' in ' '.join(templates)
            logs.append({
                'sequence': templates,
                'is_anomaly': 1 if is_anomaly else 0,
                'block_id': blk_id
            })
        
        return pd.DataFrame(logs)
    
    def _parse_with_window(self, log_file, dataset_type):
        """Parse BGL or Thunderbird logs using sliding window"""
        logs = []
        buffer = []
        
        with open(log_file, 'r') as f:
            for line in f:
                # Extract template
                result = self.template_miner.add_log_message(line)
                template = result["template_mined"]
                
                # Check if line is anomalous based on specific patterns
                # This is simplified - in practice, you'd use the provided labels
                is_anomaly = False
                if dataset_type == 'BGL':
                    is_anomaly = 'error' in line.lower() or 'fail' in line.lower()
                elif dataset_type == 'Thunderbird':
                    is_anomaly = 'error' in line.lower() or 'exception' in line.lower()
                
                buffer.append((template, 1 if is_anomaly else 0))
                
                # Process buffer when it reaches window size
                if len(buffer) >= self.window_size:
                    templates = [item[0] for item in buffer]
                    # If any line in window is anomalous, mark window as anomalous
                    is_window_anomaly = any(item[1] == 1 for item in buffer)
                    
                    logs.append({
                        'sequence': templates,
                        'is_anomaly': 1 if is_window_anomaly else 0
                    })
                    
                    # Non-overlapping window
                    buffer = []
        
        # Process any remaining logs
        if buffer:
            templates = [item[0] for item in buffer]
            is_window_anomaly = any(item[1] == 1 for item in buffer)
            
            logs.append({
                'sequence': templates,
                'is_anomaly': 1 if is_window_anomaly else 0
            })
        
        return pd.DataFrame(logs)
    
    def prepare_datasets(self, df):
        """
        Prepare training and testing datasets
        
        Args:
            df: DataFrame with processed logs
            
        Returns:
            train_data, test_data: Training and testing datasets
        """
        # Split normal logs
        normal_df = df[df['is_anomaly'] == 0]
        abnormal_df = df[df['is_anomaly'] == 1]
        
        # Create test set with all abnormal logs and equal number of normal logs
        n_abnormal = len(abnormal_df)
        normal_test_df = normal_df.sample(n=min(n_abnormal, len(normal_df)), random_state=self.seed)
        
        # Rest of normal logs go to training
        normal_train_df = normal_df.drop(normal_test_df.index)
        
        # Combine datasets
        train_data = normal_train_df
        test_data = pd.concat([normal_test_df, abnormal_df])
        
        print(f"Training data: {len(train_data)} sequences (all normal)")
        print(f"Testing data: {len(test_data)} sequences ({len(normal_test_df)} normal, {len(abnormal_df)} anomalous)")
        
        return train_data, test_data
    
    def encode_sequences(self, sequences):
        """
        Encode log sequences using Sentence-BERT
        
        Args:
            sequences: List of log template sequences
            
        Returns:
            Encoded sequences as numpy array
        """
        # Join templates in each sequence to create sentences
        sentences = [' '.join(seq) for seq in sequences]
        
        # Encode sentences
        encodings = self.sentence_model.encode(sentences, show_progress_bar=True)
        
        return encodings

def load_and_preprocess(dataset_name, data_dir='./data', window_size=20):
    """
    Load and preprocess log dataset
    
    Args:
        dataset_name: Name of dataset ('BGL', 'HDFS', 'Thunderbird')
        data_dir: Directory containing dataset files
        window_size: Size of sliding window for BGL and Thunderbird
        
    Returns:
        train_data, test_data: Training and testing datasets with encoded features
    """
    preprocessor = LogPreprocessor(window_size=window_size)
    
    # Set paths based on dataset name
    if dataset_name == 'BGL':
        log_file = os.path.join(data_dir, 'BGL', 'BGL.log')
    elif dataset_name == 'HDFS':
        log_file = os.path.join(data_dir, 'HDFS', 'HDFS.log')
    elif dataset_name == 'Thunderbird':
        log_file = os.path.join(data_dir, 'Thunderbird', 'Thunderbird.log')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Parse logs
    df = preprocessor.parse_logs(log_file, dataset_name)
    
    # Prepare datasets
    train_df, test_df = preprocessor.prepare_datasets(df)
    
    # Encode sequences
    train_features = preprocessor.encode_sequences(train_df['sequence'].tolist())
    test_features = preprocessor.encode_sequences(test_df['sequence'].tolist())
    
    # Create final datasets
    train_data = {
        'features': train_features,
        'labels': np.zeros(len(train_df)),  # All normal
        'sequences': train_df['sequence'].tolist()
    }
    
    test_data = {
        'features': test_features,
        'labels': test_df['is_anomaly'].values,
        'sequences': test_df['sequence'].tolist()
    }
    
    return train_data, test_data 