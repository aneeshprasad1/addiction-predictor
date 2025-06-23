#!/usr/bin/env python3
"""
AWS Setup for Large EEG Data Processing

Simple setup for using AWS S3 for storage and EC2 for processing.
"""

import os
from pathlib import Path

def create_aws_script():
    """Create AWS setup and processing scripts."""
    
    aws_script = '''#!/usr/bin/env python3
"""
AWS S3 + EC2 Setup for LEMON EEG Processing
"""

import boto3
import os
from pathlib import Path

# AWS Configuration
S3_BUCKET = "your-lemon-eeg-bucket"  # Create this bucket
REGION = "us-east-1"

def setup_s3_bucket():
    """Create S3 bucket for EEG data."""
    s3 = boto3.client('s3', region_name=REGION)
    
    try:
        s3.create_bucket(
            Bucket=S3_BUCKET,
            CreateBucketConfiguration={'LocationConstraint': REGION}
        )
        print(f"Created S3 bucket: {S3_BUCKET}")
    except Exception as e:
        print(f"Bucket might already exist: {e}")

def upload_to_s3(local_file, s3_key):
    """Upload file to S3."""
    s3 = boto3.client('s3')
    
    try:
        s3.upload_file(local_file, S3_BUCKET, s3_key)
        print(f"Uploaded {local_file} to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"Upload failed: {e}")

def download_from_s3(s3_key, local_file):
    """Download file from S3."""
    s3 = boto3.client('s3')
    
    try:
        s3.download_file(S3_BUCKET, s3_key, local_file)
        print(f"Downloaded s3://{S3_BUCKET}/{s3_key} to {local_file}")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    setup_s3_bucket()
    print("AWS S3 setup complete!")
'''
    
    with open('aws_eeg_processing.py', 'w') as f:
        f.write(aws_script)
    
    print("Created AWS processing script")

def create_simple_processing_script():
    """Create a simple processing script that works with cloud storage."""
    
    simple_script = '''#!/usr/bin/env python3
"""
Simple EEG Processing for Cloud Storage
Process one subject at a time to manage memory usage
"""

import os
import sys
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from tqdm import tqdm

def process_single_subject(subject_id, input_dir, output_dir):
    """Process a single subject's EEG data."""
    
    # Load EEG file
    eeg_files = list(Path(input_dir).glob(f"{subject_id}/eeg/*.edf"))
    
    if not eeg_files:
        print(f"No EEG file found for {subject_id}")
        return None
    
    eeg_file = eeg_files[0]
    
    try:
        # Load raw EEG
        raw = mne.io.read_raw_edf(eeg_file, preload=True)
        
        # Basic preprocessing
        raw.filter(0.5, 45.0)
        raw.notch_filter(50.0)
        raw.resample(250)  # Downsample to 250 Hz
        
        # Extract features (simplified)
        data = raw.get_data()
        
        # Simple feature extraction: mean power in frequency bands
        features = {
            'delta_power': np.mean(data[:, :int(4*250)]),  # 0-4 Hz
            'theta_power': np.mean(data[:, int(4*250):int(8*250)]),  # 4-8 Hz
            'alpha_power': np.mean(data[:, int(8*250):int(13*250)]),  # 8-13 Hz
            'beta_power': np.mean(data[:, int(13*250):int(30*250)]),  # 13-30 Hz
        }
        
        # Save features
        output_file = Path(output_dir) / f"{subject_id}_features.npz"
        np.savez_compressed(output_file, **features)
        
        print(f"Processed {subject_id}: {features}")
        return features
        
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return None

def main():
    """Process all subjects."""
    
    # Configuration
    input_dir = "data/bids"  # or "/content/drive/MyDrive/lemon_eeg_data/raw" for Colab
    output_dir = "data/processed"  # or "/content/drive/MyDrive/lemon_eeg_data/processed" for Colab
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of subjects
    subjects = [d.name for d in Path(input_dir).iterdir() if d.name.startswith('sub-')]
    
    print(f"Processing {len(subjects)} subjects...")
    
    # Process each subject
    for subject in tqdm(subjects):
        process_single_subject(subject, input_dir, output_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
'''
    
    with open('simple_process.py', 'w') as f:
        f.write(simple_script)
    
    print("Created simple processing script")

if __name__ == "__main__":
    create_aws_script()
    create_simple_processing_script()
    print("\\n=== CLOUD PROCESSING OPTIONS ===")
    print("\\n1. GOOGLE COLAB (Easiest):")
    print("   - Free GPU access")
    print("   - 15GB Google Drive storage")
    print("   - No setup required")
    print("   - Perfect for research")
    print("\\n2. AWS S3 + EC2 (Professional):")
    print("   - Scalable storage")
    print("   - GPU instances available")
    print("   - Pay-as-you-go")
    print("   - Good for production")
    print("\\n3. LOCAL + EXTERNAL DRIVE:")
    print("   - Use external SSD")
    print("   - Process in batches")
    print("   - Keep raw data on external drive")
    print("\\nRecommendation: Start with Google Colab!") 