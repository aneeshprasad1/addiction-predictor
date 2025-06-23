#!/usr/bin/env python3
"""
Google Colab Setup for Large EEG Data Processing

This script helps set up Google Colab for processing the LEMON EEG dataset.
Upload this to Google Drive and run in Colab for cloud-based processing.
"""

import os
import sys
from pathlib import Path

def create_colab_notebook():
    """Create a Colab notebook for EEG processing."""
    
    notebook_content = '''# LEMON EEG Dataset Processing - Google Colab
# Run this cell to mount Google Drive and install dependencies

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install mne mne-bids pandas numpy scikit-learn matplotlib seaborn

# Clone your repository (replace with your actual repo URL)
!git clone https://github.com/yourusername/addiction-predictor.git
%cd addiction-predictor

# Create data directories in Google Drive
!mkdir -p /content/drive/MyDrive/lemon_eeg_data/raw
!mkdir -p /content/drive/MyDrive/lemon_eeg_data/processed
!mkdir -p /content/drive/MyDrive/lemon_eeg_data/models

print("Setup complete! Your data will be stored in Google Drive.")
'''
    
    with open('lemon_eeg_colab_setup.py', 'w') as f:
        f.write(notebook_content)
    
    print("Created colab setup script. Upload this to Google Drive and run in Colab.")

def create_download_script():
    """Create a script to download data directly to Google Drive."""
    
    download_script = '''#!/usr/bin/env python3
"""
Download LEMON EEG data directly to Google Drive
Run this in Google Colab for cloud-based data storage
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import time

# Google Drive paths
DRIVE_ROOT = "/content/drive/MyDrive/lemon_eeg_data"
RAW_DATA_PATH = f"{DRIVE_ROOT}/raw"

def download_subject_eeg(subject_id, max_retries=3):
    """Download EEG data for a single subject to Google Drive."""
    
    # Create subject directory
    subject_dir = Path(RAW_DATA_PATH) / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Download EEG file (simplified - you'll need to adapt this to your actual data source)
    eeg_url = f"https://your-eeg-data-source.com/{subject_id}/eeg.edf"
    output_file = subject_dir / "eeg.edf"
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {subject_id} (attempt {attempt + 1})")
            
            response = requests.get(eeg_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Successfully downloaded {subject_id}")
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {subject_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retry
    
    return False

def main():
    """Download all subjects to Google Drive."""
    
    # Get list of subjects (you'll need to adapt this)
    subjects = ["sub-010001", "sub-010002", "sub-010004"]  # Add your subjects
    
    print(f"Downloading {len(subjects)} subjects to Google Drive...")
    
    successful = 0
    for subject in subjects:
        if download_subject_eeg(subject):
            successful += 1
        time.sleep(2)  # Be respectful to the server
    
    print(f"Download complete: {successful}/{len(subjects)} subjects")

if __name__ == "__main__":
    main()
'''
    
    with open('download_to_drive.py', 'w') as f:
        f.write(download_script)
    
    print("Created download script for Google Drive.")

if __name__ == "__main__":
    create_colab_notebook()
    create_download_script()
    print("\\n=== GOOGLE COLAB SETUP ===")
    print("1. Upload these scripts to Google Drive")
    print("2. Open Google Colab (colab.research.google.com)")
    print("3. Mount your Google Drive")
    print("4. Run the setup script")
    print("5. Download data directly to Drive")
    print("6. Process data in Colab with GPU acceleration")
    print("\\nBenefits:")
    print("- 15GB free storage in Google Drive")
    print("- GPU acceleration available")
    print("- No local storage needed")
    print("- Easy sharing and collaboration") 