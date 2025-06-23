#!/usr/bin/env python3
"""
Test script to download EEG data for a single subject
"""

import requests
import re
from pathlib import Path
from tqdm import tqdm

# FTP server base URL
BASE_URL = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"

def find_eeg_files_recursive(base_url, current_path=""):
    """Recursively find EEG files in nested directories"""
    eeg_files = []
    
    try:
        url = f"{base_url}/{current_path}" if current_path else base_url
        print(f"Searching: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse directory listing
        for line in response.text.split('\n'):
            if 'href="' in line:
                # Extract filename/directory from href
                match = re.search(r'href="([^"]+)"', line)
                if match:
                    item = match.group(1)
                    
                    # Skip parent directory links
                    if item in ['../', './']:
                        continue
                    
                    # Remove trailing slash for directories
                    if item.endswith('/'):
                        item = item[:-1]
                    
                    print(f"  Found: {item}")
                    
                    # Check if it's an EEG file or related file
                    if item.lower().endswith(('.edf', '.fif', '.bdf', '.set', '.eeg', '.vhdr', '.vmrk')):
                        full_path = f"{current_path}/{item}" if current_path else item
                        eeg_files.append(full_path)
                        print(f"    -> EEG file: {full_path}")
                    
                    # Recursively search subdirectories (but not too deep)
                    elif not item.startswith('sub-') and not '.' in item:  # Likely a subdirectory
                        print(f"    -> Entering subdirectory: {item}")
                        sub_path = f"{current_path}/{item}" if current_path else item
                        sub_files = find_eeg_files_recursive(base_url, sub_path)
                        eeg_files.extend(sub_files)
        
        return eeg_files
        
    except Exception as e:
        print(f"Error searching {current_path}: {e}")
        return []

def download_file(file_url, output_file):
    """Download a single file with progress bar"""
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        filename = Path(output_file).name
        
        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        if output_file.exists():
            output_file.unlink()
        return False

def main():
    """Test with a single subject"""
    test_subject = "sub-010005"
    subject_url = f"{BASE_URL}/{test_subject}"
    
    print(f"Testing recursive search and download for {test_subject}")
    print("=" * 50)
    
    eeg_files = find_eeg_files_recursive(subject_url)
    
    print("\n" + "=" * 50)
    print(f"Found {len(eeg_files)} EEG-related files:")
    for file_path in eeg_files:
        print(f"  {file_path}")
    
    # Test download
    print("\n" + "=" * 50)
    print("Testing download...")
    
    # Create test output directory
    output_dir = Path("data/test_download")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_downloads = 0
    for file_path in eeg_files:
        filename = Path(file_path).name
        file_url = f"{subject_url}/{file_path}"
        output_file = output_dir / filename
        
        print(f"\nDownloading {filename}...")
        if download_file(file_url, output_file):
            successful_downloads += 1
            print(f"✅ Successfully downloaded {filename}")
        else:
            print(f"❌ Failed to download {filename}")
    
    print(f"\nDownload test complete: {successful_downloads}/{len(eeg_files)} files downloaded")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main() 