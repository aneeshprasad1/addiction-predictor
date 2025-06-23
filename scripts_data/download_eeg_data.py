#!/usr/bin/env python3
"""
Download EEG data from LEMON dataset FTP server
"""

import os
import requests
import urllib.parse
from pathlib import Path
import time
from tqdm import tqdm
import re

# FTP server base URL
BASE_URL = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"

def get_subject_list():
    """Get list of available subjects from the FTP server"""
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        
        # Parse the directory listing to find subject folders
        subjects = []
        for line in response.text.split('\n'):
            if 'sub-' in line and 'href' in line:
                # Extract subject ID from href
                start = line.find('sub-')
                end = line.find('/', start)
                if end == -1:
                    end = line.find('"', start)
                if end != -1:
                    subject = line[start:end]
                    subjects.append(subject)
        
        return sorted(list(set(subjects)))
    except Exception as e:
        print(f"Error getting subject list: {e}")
        return []

def find_eeg_files_recursive(base_url, current_path=""):
    """Recursively find EEG files in nested directories"""
    eeg_files = []
    
    try:
        url = f"{base_url}/{current_path}" if current_path else base_url
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
                    
                    # Check if it's an EEG file or related file
                    if item.lower().endswith(('.edf', '.fif', '.bdf', '.set', '.eeg', '.vhdr', '.vmrk')):
                        full_path = f"{current_path}/{item}" if current_path else item
                        eeg_files.append(full_path)
                    
                    # Recursively search subdirectories (but not too deep)
                    elif not item.startswith('sub-') and not '.' in item:  # Likely a subdirectory
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

def download_subject_eeg(subject_id, output_dir):
    """Download EEG data for a specific subject"""
    subject_url = f"{BASE_URL}/{subject_id}"
    
    try:
        print(f"Searching for EEG files in {subject_id}...")
        
        # Recursively find all EEG files for this subject
        eeg_files = find_eeg_files_recursive(subject_url)
        
        if not eeg_files:
            print(f"No EEG files found for {subject_id}")
            return False
        
        print(f"Found {len(eeg_files)} EEG-related files for {subject_id}")
        
        # Create output directory
        subject_dir = Path(output_dir) / subject_id / "eeg"
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Download each EEG file
        successful_downloads = 0
        for file_path in eeg_files:
            # Extract just the filename from the path
            filename = Path(file_path).name
            
            file_url = f"{subject_url}/{file_path}"
            output_file = subject_dir / filename
            
            if output_file.exists():
                print(f"File already exists: {output_file}")
                successful_downloads += 1
                continue
            
            print(f"Downloading {filename} for {subject_id}...")
            
            if download_file(file_url, output_file):
                successful_downloads += 1
                print(f"Successfully downloaded {filename}")
        
        print(f"Downloaded {successful_downloads}/{len(eeg_files)} files for {subject_id}")
        return successful_downloads > 0
        
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return False

def main():
    """Main download function"""
    # Create output directory
    output_dir = Path("data/bids")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Getting list of available subjects...")
    subjects = get_subject_list()
    
    if not subjects:
        print("No subjects found. Please check the FTP server.")
        return
    
    print(f"Found {len(subjects)} subjects")
    
    # Get subjects that have IAT scores
    iat_file = Path("data/phenotype/IAT.tsv")
    if iat_file.exists():
        with open(iat_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            iat_subjects = [line.split('\t')[0] for line in lines if line.strip() and 'n/a' not in line]
        print(f"Subjects with IAT scores: {len(iat_subjects)}")
        
        # Filter to subjects with both IAT and EEG data
        available_subjects = [s for s in subjects if s in iat_subjects]
        print(f"Subjects with both IAT and EEG data: {len(available_subjects)}")
    else:
        available_subjects = subjects
        print("IAT file not found, downloading all subjects")
    
    # Download EEG data for each subject
    successful_downloads = 0
    
    for i, subject in enumerate(available_subjects, 1):
        print(f"\n[{i}/{len(available_subjects)}] Processing {subject}")
        
        if download_subject_eeg(subject, output_dir):
            successful_downloads += 1
        
        # Add a small delay to be respectful to the server
        time.sleep(1)
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded EEG data for {successful_downloads}/{len(available_subjects)} subjects")
    print(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    main() 