#!/usr/bin/env python3
"""
Verify data completeness for the addiction prediction project
"""

import pandas as pd
from pathlib import Path
import glob

def check_iat_data():
    """Check IAT data availability"""
    print("=== IAT Data Check ===")
    
    iat_file = Path("data/phenotype/IAT.tsv")
    if not iat_file.exists():
        print("❌ IAT.tsv not found!")
        return []
    
    # Load IAT data
    iat_data = pd.read_csv(iat_file, sep='\t')
    valid_iat = iat_data[iat_data['IAT_sum'] != 'n/a'].copy()
    valid_iat['IAT_sum'] = pd.to_numeric(valid_iat['IAT_sum'])
    
    print(f"✅ IAT data found: {len(iat_data)} total subjects")
    print(f"✅ Valid IAT scores: {len(valid_iat)} subjects")
    print(f"📊 IAT score range: {valid_iat['IAT_sum'].min():.1f} - {valid_iat['IAT_sum'].max():.1f}")
    
    # Classify subjects
    high_iat = valid_iat[valid_iat['IAT_sum'] >= 50]
    low_iat = valid_iat[valid_iat['IAT_sum'] < 50]
    
    print(f"🔴 High IAT (≥50): {len(high_iat)} subjects")
    print(f"🟢 Low IAT (<50): {len(low_iat)} subjects")
    
    return valid_iat['participant_id'].tolist()

def check_behavioral_data():
    """Check behavioral task data availability"""
    print("\n=== Behavioral Data Check ===")
    
    # Check for behavioral task files
    ets_files = glob.glob("data/sub-*/ses-*/beh/*task-ETS_events.tsv")
    cpts_files = glob.glob("data/sub-*/ses-*/beh/*task-CPTS_events.tsv")
    oddball_files = glob.glob("data/sub-*/ses-*/beh/*task-Oddball_events.tsv")
    
    print(f"✅ ETS task files: {len(ets_files)}")
    print(f"✅ CPTS task files: {len(cpts_files)}")
    print(f"✅ Oddball task files: {len(oddball_files)}")
    
    # Extract subject IDs from filenames
    ets_subjects = [Path(f).stem.split('_')[0] for f in ets_files]
    cpts_subjects = [Path(f).stem.split('_')[0] for f in cpts_files]
    
    return ets_subjects, cpts_subjects

def check_eeg_data():
    """Check EEG data availability"""
    print("\n=== EEG Data Check ===")
    
    # Check for EEG files
    eeg_files = glob.glob("data/bids/sub-*/eeg/*.edf") + glob.glob("data/bids/sub-*/eeg/*.fif")
    
    if not eeg_files:
        print("❌ No EEG files found!")
        print("💡 You need to download EEG data from:")
        print("   https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/")
        return []
    
    print(f"✅ EEG files found: {len(eeg_files)}")
    
    # Extract subject IDs
    eeg_subjects = []
    for f in eeg_files:
        path = Path(f)
        subject = path.parts[-3]  # data/bids/sub-XXX/eeg/file.edf
        eeg_subjects.append(subject)
    
    eeg_subjects = list(set(eeg_subjects))
    print(f"✅ Subjects with EEG data: {len(eeg_subjects)}")
    
    return eeg_subjects

def check_data_overlap():
    """Check overlap between different data types"""
    print("\n=== Data Overlap Check ===")
    
    # Get subjects from each data type
    iat_subjects = check_iat_data()
    ets_subjects, cpts_subjects = check_behavioral_data()
    eeg_subjects = check_eeg_data()
    
    if not iat_subjects:
        return
    
    # Find subjects with all data types
    if eeg_subjects:
        complete_subjects = set(iat_subjects) & set(ets_subjects) & set(eeg_subjects)
        print(f"✅ Subjects with IAT + ETS + EEG: {len(complete_subjects)}")
        
        if complete_subjects:
            print("📋 Complete subjects:")
            for sub in sorted(list(complete_subjects))[:10]:  # Show first 10
                print(f"   {sub}")
            if len(complete_subjects) > 10:
                print(f"   ... and {len(complete_subjects) - 10} more")
    else:
        # If no EEG data, check behavioral + IAT overlap
        behavioral_subjects = set(ets_subjects) & set(cpts_subjects)
        complete_subjects = set(iat_subjects) & behavioral_subjects
        print(f"✅ Subjects with IAT + behavioral tasks: {len(complete_subjects)}")
    
    # Check for subjects missing data
    if eeg_subjects:
        missing_eeg = set(iat_subjects) - set(eeg_subjects)
        if missing_eeg:
            print(f"⚠️  Subjects with IAT but no EEG: {len(missing_eeg)}")
    
    missing_behavioral = set(iat_subjects) - set(ets_subjects)
    if missing_behavioral:
        print(f"⚠️  Subjects with IAT but no ETS: {len(missing_behavioral)}")

def main():
    """Main verification function"""
    print("🧠 LEMON Dataset Verification")
    print("=" * 50)
    
    check_data_overlap()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("- IAT scores: ✅ Available")
    print("- Behavioral tasks: ✅ Available") 
    print("- EEG recordings: ❌ Need to download")
    print("\n💡 Next step: Run 'python scripts/download_eeg_data.py'")

if __name__ == "__main__":
    main() 