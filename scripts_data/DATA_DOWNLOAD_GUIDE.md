# 📥 LEMON EEG Dataset Download Guide

## Overview

This guide will help you download and prepare the LEMON (Leipzig Mind‑Brain‑Body) EEG dataset for the internet addiction classification project.

## 🎯 What You Need

For this project, you need:
- **EEG Data**: 62-channel resting-state EEG recordings
- **Participant Info**: IAT (Internet Addiction Test) scores to classify subjects
- **Metadata**: BIDS-compliant dataset description and participant information

## 📋 Step-by-Step Download Instructions

### Option 1: Direct Download from MPI CBS (Recommended)

1. **Visit the official LEMON dataset page**:
   ```
   https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/
   ```

2. **Download the following files**:
   - `participants.tsv` - Contains participant information and IAT scores
   - `dataset_description.json` - BIDS dataset metadata
   - EEG data files (`.edf` or `.fif` format) for the 49 subjects
   - `README` file for additional information

3. **Organize the files**:
   ```
   data/
   ├── bids/
   │   ├── participants.tsv
   │   ├── dataset_description.json
   │   └── sub-*/ (EEG data folders)
   └── raw/ (backup of original files)
   ```

### Option 2: OpenNeuro (Alternative)

1. **Visit OpenNeuro**:
   ```
   https://openneuro.org/datasets/ds000221/versions/1.0.0
   ```

2. **Download the dataset** using the OpenNeuro interface

3. **Extract and organize** as shown above

## 🔍 Identifying Internet Addiction Subjects

The key file you need is `participants.tsv`, which contains:
- `participant_id`: Subject identifier
- `IAT_score`: Internet Addiction Test score
- Other demographic and behavioral data

**Classification criteria** (based on the original paper):
- **High IAT (Internet Addicted)**: IAT score ≥ 50
- **Low IAT (Control)**: IAT score < 50

## 🚀 Quick Start

1. **Run the download script**:
   ```bash
   python scripts/download_data.py
   ```

2. **Follow the manual download instructions** if automated download fails

3. **Verify your data**:
   ```bash
   python scripts/download_data.py
   ```

## 📁 Expected File Structure

After download, your `data/` directory should look like:

```
data/
├── bids/
│   ├── participants.tsv
│   ├── dataset_description.json
│   ├── sub-001/
│   │   └── eeg/
│   │       └── sub-001_task-rest_eeg.edf
│   ├── sub-002/
│   │   └── eeg/
│   │       └── sub-002_task-rest_eeg.edf
│   └── ... (more subjects)
├── raw/ (backup of original files)
└── processed/ (will be created during preprocessing)
```

## ⚠️ Important Notes

- **Dataset Size**: ~2GB total
- **Download Time**: 10-30 minutes depending on connection
- **Storage**: Ensure you have at least 5GB free space
- **Format**: BIDS-compliant EEG data (.edf or .fif format)
- **Subjects**: Focus on the 49 subjects with IAT scores

## 🔧 Troubleshooting

### Common Issues:

1. **Download fails**: Try using a different browser or download manager
2. **File corruption**: Re-download the problematic files
3. **Missing files**: Ensure all required files are downloaded
4. **Permission errors**: Check file permissions and disk space

### Verification Commands:

```bash
# Check if required files exist
ls -la data/bids/participants.tsv
ls -la data/bids/dataset_description.json

# Count number of subject directories
ls data/bids/ | grep "sub-" | wc -l

# Check file sizes
du -sh data/
```

## 📞 Support

If you encounter issues:
1. Check the LEMON dataset documentation
2. Review the OpenNeuro dataset page
3. Check file permissions and disk space
4. Ensure stable internet connection

## 🎯 Next Steps

Once data is downloaded:
1. Run preprocessing: `python scripts/preprocess.py`
2. Start training: `python scripts/train.py`
3. Monitor with Weights & Biases

---

**Happy downloading! 🧠📊** 