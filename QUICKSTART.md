# 🚀 Quick Start Guide

This guide will help you get the LEMON EEG CNN project up and running quickly.

## Prerequisites

- Python 3.8 or higher
- Git
- (Optional) Docker
- (Optional) CUDA-capable GPU for faster training

## 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd ia-predictor

# Setup the project (installs dependencies and creates directories)
make setup
```

## 2. Data Preparation

### Option A: Use Synthetic Data (for testing)
The preprocessing script includes synthetic data generation for testing purposes. No additional setup needed.

### Option B: Use Real LEMON Data
1. Download the LEMON dataset from [MPI CBS](https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/)
2. Organize it in BIDS format in `data/bids/`
3. Ensure you have a `participants.tsv` file with IAT scores

## 3. Quick Test

Test that everything is working:

```bash
# Test model architecture
make test-model

# Check data availability
make check-data
```

## 4. Run the Pipeline

### Option A: Full Pipeline (Recommended)
```bash
# Run the complete pipeline: preprocess → train → evaluate
make pipeline
```

### Option B: Step by Step
```bash
# 1. Preprocess EEG data
make preprocess

# 2. Train the CNN model
make train

# 3. Evaluate the model
make evaluate
```

## 5. Monitor Training

The training script automatically logs to Weights & Biases. To view:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Set your entity in `configs/default_config.yaml`
3. Training metrics will be logged automatically

## 6. View Results

After training, you'll find:
- Model checkpoint: `artifacts/models/best_model.pth`
- Evaluation results: `artifacts/results/evaluation_results.pkl`
- W&B logs: `wandb/` directory

## 7. Development

### Start Jupyter Notebook
```bash
make jupyter
```

### Run Quality Checks
```bash
make quality
```

### Clean Generated Files
```bash
make clean
```

## 8. Docker (Optional)

If you prefer using Docker:

```bash
# Build the Docker image
make docker-build

# Run the container
make docker-run
```

## Configuration

The main configuration is in `configs/default_config.yaml`. Key settings:

- **Data paths**: Update paths to your data
- **Model parameters**: Adjust CNN architecture
- **Training parameters**: Learning rate, batch size, etc.
- **W&B settings**: Your project and entity names

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've run `make setup`
2. **CUDA errors**: Set `device: "cpu"` in config for CPU-only training
3. **Memory errors**: Reduce batch size in config
4. **W&B errors**: Set `entity: null` in config to skip W&B logging

### Getting Help

- Check the full README.md for detailed documentation
- Run `make help` to see all available commands
- Check logs in the terminal for error messages

## Expected Results

With the default configuration, you should expect:
- **Accuracy**: ~85-90% (depending on data quality)
- **Training time**: 10-30 minutes (CPU), 2-5 minutes (GPU)
- **Model size**: ~2-5 MB

## Next Steps

1. **Experiment**: Try different model architectures
2. **Hyperparameter tuning**: Use W&B sweeps
3. **Data augmentation**: Add more preprocessing steps
4. **Interpretability**: Analyze model decisions
5. **Deployment**: Convert to production model

## Support

For issues and questions:
- Check the README.md for detailed documentation
- Review the code comments for implementation details
- Open an issue on the project repository

---

**Happy coding! 🧠⚡** 