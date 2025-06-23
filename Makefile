# Makefile for LEMON EEG CNN Project

.PHONY: help install setup preprocess train evaluate test clean docker-build docker-run jupyter

# Default target
help:
	@echo "LEMON EEG CNN Project - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install     - Install Python dependencies"
	@echo "  setup       - Complete project setup (install + create dirs)"
	@echo ""
	@echo "Data Processing:"
	@echo "  preprocess  - Run EEG preprocessing pipeline"
	@echo ""
	@echo "Model Training:"
	@echo "  train       - Train the CNN model"
	@echo "  evaluate    - Evaluate trained model"
	@echo ""
	@echo "Development:"
	@echo "  test        - Run tests"
	@echo "  jupyter     - Start Jupyter notebook server"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       - Clean generated files"
	@echo "  help        - Show this help message"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Complete project setup
setup: install
	@echo "Setting up project directories..."
	mkdir -p data/raw data/processed data/bids
	mkdir -p notebooks scripts models configs wandb
	@echo "✅ Project setup completed!"

# Run preprocessing
preprocess:
	@echo "Running EEG preprocessing pipeline..."
	python pipeline/preprocess.py
	@echo "✅ Preprocessing completed!"

# Train model
train:
	@echo "Training CNN model..."
	python pipeline/train.py
	@echo "✅ Training completed!"

# Evaluate model
evaluate:
	@echo "Evaluating trained model..."
	python pipeline/evaluate.py
	@echo "✅ Evaluation completed!"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests completed!"

# Start Jupyter notebook server
jupyter:
	@echo "Starting Jupyter notebook server..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t lemon-eeg-cnn .
	@echo "✅ Docker image built successfully!"

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/wandb:/app/wandb \
		-p 8888:8888 \
		lemon-eeg-cnn
	@echo "✅ Docker container stopped!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/*
	rm -rf wandb/*
	rm -f *.pth
	rm -f *.pkl
	rm -f *.npz
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	@echo "✅ Cleanup completed!"

# Full pipeline (preprocess -> train -> evaluate)
pipeline: preprocess train evaluate
	@echo "✅ Full pipeline completed!"

# Development setup with additional tools
dev-setup: setup
	@echo "Installing development dependencies..."
	pip install -e ".[dev,notebooks]"
	@echo "✅ Development setup completed!"

# Quick test of model architecture
test-model:
	@echo "Testing model architecture..."
	python -c "from models.cnn_model import create_model; import yaml; config = yaml.safe_load(open('configs/default_config.yaml')); model = create_model(config); print('✅ Model created successfully!')"

# Check data availability
check-data:
	@echo "Checking data availability..."
	@if [ -d "data/bids" ]; then \
		echo "✅ BIDS data directory found"; \
		ls -la data/bids/; \
	else \
		echo "⚠️  BIDS data directory not found"; \
		echo "Please download the LEMON dataset and organize it in BIDS format"; \
	fi

# Generate project documentation
docs:
	@echo "Generating project documentation..."
	@if command -v pdoc3 >/dev/null 2>&1; then \
		pdoc3 --html --output-dir docs models/ scripts/; \
		echo "✅ Documentation generated in docs/"; \
	else \
		echo "⚠️  pdoc3 not found. Install with: pip install pdoc3"; \
	fi

# Format code
format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black .; \
		echo "✅ Code formatted with black"; \
	else \
		echo "⚠️  black not found. Install with: pip install black"; \
	fi

# Lint code
lint:
	@echo "Linting code..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 .; \
		echo "✅ Code linting completed"; \
	else \
		echo "⚠️  flake8 not found. Install with: pip install flake8"; \
	fi

# Type checking
type-check:
	@echo "Running type checks..."
	@if command -v mypy >/dev/null 2>&1; then \
		mypy models/ scripts/; \
		echo "✅ Type checking completed"; \
	else \
		echo "⚠️  mypy not found. Install with: pip install mypy"; \
	fi

# Quality checks
quality: format lint type-check
	@echo "✅ All quality checks completed!"

# Show project status
status:
	@echo "Project Status:"
	@echo "==============="
	@echo "Python version: $(shell python --version)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "MNE version: $(shell python -c 'import mne; print(mne.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Data directories:"
	@ls -la data/ 2>/dev/null || echo "data/ directory not found"
	@echo ""
	@echo "Model files:"
	@ls -la *.pth 2>/dev/null || echo "No model files found"
	@echo ""
	@echo "Configuration:"
	@ls -la configs/ 2>/dev/null || echo "configs/ directory not found" 