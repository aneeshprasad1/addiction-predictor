import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class EEGCNN5Layer(nn.Module):
    """
    5-Layer CNN for EEG classification as described in the MDPI paper.
    Supports both spectral and raw time series inputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(EEGCNN5Layer, self).__init__()
        
        self.config = config
        self.input_channels = config['model']['input_channels']
        self.spectral_bins = config['model']['spectral_bins']
        self.conv_channels = config['model']['conv_channels']
        self.conv_kernel_sizes = config['model']['conv_kernel_sizes']
        self.pool_kernel_sizes = config['model']['pool_kernel_sizes']
        self.dropout_rate = config['model']['dropout_rate']
        self.spectral_mode = config['preprocessing']['spectral_mode']
        
        # Calculate input dimensions
        if self.spectral_mode:
            # Spectral mode: (batch, channels, frequency_bins)
            self.input_height = self.input_channels  # 62 channels
            self.input_width = self.spectral_bins   # 64 frequency bins
        else:
            # Raw time series mode: (batch, channels, time_points)
            window_length = config['data']['window_length']
            sampling_rate = config['data']['sampling_rate']
            self.input_height = self.input_channels  # 62 channels
            self.input_width = int(window_length * sampling_rate)  # time points
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        in_channels = 1  # Start with 1 channel (we'll reshape input)
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(self.conv_channels, self.conv_kernel_sizes, self.pool_kernel_sizes)
        ):
            # Convolutional layer
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
                padding='same'
            )
            self.conv_layers.append(conv)
            
            # Batch normalization
            bn = nn.BatchNorm2d(out_channels)
            self.batch_norms.append(bn)
            
            # Pooling layer
            pool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))
            self.pool_layers.append(pool)
            
            in_channels = out_channels
        
        # Calculate flattened size after convolutions
        self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # Binary classification
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_flattened_size(self):
        """Calculate the size of flattened features after convolutions."""
        # Create a dummy input to calculate output size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.input_height, self.input_width)
            
            x = dummy_input
            for conv, bn, pool in zip(self.conv_layers, self.batch_norms, self.pool_layers):
                x = conv(x)
                x = bn(x)
                x = F.relu(x)
                x = pool(x)
            
            self.flattened_size = x.numel()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_points) for raw mode
               or (batch_size, channels, freq_bins) for spectral mode
        
        Returns:
            Output tensor of shape (batch_size, 2) for binary classification
        """
        # Reshape input: (batch, channels, time/freq) -> (batch, 1, channels, time/freq)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        for conv, bn, pool in zip(self.conv_layers, self.batch_norms, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create the appropriate model based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized model
    """
    architecture = config['model']['architecture']
    
    if architecture == "cnn_5layer":
        return EEGCNN5Layer(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    import yaml
    
    # Load config
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    if config['preprocessing']['spectral_mode']:
        # Spectral mode
        test_input = torch.randn(batch_size, config['model']['input_channels'], 
                                config['model']['spectral_bins'])
    else:
        # Raw time series mode
        window_length = config['data']['window_length']
        sampling_rate = config['data']['sampling_rate']
        time_points = int(window_length * sampling_rate)
        test_input = torch.randn(batch_size, config['model']['input_channels'], time_points)
    
    output = model(test_input)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model architecture:\n{model}") 