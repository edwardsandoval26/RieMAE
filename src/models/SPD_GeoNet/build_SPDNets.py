from torch import nn
from .spdnet.spd import SPDTransform, SPDRectified, SPDTangentSpace

class SPDNetClassifier(nn.Module):
    def __init__(self, input_dim=192, bire_blocks=1, num_classes=10):
        super(SPDNetClassifier, self).__init__()
        
        # Encoder layers (SPD processing layers)
        encoder_layers = []
        encoder_output_dim = input_dim // 2
        
        for _ in range(bire_blocks):
            encoder_output_dim = input_dim // 2
            encoder_layers.append(SPDTransform(input_dim, encoder_output_dim, 1))
            encoder_layers.append(SPDRectified())
            input_dim = encoder_output_dim
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        # Tangent Space layer to flatten SPD representation
        self.tangent_space = SPDTangentSpace(encoder_output_dim)
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(encoder_output_dim, 128),  # Adjust dimensions as needed
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer with `num_classes` neurons
        )
        
    def forward(self, x):
        # Pass through encoder (SPD processing layers)
        x_enc = self.encoder_layers(x)
        
        # Flatten SPD representation using tangent space
        x_flat = self.tangent_space(x_enc)
        
        # Pass through fully connected layers for classification
        x_class = self.fc(x_flat)

        return x_class
