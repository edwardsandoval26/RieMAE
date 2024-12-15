from torch import nn
from .spdnet.spd import SPDTransform, SPDRectified, SPDTangentSpace


class ASPDNet(nn.Module):
    def __init__(self, input_dim=128, bire_blocks=1):
        super(ASPDNet, self).__init__()
        
        # Encoder layers
        encoder_layers = []
        encoder_output_dim = input_dim // 2
        
        for _ in range(bire_blocks):
            encoder_output_dim = input_dim // 2
            encoder_layers.append(SPDTransform(input_dim, encoder_output_dim, 1))
            encoder_layers.append(SPDRectified())
            input_dim = encoder_output_dim
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        decoder_input_dim = encoder_output_dim
        decoder_output_dim = decoder_input_dim * 2
        
        for _ in range(bire_blocks):
            decoder_layers.append(SPDTransform(decoder_input_dim, decoder_output_dim, 1))
            decoder_layers.append(SPDRectified())
            decoder_input_dim = decoder_output_dim
            decoder_output_dim = decoder_input_dim * 2
        
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        # Pass through encoder
        x_enc = self.encoder_layers(x)

        # Pass through decoder
        x = self.decoder_layers(x_enc)

        return x, x_enc

