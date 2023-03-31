import torch
import torch.nn
import torch.nn.functional as F
from .transformer import TransformerDecoderBase, ActivationFunction, TransformerEncoderLayer, TransformerDecoderLayer, \
                         Transformer


class UniversalTransformerEncoder(torch.nn.Module):
    def __init__(self, layer, depth: int, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)
        self.layers = [self.layer] * depth

    def forward(self, data: torch.Tensor, *args, **kwargs):
        output_attentions = kwargs.get('output_attentions', False)
        attentions = []
        for l in self.layers:
            data = l(data, *args, **kwargs)
            if output_attentions:
                data, layer_attentions = data
                attentions.append(layer_attentions)
        if output_attentions:
            return data, attentions
        else:
            return data


class UniversalTransformerDecoder(TransformerDecoderBase):
    def __init__(self, layer, depth: int, d_model: int, *args, **kwargs):
        super().__init__(d_model)
        self.layer = layer(d_model, *args, **kwargs)
        self.layers = [self.layer] * depth

    def forward(self, data: torch.Tensor, *args, **kwargs):
        output_attentions = kwargs.get('output_attentions', False)
        decoder_attentions = []
        cross_attentions = []
        for l in self.layers:
            data = l(data, *args, **kwargs)
            if output_attentions:
                data, (decoder_layer_attentions, cross_layer_attentions) = data
                decoder_attentions.append(decoder_layer_attentions)
                cross_attentions.append(cross_layer_attentions)
        if output_attentions:
            return data, decoder_attentions, cross_attentions
        else:
            return data


def UniversalTransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: UniversalTransformerEncoder(layer, *args, **kwargs)


def UniversalTransformerDecoderWithLayer(layer=TransformerDecoderLayer):
    return lambda *args, **kwargs: UniversalTransformerDecoder(layer, *args, **kwargs)


class UniversalTransformer(Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: ActivationFunction = F.relu, **kwargs):

        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation,
                         UniversalTransformerEncoderWithLayer(),
                         UniversalTransformerDecoderWithLayer())
