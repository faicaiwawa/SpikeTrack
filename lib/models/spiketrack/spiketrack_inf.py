"""
SpikeTrack Model
"""
import torch.nn.functional as F
from torch import nn
import copy
from lib.models.spiketrack.decoder import build_decoder
from .sdtv3 import build_backbone, mem_update
from .sdtv3_search_inference import build_backbone_search
from .sdtv3_temp_inference import build_backbone_temp


class SPIKETRACK(nn.Module):
    """ This is the base class for SeqTrack """
    def __init__(self, encoder, decoder):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.spike_rate_dict = {}
        #self.register_spike_rate_hooks()

        self.current_image_idx = 0
    def register_spike_rate_hooks(self):
        for name, module in self.named_modules():
            if isinstance(module, mem_update):
                def hook_fn(module, input, output, name=name):

                    if output.shape[0] > 3: # temporary solution, for temporal-fusion-mlp neuron input shape [B*H*W, C, T]
                        output = output.transpose(0, 2)

                    spatial_dims = tuple(range(1, output.ndim))
                    firing_rates = output.float().mean(dim=spatial_dims) * 4

                    firing_rates = firing_rates.cpu().tolist()

                    image_key = f'image_{self.current_image_idx}'
                    self.spike_rate_dict.setdefault(name, {}).setdefault(image_key, []).extend(firing_rates)

                module.register_forward_hook(hook_fn)


    def reset_image_counter(self):
        self.current_image_idx += 1

    def forward(self, images_list=None, xz=None, seq=None, mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(images_list)
        elif mode == "decoder":
            return self.forward_decoder(xz)
        else:
            raise ValueError

    def forward_encoder(self, images_list, cache):

        xz = self.encoder(images_list, cache)

        return xz


    def forward_decoder(self, xz):

        out = self.decoder(xz)

        return out

    def inference_decoder(self, xz):
        # Forward the decoder

        out = self.decoder(xz)
        self.reset_image_counter()
        if len(self.spike_rate_dict)>0 :
            spike_rate_dict = self.spike_rate_dict
        else:
            spike_rate_dict = None

        return out, spike_rate_dict



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_spiketrack(cfg):
    encoder_search = build_backbone_search(cfg)
    encoder_temp = build_backbone_temp(cfg)
    decoder = build_decoder(cfg, cfg.MODEL.HIDDEN_DIM)
    model = SPIKETRACK(
        encoder_search,
        decoder,
    )

    return model, encoder_temp
