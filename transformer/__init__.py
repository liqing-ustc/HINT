import sys
sys.path.append("./transformer/")

import torch.nn
from layers.transformer import Transformer, UniversalTransformer, RelativeTransformer, UniversalRelativeTransformer
from models import TransformerEncDecModel

def create_model(config) -> torch.nn.Module:
	rel_args = dict(pos_embeddig=(lambda x, offset: x), embedding_init="xavier")
	trafos = {
		"scaledinit": (Transformer, dict(embedding_init="kaiming", scale_mode="down")),
		"opennmt": (Transformer, dict(embedding_init="xavier", scale_mode="opennmt")),
		"noscale": (Transformer, {}),
		"universal_noscale": (UniversalTransformer, {}),
		"universal_scaledinit": (UniversalTransformer, dict(embedding_init="kaiming", scale_mode="down")),
		"universal_opennmt": (UniversalTransformer, dict(embedding_init="xavier", scale_mode="opennmt")),
		"relative": (RelativeTransformer, rel_args),
		"relative_universal": (UniversalRelativeTransformer, rel_args)
	}

	constructor, args = trafos[config.transformer]

	return TransformerEncDecModel(config.in_vocab_size, config.out_vocab_size, config.hid_dim,
									num_encoder_layers=config.enc_layers,
									num_decoder_layers=config.dec_layers,
									transformer=constructor,
									decoder_sos=config.decoder_sos,
									decoder_eos=config.decoder_eos,
									in_embedding_size=config.emb_dim,
									out_embedding_size=config.emb_dim,
									nhead=config.nhead,
									dropout=config.dropout,
									**args)