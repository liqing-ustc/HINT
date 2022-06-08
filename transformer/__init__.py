import sys
sys.path.append("./transformer/")

import torch.nn
from layers.transformer import Transformer, UniversalTransformer, RelativeTransformer, UniversalRelativeTransformer
from models import TransformerEncDecModel, BertModel

def create_model(config) -> torch.nn.Module:
	rel_args = dict(pos_embeddig=(lambda x, offset: x), embedding_init="xavier", max_rel_pos=config.max_rel_pos)
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

	constructor, args = trafos[config.model.replace('TRAN.', '')]
	model_class = BertModel if config.result_encoding == 'sin' else TransformerEncDecModel

	return model_class(n_input_tokens=config.in_vocab_size, 
						n_out_tokens=config.out_vocab_size, 
						state_size=config.hid_dim,
						num_encoder_layers=config.enc_layers,
						num_decoder_layers=config.dec_layers,
						transformer=constructor,
						decoder_sos=config.decoder_sos,
						decoder_eos=config.decoder_eos,
						in_embedding_size=config.emb_dim,
						out_embedding_size=config.emb_dim,
						nhead=config.nhead,
						dropout=config.dropout,
						pos_emb_type=config.pos_emb_type,
						output_attentions=config.output_attentions,
						**args)