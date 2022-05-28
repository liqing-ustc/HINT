from torch import nn
import torchvision

def make_model(config):
	pretrained = config.perception_pretrain == 'imagenet'
	model = torchvision.models.resnet18(pretrained=pretrained)
	model.fc = nn.Linear(model.fc.in_features, config.emb_dim)
	return model