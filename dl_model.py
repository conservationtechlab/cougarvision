import torch

import timm


#Path to Save Model to
PATH = 'classifier_models/ig_resnext101_32x8d.pt'

# model = EfficientNet.from_pretrained('efficientnet-b1')
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)

model = timm.create_model('ig_resnext101_32x8d', pretrained=True)
model.eval()

# Uncomment below for EfficientNet models
# model.set_swish(memory_efficient=False)


example = torch.rand(1, 3, 600, 600)
traced_script_module = torch.jit.trace(model, example)
torch.jit.save(traced_script_module, PATH)

