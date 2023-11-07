from voiceNet import *
import torch
from torchviz import make_dot

# Create an instance of the model
model = VoiceClassificationModel(input_size=748)

# Create a random input tensor of the appropriate shape
input_tensor = torch.randn(1, 1, 748)  # Batch size 1, 1 channel, input size 748

# Pass the input through the model to generate a visualization
output = model(input_tensor)
dot = make_dot(output)

# Save the visualization to a file (optional)
dot.render("audio_classification_model", format="png")