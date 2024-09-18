"""
Contains PyTorch model code to instantiate a TinyVGG model from CNN explainer website
"""
import torch

from torch import nn

class TinyVGG(nn.Module):
  """
  Model architecture that replicates the TinyVGG
  model from CNN explainer website  https://poloclub.github.io/cnn-explainer/
  
  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int
               ):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        # Create conv layer
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3, # often also referred to as filter size, refers to the dimensions of the sliding window over the input. can be tuple, ex:(3, 3)
                  stride=1, # indicates how many pixels the kernel should be shifted over at a time.
                  padding=1 # By adding padding, the model retains information from the edges of the input data, which would otherwise be lost without padding.
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3, # often also referred to as filter size, refers to the dimensions of the sliding window over the input. can be tuple, ex:(3, 3)
                  stride=1, # indicates how many pixels the kernel should be shifted over at a time.
                  padding=1 # By adding padding, the model retains information from the edges of the input data, which would otherwise be lost without padding.
                  ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2
                     ), # stride by default is the same as kernel size
        #nn.Dropout(p=0.5)#   Dropout with a 70% drop rate
    )

    self.conv_block_2 = nn.Sequential(
        # Create conv layer
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3, # often also referred to as filter size, refers to the dimensions of the sliding window over the input. can be tuple, ex:(3, 3)
                  stride=1, # indicates how many pixels the kernel should be shifted over at a time.
                  padding=1 # By adding padding, the model retains information from the edges of the input data, which would otherwise be lost without padding.
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3, # often also referred to as filter size, refers to the dimensions of the sliding window over the input. can be tuple, ex:(3, 3)
                  stride=1, # indicates how many pixels the kernel should be shifted over at a time.
                  padding=1 # By adding padding, the model retains information from the edges of the input data, which would otherwise be lost without padding.
                  ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2
                     ), # stride by default is the same as kernel size
        nn.Dropout(p=0.5) #  Dropout with a 50% drop rate
    )
    self.conv_block_3 = nn.Sequential(
        # Create conv layer
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3, # often also referred to as filter size, refers to the dimensions of the sliding window over the input. can be tuple, ex:(3, 3)
                  stride=1, # indicates how many pixels the kernel should be shifted over at a time.
                  padding=0 # By adding padding, the model retains information from the edges of the input data, which would otherwise be lost without padding.
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3, # often also referred to as filter size, refers to the dimensions of the sliding window over the input. can be tuple, ex:(3, 3)
                  stride=1, # indicates how many pixels the kernel should be shifted over at a time.
                  padding=0 # By adding padding, the model retains information from the edges of the input data, which would otherwise be lost without padding.
                  ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2
                     ), # stride by default is the same as kernel size
        nn.Dropout(p=0.5)  # Dropout with a 70% drop rate
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*256, out_features=output_shape) # 7*7 = 49 which is the shape when the image compressed is flatten
    )
  def forward(self, x: torch.Tensor)-> torch.Tensor:
    return self.classifier(self.conv_block_2(self.conv_block_1(x))) #benefited from operator fusion which behind the scene speeds up how GPU perform computation cuz its 1 step
    # https://horace.io/brrr_intro.html
