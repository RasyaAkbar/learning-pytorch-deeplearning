"""
Trains a PyTorch image classification model using device agnostic code
"""
import os
import torch

from torchvision import transforms
from timeit import default_timer as timer

import data_setup, engine, model_builder, utils

import argparse
parser = argparse.ArgumentParser(prog='exercise/train')

# Define the expected arguments
parser.add_argument('--learning_rate', type=float, help="learning rate")
parser.add_argument('--batch_size', type=int, help="batch size")
parser.add_argument('--num_epochs', type=int, help="num epochs")

args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs or 5
BATCH_SIZE = args.batch_size or 32
HIDDEN_UNITS = 10
LEARNING_RATE = args.learning_rate or 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create DataLoader's and get class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, 
                                                                               test_dir=test_dir,
                                                                               transform=data_transform,
                                                                               batch_size=BATCH_SIZE)

# Create or load model
if (os.path.isfile("models/05_going_modular_script_mode_tinyvgg_model.pth")):
  # To load a saved state_dict we have to instantiate a new instance of our model class
  # Load the saved state_dict of model (this will update the new instance with updated parameters)
  model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)).to(device)
  model.load_state_dict(torch.load(f="models/05_going_modular_script_mode_tinyvgg_model.pth", weights_only=True))
else:
  model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)
                              ).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE
                             )

# Start the timer
start_time = timer()

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model to file
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth") 
