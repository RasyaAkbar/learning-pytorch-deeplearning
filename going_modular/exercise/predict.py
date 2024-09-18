"""
Predict a custom image using loaded model
"""
print("type(custom_image_transformed.numpy())")
import torchvision
import matplotlib.pyplot as plt
import torch
from typing import Dict

device = "cuda" if torch.cuda.is_available() else "cpu"
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names:Dict[int, str],
                        transform: torchvision.transforms,
                        device: torch.device = device
                        ):
  # Read in custom image
  
  custom_image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255
  
  # Transform target image
  custom_image_transformed = transform(custom_image)
  
  model.eval()
  with torch.inference_mode():
    custom_image_pred = model(custom_image_transformed.unsqueeze(0).to(device)) # add 1 batch

  # Turn logits into prediction probabities
  custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)

  # Turn prediction probabities into prediction label
  custom_image_label=custom_image_pred_probs.argmax(dim=1)

  plt.imshow(custom_image_transformed.permute(1,2,0).numpy())
  plt.title(f"Prediction: {class_names[custom_image_label]} | Confidence: {custom_image_pred_probs.max()*100:.1f}%")
  plt.axis(False)
