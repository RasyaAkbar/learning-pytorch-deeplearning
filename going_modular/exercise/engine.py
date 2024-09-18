"""
Contains functions for training and testing a PyTorch model.
"""
from typing import Dict, List, Tuple

import torch

# Import tqdm for progress bar
# tqdm.auto recognize computer enviroment we use and give the best type progress bar, ex: jupyter notebook bar differ from python script
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device
               ):
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
    
    (0.1112, 0.8743)
  """
  ### Training
  train_loss, train_acc = 0, 0
  # Put model into training model
  model.train()

  # Add a loop to loop through the training batches
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    # 1. Forward pass
    y_pred = model(X)

    l1_lambda = 0.001
    l1_norm = sum(p.abs().sum() for p in model.parameters()) # L1 reg

    #2. Calculate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item() # Accumulate train loss

    # Calculate accuracy metric
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc+= ((y_pred_class==y).sum().item()/len(y_pred)) * 100

    #3. Optimizer zero grad
    optimizer.zero_grad()

    #4. Loss backward
    loss.backward()

    #5. Optimizer step
    optimizer.step()

    #if(batch % 2 == 0):
    #  print(f"Looked at {batch * len(X)} samples out of {len(train_dataloader.dataset)} samples")

  # Divide total train loss by length of train dataloader
  train_loss /= len(dataloader) # Average loss per batch

  # Average accuracy per batch
  train_acc /= len(dataloader)

  print(f"\nTrain loss: {train_loss:.4f} , Train accuracy: {train_acc:.4f}% ")
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device = device
              ):
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:
    
    (0.0223, 0.8985)
  """
  ### Testing
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X_test, y_test in dataloader:
      X_test, y_test = X_test.to(device), y_test.to(device)

      # 1. Forward pass
      test_pred_logits = model(X_test)

      #2. Calculate loss
      loss = loss_fn(test_pred_logits, y_test)
      test_loss += loss.item() # Accumulate test loss

      # Calculate accuracy metric
      #test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
      #test_acc+= ((test_pred_labels==y_test).sum().item()/len(test_pred_logits)) * 100
      test_pred_labels = test_pred_logits.argmax(dim=1) # another way of doing it (?) further testing required
      test_acc+= ((test_pred_labels==y_test).sum().item()/len(test_pred_labels)) * 100

    # Average loss per batch
    test_loss /= len(dataloader)

    # Average accuracy per batch
    test_acc /= len(dataloader)

  print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}%\n")
  return test_loss, test_acc

def train(model: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device = device,
            epochs: int = 5
          ):
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Create training and test loop
  for epoch in tqdm(range(epochs)): # the way tqdm works is to wrap our iterator in tqdm
    print(f"Epoch: {epoch}\n-----")
    train_loss, train_acc = train_step(model=model,
               dataloader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer)
    model.eval()
    test_loss, test_acc = test_step(model=model,
              dataloader=test_dataloader,
              loss_fn=loss_fn)

    # Update result dictionary
    train_loss = train_loss
    #print(train_loss, train_acc, test_loss, test_acc)
    results["train_loss"].append(train_loss)#.cpu()
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)#.cpu()
    results["test_acc"].append(test_acc)


  return results
