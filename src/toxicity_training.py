import torch
import torch.nn.functional as F
from toxicity_dataset import load_dataset, split_dataset

def training_loop(n_epochs, model, optimizer, loss_fn, training_loader, validation_loader):
  for epoch in range(1, n_epochs + 1):
    for x_train, y_train in training_loader:
      y_predicted = model(x_train)
      loss = loss_fn(y_predicted, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Disable grad for calculating validation metrics, since backpropagation
    # is not needed and this should improve performance.
    with torch.no_grad():
        total = 0
        correct = 0
        for x_val, y_val in validation_loader:
          outputs = model(x_val)
          outputs = F.sigmoid(outputs) >= 0.5
          results = outputs == y_val
          correct += results.sum()
          total += results.numel()

        print('Epoch: %d, Train Loss: %f, Val Accuracy: %f' % (epoch, float(loss), correct / total))

def train_model(input_filename, model, optimizer, loss_fn, epochs, batch_size, device = 'cpu'):
    full_dataset = load_dataset(input_filename)
    (train_dataset, validation_dataset) = split_dataset(full_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=4096, shuffle=False, generator=torch.Generator(device=device))

    training_loop(
        n_epochs=epochs,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        training_loader=train_loader,
        validation_loader=validation_loader
    )
