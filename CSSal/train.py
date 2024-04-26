from tqdm import tqdm
import torch.optim as optim


from model import Generator
from dataloder import SalDS
from utils import SalLoss 

# W&B init 


# Log init 


# ... (rest of your code)

model = Generator()

# Hyperparameters
learning_rate = 1e-3
epochs = 10

# Create data loaders (replace with your data loading logic)
train_loader = ...
val_loader = ...  # Optional validation loader

# Create optimizer
optimizer = optim.Adam(g.parameters(), lr=learning_rate)

# Define loss function (e.g., mean squared error for grayscale reconstruction)
criterion = nn.MSELoss()

model.train()
# Training loop
for epoch in range(epochs):
    print(f"Epoch: {epoch+1}/{epochs}")
    pbar = tqdm(train_loader)  # Wrap data loader with tqdm
    for i, (data, _) in enumerate(pbar):
        # Move data to the device (GPU if available)
        data = data.to(device)

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, data)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm description (optional)
        pbar.set_description(f"Loss: {loss.item():.6f}")

    # Optional validation step (evaluate on validation data)
    if val_loader:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for j, (val_data, _) in enumerate(val_loader):
                val_data = val_data.to(device)
                val_output = g(val_data)
                val_loss += criterion(val_output, val_data).item()
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.6f}")
    
    model.train()

print("Training complete!")
