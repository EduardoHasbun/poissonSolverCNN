import matplotlib.pyplot as plt
import numpy as np

# Load the text file
file_path = "C:/Codigos/poissonSolverCNN/2D/training/trained_models/UNet4_ks3_rf200.pth_losses.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

# Extract losses
epoch_losses = []
batch_losses = []
laplacian_losses = []
dirichlet_losses = []
reading_section = None

# Weight factors
laplacian_weight = 8.0e+5  # Adjust as needed
dirichlet_weight = 1.0e+3  # Example weight, modify accordingly

for line in lines:
    line = line.strip()
    if not line:  # Skip empty lines
        continue
    
    # Identify section headers
    if "Epoch Losses:" in line:
        reading_section = "epoch"
        continue
    elif "Batch Losses:" in line:
        reading_section = "batch"
        continue
    elif "Laplacian Losses:" in line:
        reading_section = "laplacian"
        continue
    elif "Dirichlet Losses:" in line:
        reading_section = "dirichlet"
        continue
    elif "Total Losses:" in line:
        reading_section = "total"
        continue

    # Handle potential parsing issues
    try:
        values = [float(x) for x in line.split(",") if x.strip()]  # Remove empty entries
        if reading_section == "epoch":
            epoch_losses.extend([x / laplacian_weight for x in values])
        elif reading_section == "batch":
            batch_losses.extend([x / laplacian_weight for x in values])
        elif reading_section == "laplacian":
            laplacian_losses.extend([x / laplacian_weight for x in values])
        elif reading_section == "dirichlet":
            dirichlet_losses.extend([x / dirichlet_weight for x in values])
    except ValueError as e:
        print(f"Skipping malformed line: {line}")

# Define how many steps you want to plot
max_steps = 5000  # Change this value as needed

# Slice data to limit the number of steps displayed
laplacian_losses = laplacian_losses[:max_steps]
dirichlet_losses = dirichlet_losses[:max_steps]

# Define a reasonable maximum for better visualization
max_loss = max(
    max(laplacian_losses, default=0),
    max(dirichlet_losses, default=0)
)
y_limit = max_loss * 0.08  # Set limit to 10% of max loss

# Plot
plt.figure(figsize=(10, 5))
if laplacian_losses:
    plt.plot(laplacian_losses, label="Laplacian Losses")
if dirichlet_losses:
    plt.plot(dirichlet_losses, label="Dirichlet Losses")

plt.ylim(0, y_limit)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
