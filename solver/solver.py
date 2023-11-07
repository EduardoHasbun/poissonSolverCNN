import torch
import torchvision.transforms as transforms

# Load the trained model
model = UNet(in_channels, out_channels)
model.load_state_dict(torch.load('unet_model.pth'))
model.eval()  # Set the model to evaluation mode

# Prepare your input data (replace this with your actual data)
input_data = torch.randn(1, in_channels, 128, 128)

# Preprocess the input data
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert data to a tensor
])

input_data = preprocess(input_data)

# Make predictions
with torch.no_grad():
    output = model(input_data)

# The 'output' will contain the segmentation map for your input data
# You can post-process the output as needed for your specific application
