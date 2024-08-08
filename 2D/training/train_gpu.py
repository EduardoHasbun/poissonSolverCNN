import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())  # Should return 0 or other device number
print(torch.cuda.device_count())  # Should return number of GPUs
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
