import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from yolov5.datasets import YoloDataset
from yolov5.models import YoloV5s

# Initialize the YOLOv5s model
model = YoloV5s()

# Load the pre-trained weights for the model
model.load_state_dict(torch.load('path/to/pretrained/weights'))

# Freeze the layers except for the last three layers
for param in model.parameters():
    param.requires_grad = False
for param in model.model[-1].parameters():
    param.requires_grad = True

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()

# Create the data loader for the traffic light dataset
dataset = YoloDataset('path/to/traffic/light/dataset')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Fine-tune the model on the traffic light dataset
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch {} Batch {}: Loss = {}'.format(epoch, batch_idx, loss.item()))

# Save the fine-tuned weights for the model
torch.save(model.state_dict(), 'path/to/fine-tuned/weights')