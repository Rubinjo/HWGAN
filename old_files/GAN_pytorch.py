import torch
from torch import nn
from torchvision import transforms, datasets

import math
import matplotlib.pyplot as plt

from Data_Loader import get_data



# Random generator seed for replication
torch.manual_seed(111)
# Number of batches
BATCH_SIZE = 32
# Learning rate
LR = 0.0002
# Number of epochs
NUM_EPOCHS = 3

print("You are using pytorch version " + torch. __version__)

# Create execution device that points to GPU or CPU
device = ""
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("GPU detected, using GPU for calculations...")
else:
	device = torch.device("cpu")
	print("No GPU detected, using CPU for calculations...")

# Get data
data = get_data("EMNIST")
# Create loader with data, so that we can iterate over it
train_loader = torch.utils.data.DataLoader(
	data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		# self.cnn_layers = nn.Sequential(
		# 	nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
		# )

		self.linear_layers = nn.Sequential(
			nn.Linear(784, 1024),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(256, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		# x = self.cnn_layers(x)
		x = x.view(x.size(0), 784)
		x = self.linear_layers(x)
		return x

discriminator = Discriminator().to(device=device)

class Generator(nn.Module):
	def __init__(self):
		super().__init__()

		# self.cnn_layers = nn.Sequential(
		# 	nn.Conv2d(10, 4, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(inplace=True),
		# )

		self.linear_layers = nn.Sequential(
			nn.Linear(100, 256),
			nn.ReLU(),
			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Linear(512, 1024),
			nn.ReLU(),
			nn.Linear(1024, 784),
			nn.Tanh(),
		)

	def forward(self, x):
		# output = self.cnn_layers(x)
		output = self.linear_layers(x)
		output = output.view(x.size(0), 1, 28, 28)
		return output

generator = Generator().to(device=device)

loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LR)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=LR)

# Iterate through epochs
for epoch in range(NUM_EPOCHS):
	for n, (real_samples, _) in enumerate(train_loader):

		# Data for training the discriminator
		real_samples = real_samples.to(device=device)
		real_samples_labels = torch.ones((BATCH_SIZE, 1)).to(
			device=device
		)
		latent_space_samples = torch.randn(BATCH_SIZE, 100).to(
			device=device
		)
		generated_samples = generator(latent_space_samples)
		generated_samples_labels = torch.zeros((BATCH_SIZE, 1)).to(
			device=device
		)
		all_samples = torch.cat((real_samples, generated_samples))
		all_samples_labels = torch.cat(
			(real_samples_labels, generated_samples_labels)
		)

		# Training the discriminator
		discriminator.zero_grad()
		output_discriminator = discriminator(all_samples)
		loss_discriminator = loss_function(
			output_discriminator, all_samples_labels
		)
		loss_discriminator.backward()
		optimizer_discriminator.step()

		# Data for training the generator
		latent_space_samples = torch.randn(BATCH_SIZE, 100).to(
			device=device
		)

		# Training the generator
		generator.zero_grad()
		generated_samples = generator(latent_space_samples)
		output_discriminator_generated = discriminator(generated_samples)
		loss_generator = loss_function(
			output_discriminator_generated, real_samples_labels
		)
		loss_generator.backward()
		optimizer_generator.step()

		# Show loss
		if n == BATCH_SIZE - 1:
			print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
			print(f"Epoch: {epoch} Loss G.: {loss_generator}")

#Create model sample
latent_space_samples = torch.randn(BATCH_SIZE, 100).to(device=device)
generated_samples = generator(latent_space_samples)

generated_samples = generated_samples.cpu().detach()
for i in range(16):
	ax = plt.subplot(4, 4, i + 1)
	plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
	plt.xticks([])
	plt.yticks([])
plt.show()
