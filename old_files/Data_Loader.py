import math
from torchvision import transforms, datasets

from PIL import Image
import os
import sys

#Check for corrupt files
def check_data(dir):
	max = [0,0]
	for root, dirs, files in os.walk(dir):
		for name in files:
			try:
				v_image = Image.open(os.path.join(root, name))
				v_image.verify()
				size = v_image.size
				if size[0] > max[0]:
					max[0] = size[0]
				if size[1] > max[1]:
					max[1] = size[1]
			except Exception:
				print("file " + os.path.join(root, name) + " is corrupt and has been deleted.")
				os.remove(os.path.join(root, name))
	return max

# Load MNIST data
def mnist_data():
	compose = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5,), (0.5,))
		])
	return datasets.MNIST(root="./dataset", train=True, transform=compose, download=True)

# Load EMNIST data
def emnist_data():
	compose = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5,), (0.5,))
		])
	return datasets.EMNIST(root="./dataset", split="byclass", train=True, transform=compose, download=True)

# Load IAM words data
def iam_data(maxDimension):
	compose = transforms.Compose(
		[transforms.Pad(max(maxDimension), fill=255),
		 transforms.CenterCrop((maxDimension[1], maxDimension[0])),
		 transforms.Grayscale(),
		 transforms.ToTensor(),
		 transforms.Normalize((0.5,), (0.5,))
		])
	return datasets.ImageFolder(root = "./dataset/IAM/words/", transform=compose)

#Get specified dataset
def get_data(dataset):
    if(dataset == "IAM"):
        # Check IAM dataset for corrupt files and get max image dimensions
        maxDimension = check_data("./dataset/IAM/words/")
        data = iam_data(maxDimension)
    elif(dataset == "EMNIST"):
        data = emnist_data()
    elif(dataset == "MNIST"):
        data = mnist_data()
    else:
        sys.exit("No correct dataset specified")
    return data