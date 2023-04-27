import os
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision.transforms import Compose
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

#will go to config
image_size = 128
channels = 1
batch_size = 6
data_path = "/Users/sbhar/Riju/TorchDemoCode/TrinaCode/bitmojis/"

def show_landmarks(pixel_values):
    """Show image with landmarks"""
    plt.imshow(pixel_values[0],cmap="gray")
    plt.pause(0.001)  # pause a bit so that plots are updated


def display_dataset_gray(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['pixel_values'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break

class BitmogiDataset(Dataset):
    """Bitmoji Dataset"""
    def __init__(self,data_dir,transform=None) -> None:
        """
        data_dir  = Directory with all the images
        transform = Optional transform to be applied on a sample
        """
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))
    
    """
    def __len__(self):
        count = 0
        for path in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir,path)):
                count = count + 1
        
        return count
    """
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index  = index.tolist()
        i_path = ""
        for i,path in enumerate(os.listdir(self.data_dir)):
            if os.path.isfile(os.path.join(self.data_dir,path)):
                if i == index:
                    i_path = os.path.join(self.data_dir,path)
        
        #img_path = os.path.join(i_path)
        img_path  = i_path
        image = Image.open(img_path).convert("L")
        if self.transform:
            sample = {'pixel_values':self.transform(image)}
        else:
            sample   = {'pixel_values': image}
        
        return sample

"""


# define image transformations (e.g. using torchvision)
transform_ = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])


bitmoji_dataset = BitmogiDataset(data_dir=data_path,transform=transform_)

# create dataloader
dataloader = DataLoader(bitmoji_dataset, batch_size=batch_size, shuffle=True)
batch = next(iter(dataloader))
print(batch.keys())


#display_dataset_gray(bitmoji_dataset)

"""