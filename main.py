from torch.optim import Adam
import torch
from u_net import Unet
from train import train_model
from prepare_data import BitmogiDataset
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

#will go to config
image_size = 128
channels   = 1
epochs     = 1  
data_path  = "/Users/sbhar/Riju/TorchDemoCode/TrinaCode/bitmojis/"
batch_size = 6
results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)

def main():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
            device = torch.device("cpu")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
            device = torch.device("cpu")

    else:
        device = torch.device("mps")
    
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # define image transformations (e.g. using torchvision)
    transform_ = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)    
        ])

    bitmoji_dataset = BitmogiDataset(data_dir=data_path,transform=transform_)
    dataloader = DataLoader(bitmoji_dataset, batch_size=batch_size, shuffle=True)

    train_model(epochs=epochs,model=model,optimizer=optimizer,device=device,results_folder=results_folder,dataloader=dataloader)


if __name__ == "__main__":
    main()
