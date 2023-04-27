import torch
from utils import timesteps,num_to_groups
from loss_fn import p_losses
from sampling import sample
from torchvision.utils import save_image

#will go to config
save_and_sample_every = 1000
channels = 3
image_size = 128

def train_model(epochs,model,optimizer,device,results_folder,dataloader):
    print("Starting Training...")
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            print("Running Step : ",step)
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()
            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(model,image_size=image_size, batch_size=n, channels=channels), batches))
                all_images_list = [torch.tensor(image_list) for image_list in all_images_list]
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                all_images_perm = all_images.permute(0, 2, 1, 3, 4)
                print("Saving Image")
                for i,image in enumerate(all_images_perm):
                    save_image(all_images_perm[i], str(results_folder / f'sample-{milestone}.png'),nrow = 1)