import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128

def build_loss_vae(lambda_reconstruct = 0.5, lambda_kl = 0.5): #closure
    def loss_vae(x, x_hat, mean, logvar):
        reconstruct_loss = lambda_reconstruct * (x - x_hat).pow(2).sum()
        KL_loss = -lambda_kl * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruct_loss + KL_loss
    return loss_vae

def train_model_vae(data_loader, model, criterion, optimizer, nepochs):
    #List to store loss to visualize
    train_losses = []
    train_acc = []
    start_epoch = 0

    for epoch in range(start_epoch, nepochs):
        train_loss = 0.
        valid_loss = 0.
        correct = 0

        model.train()
        for batch_idx, (input_, target) in enumerate(data_loader):
            input_ = input_.to(device)
            target = target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            x_hat, mean, logvar = model(input_)

            # calculate the batch loss
            loss = criterion(input_,x_hat,mean,logvar)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item() * input_.size(0)

        # calculate average losses
        train_loss = train_loss/len(data_loader.dataset)
        train_losses.append(train_loss)

        # print losses statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))