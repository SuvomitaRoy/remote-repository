import torch
import argparse
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from vae import VAE_FC, build_loss_vae, train_model_vae


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    train_data = MNIST(root="./data", train=True, transform=ToTensor(), download=True)
    test_data = MNIST(root="./data", train=False, transform=ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    train_size = len(train_data)
    test_size = len(test_data)

    classes = [f"{i}" for i in range(10)]

    layers = [28**2, 500,200,50]
    model = VAE_FC(layers,latent_dim=10).to(device)
    criterion = build_loss_vae(lambda_reconstruct=0.5, lambda_kl=0.5)
    optimizer = optim.Adam(model.parameters(), lr = .01)
    nepochs = 10
    train_model_vae(train_loader,model,criterion, optimizer,nepochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fully Connected Variational Autoencoder')
    parser.add_argument("--batch_size", type=int, default=128,
                      help="The batch size to use for training.")
    parser.add_argument("--latent_dim", type=int, default=10,
                      help="The latent dimension to use for training.")
    parser.add_argument("--lambda_reconstruct", type=int, default=0.5,
                      help="The lambda reconstruct to use for training.")
    parser.add_argument("--lambda_kl", type=int, default=0.5,
                      help="The lambda_kl to use for training.")
    parser.add_argument("--lr", type=int, default=0.01,
                      help="The learning rate to use for training.")
    parser.add_argument("--nepochs", type=int, default=10,
                      help="The number of epochs to use for training.")
    parser.add_argument("--layers", type=list, default=[28**2, 500,200,50],
                      help="The list of layers to use for training.")
    parser.add_argument("--datasets_path", type=str, default="~/datasets",
                      help="The dataset path to use for training.")
    args = parser.parse_args()
    print('Model Loading...')

