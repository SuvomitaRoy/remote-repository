import torch
import argparse
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from vae import VAE_FC, build_loss_vae, train_model_vae

datasets_path = "~/datasets"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument("--layers", type=list, default=[784,500,200,50],
                      help="The list of layers to use for training.")
    parser.add_argument("--datasets_path", type=str, default="~/datasets",
                      help="The dataset path to use for training.")
    args = parser.parse_args()
    print('Model Loading...')

    transform = transforms.Compose([
    transforms.ToTensor(),
    ]) 

    train_data = datasets.MNIST(datasets_path, train = True,
                                download = True, transform = transform)
    test_data = datasets.MNIST(datasets_path, train=False,
                                download = True, transform = transform)

    train_size = len(train_data)
    test_size = len(test_data)

    # build the data loaders
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle = False)

    # specify the image classes
    classes = [f"{i}" for i in range(10)]

    model = VAE_FC(args.layers,args.latent_dim).to(device)
    criterion = build_loss_vae(args.lambda_reconstruct, args.lambda_kl)
    optimizer = optim.Adam(model.parameters(), args.lr)
    nepochs = 10
    train_model_vae(train_loader,model,criterion, optimizer,nepochs)
    torch.save({"vae_fc": model.state_dict()}, "VAE_FC_MNIST.pkl")
    dct_load = torch.load("VAE_FC_MNIST.pkl", weights_only = True)
    model.load_state_dict(dct_load["vae_fc"])
