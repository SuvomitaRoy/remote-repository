import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE_FC(nn.Module):
    def __init__(self, layers, latent_dim = 200, leak = .1):
        super(VAE_FC, self).__init__()
        
        encoder = []
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            encoder.append(nn.Linear(l_in, l_out))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder) 

        # latent mean and variance
        self.fc_mean = nn.Linear(layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(layers[-1], latent_dim)
        
        # decoder input
        self.decoder_input = nn.Linear(latent_dim, layers[-1]) 
        rev_layers = list(reversed(layers))
        decoder = [nn.ReLU()]
        for l_in, l_out in zip(rev_layers[:-2], rev_layers[1:-1]):
            decoder.append(nn.Linear(l_in, l_out))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(layers[1], layers[0]))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)
     
    def encode(self, x):
        x = x.view(x.size(0),-1)
        x = self.encoder(x)
        x_means = self.fc_mean(x)
        x_logvar = self.fc_logvar(x)
        return x_means, x_logvar
    
    def reparameterization(self, mean, logvar):
        eps = torch.randn_like(logvar).to(device)
        return mean + logvar.exp() * eps

    def decode(self, x):
        x = self.decoder_input(x)
        x = self.decoder(x)
        return x.view(x.size(0),1,28,28)
        #return x

    def forward(self, x):
        x_means, x_logvar = self.encode(x)
        eps = self.reparameterization(x_means, x_logvar)
        x_hat = self.decode(eps)
        return x_hat, x_means, x_logvar
    
    def sample(self, num_samples): 
        samples = torch.randn(num_samples, latent_dim)
        x = self.decode(samples)
        return x
    
    def reconstruct(self, x):
        with torch.no_grad():
            mean, logvar = self.encode(x)
            return self.decode(mean)