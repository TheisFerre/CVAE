import torch
import numpy as np

class CVAE(torch.nn.Module):
    def __init__(self, latent_size, ce_weight, KLD_weight):
	##Initialization
        super(CVAE, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.LeakyReLU()
        self.latent_size = int(latent_size)
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=1)

        self.ce_weight = ce_weight
        self.KLD_weight = KLD_weight
	
        self.encode_conv1 = torch.nn.Conv1d(in_channels=4, out_channels=30, kernel_size = 20, stride=20)
        
        self.encode_mu = torch.nn.Linear(30*100, self.latent_size)
        self.encode_logvar = torch.nn.Linear(30*100, self.latent_size)
        
        self.decode_latent = torch.nn.Linear(self.latent_size, 30*100)
  
        self.decode_conv1 = torch.nn.ConvTranspose1d(in_channels=30, out_channels=4, kernel_size=20, stride=20)
        
    
    def reparameterize(self, mu, logvar):
    
        """
        This function reparameterizes the latent space by using a normal-gaussian of size [batchsize, latent_vector].
        The normal-gaussian is scaled with the std, and then mu is added. 
        This enables the VAE to backpropagate since the std and mu vector are not randomly sampled from a gaussian distribution.
        Instead they are scaled and added.
        """
        if len(mu) != len(logvar):
            raise ValueError('Vector for mu and sigma have to be the same length')
        else:
            self.logvar = torch.exp(logvar/2)
            self.epsilon = torch.randn((mu.size())).float()#.cuda()
            self.latent_vector = torch.mul(self.epsilon, self.logvar) + mu
        
        return self.latent_vector
    
    def encode(self, x):
	'''
	The first part of the model, which encodes an input into a latent representation.
	'''
        x = self.dropout(self.relu(self.encode_conv1(x)))
        x = x.view(-1, 30*100)
        self.mu = self.encode_mu(x)
        self.logvar = self.softplus(self.encode_logvar(x))
        
        return self.reparameterize(self.mu, self.logvar), self.mu, self.logvar
    
    def decode(self, x):
	'''
	The second part of the model, where the latent representation is decoded.
	'''
        x = (self.decode_latent(x))
        x = x.view(-1, 30, 100)
        x = self.softmax(self.decode_conv1(x))
      
        return x
    
    def forward(self, x):
        '''
	Forward probagation through the model.
	'''
        latent_vector, mu, logvar = self.encode(x)
        x = self.decode(latent_vector)
        return x, mu, logvar

	
    def calc_loss(self, reconstructed_x, x, mu, logvar):
	'''
	Function for calculating ce_loss and KLD_loss.
	'''
        ce_loss = torch.nn.functional.cross_entropy(reconstructed_x, x.argmax(1), reduction = 'sum') * self.ce_weight
        KLD_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()) * self.KLD_weight
        total_loss = ce_loss + KLD_loss
        return total_loss, ce_loss, KLD_loss





