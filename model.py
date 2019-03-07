import torch
import numpy as np

class CVAE(torch.nn.Module):
    def __init__(self, latent_size, ce_weight, KLD_weight, channels, kernels, strides):
	##Initialization
        super(CVAE, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.LeakyReLU()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ce_weight = ce_weight
        self.KLD_weight = KLD_weight

        self.latent_size = latent_size	
        self.channels = channels
        self.kernels = kernels
        self.strides = strides

        self.fc_size = 2000
        for i in range(len(self.channels)):
                self.fc_size = ((self.fc_size - self.kernels[i])/self.strides[i])+1


        ##Encoder_layers
        self.encoder_layers = torch.nn.ModuleList()
        self.encoder_norms = torch.nn.ModuleList()

        ##Insert layers to Encoder_layers
        layer_count = 0
        for parameters in zip(self.channels, self.kernels, self.strides):
                if layer_count == 0:
                        self.encoder_layers.append(torch.nn.Conv1d(in_channels=4, out_channels=parameters[0], kernel_size=parameters[1], stride=parameters[2]))
                        self.encoder_norms.append(torch.nn.BatchNorm1d(parameters[0]))
                else:
                        self.encoder_layers.append(torch.nn.Conv1d(in_channels=self.channels[layer_count-1], out_channels=parameters[0], kernel_size=parameters[1], stride=parameters[2]))
                        self.encoder_norms.append(torch.nn.BatchNorm1d(parameters[0]))
                layer_count += 1

        self.encode_mu = torch.nn.Linear(int(self.fc_size*self.channels[-1]), self.latent_size)
        self.encode_logvar = torch.nn.Linear(int(self.fc_size*self.channels[-1]), self.latent_size)

        ##Decoder layers
        self.latent = torch.nn.Linear(self.latent_size, int(self.fc_size*self.channels[-1]))
        self.decoder_layers = torch.nn.ModuleList()
        self.decoder_norms = torch.nn.ModuleList()

        ##Insert layers to Decoder layers
        layer_count = 0
        for parameters in zip(self.channels[::-1], self.kernels[::-1], self.strides[::-1]):
                if len(self.channels) > 1+layer_count:
                        self.decoder_layers.append(torch.nn.ConvTranspose1d(in_channels=parameters[0], out_channels=self.channels[::-1][layer_count+1], kernel_size=parameters[1], stride=parameters[2]))
                        self.decoder_norms.append(torch.nn.BatchNorm1d(self.channels[::-1][layer_count+1]))
                layer_count += 1
        self.output_layer = torch.nn.ConvTranspose1d(in_channels=self.channels[0], out_channels=4, kernel_size=self.kernels[0], stride=self.strides[0])
        
        
    
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
        for layer, batchnorm in zip(self.encoder_layers, self.encoder_norms):
                x = batchnorm(self.dropout(self.relu(layer(x))))
        x = x.view(-1, int(self.fc_size*self.channels[-1]))
        self.mu = self.encode_mu(x)
        self.logvar = self.softplus(self.encode_logvar(x))
        
        return self.reparameterize(self.mu, self.logvar), self.mu, self.logvar
    
    def decode(self, x):
        '''
        The second part of the model, where the latent representation is decoded.
        '''
        x = self.latent(x)
        x = x.view(-1, self.channels[-1], int(self.fc_size))
        if len(self.channels) > 1:
                for layer, batchnorm in zip(self.decoder_layers, self.decoder_norms):
                        x = batchnorm(self.dropout(self.relu(layer(x))))
        
        reconstruction = self.softmax(self.output_layer(x))
      
        return reconstruction
    
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





