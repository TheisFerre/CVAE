import torch
import numpy as np
import math

class CVAE(torch.nn.Module):
    def __init__(self, latent_size, alpha, dropout, channels, kernels, strides, padding, cuda_flag):
	##Initialization
        super(CVAE, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.relu = torch.nn.LeakyReLU()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=1)
        self.alpha = alpha
        self.cuda_flag = cuda_flag

        self.latent_size = latent_size	
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.padding = padding

        self.fc_size = 2000
        for i in range(len(self.channels)):
                self.fc_size = ((self.fc_size - self.kernels[i]+2*padding[i])/self.strides[i])+1
        print(self.fc_size)


        ##Encoder_layers
        self.encoder_layers = torch.nn.ModuleList()
        self.encoder_norms = torch.nn.ModuleList()

        ##Insert layers to Encoder_layers
        layer_count = 0
        for parameters in zip(self.channels, self.kernels, self.strides, self.padding):
                if layer_count == 0:
                        self.encoder_layers.append(torch.nn.Conv1d(in_channels=4, out_channels=parameters[0], kernel_size=parameters[1], stride=parameters[2], padding=parameters[3]))
                        self.encoder_norms.append(torch.nn.BatchNorm1d(parameters[0]))
                else:
                        self.encoder_layers.append(torch.nn.Conv1d(in_channels=self.channels[layer_count-1], out_channels=parameters[0], kernel_size=parameters[1], stride=parameters[2], padding=parameters[3]))
                        self.encoder_norms.append(torch.nn.BatchNorm1d(parameters[0]))
                layer_count += 1
	
        #self.encode_fc1 = torch.nn.Linear(int(self.fc_size*self.channels[-1]), 250)
        #self.encode_fc2 = torch.nn.Linear(250, 100)
        #self.encode_mu = torch.nn.Linear(100, self.latent_size)
        #self.encode_logvar = torch.nn.Linear(100, self.latent_size)

        self.encode_mu = torch.nn.Linear(int(self.fc_size*self.channels[-1]), self.latent_size)
        self.encode_logvar = torch.nn.Linear(int(self.fc_size*self.channels[-1]), self.latent_size)

        ##Decoder layers
        self.latent = torch.nn.Linear(self.latent_size, int(self.fc_size*self.channels[-1]))
        #self.latent = torch.nn.Linear(self.latent_size, 100)
        #self.decode_fc2 = torch.nn.Linear(100, 250)
        #self.decode_fc1 = torch.nn.Linear(250, int(self.fc_size*self.channels[-1])) 	


        self.decoder_layers = torch.nn.ModuleList()
        self.decoder_norms = torch.nn.ModuleList()

        ##Insert layers to Decoder layers
        layer_count = 0
        for parameters in zip(self.channels[::-1], self.kernels[::-1], self.strides[::-1], self.padding[::-1]):
                if len(self.channels) > 1+layer_count:
                        self.decoder_layers.append(torch.nn.ConvTranspose1d(in_channels=parameters[0], out_channels=self.channels[::-1][layer_count+1], kernel_size=parameters[1], stride=parameters[2], padding=parameters[3]))
                        self.decoder_norms.append(torch.nn.BatchNorm1d(self.channels[::-1][layer_count+1]))
                layer_count += 1
        self.output_layer = torch.nn.ConvTranspose1d(in_channels=self.channels[0], out_channels=4, kernel_size=self.kernels[0], stride=self.strides[0], padding=self.padding[0])
        
        
    
    def reparameterize(self, mu, logvar):
    
        """
        This function reparameterizes the latent space by using a normal-gaussian of size [batchsize, latent_vector].
        The normal-gaussian is scaled with the std, and then mu is added. 
        This enables the VAE to backpropagate since the std and mu vector are not randomly sampled from a gaussian distribution.
        Instead they are scaled and added.
        """
        logvar = torch.exp(logvar/2)
        if self.cuda_flag:
                epsilon = torch.randn((mu.size())).float().cuda()
        else:
                epsilon = torch.randn((mu.size())).float()

        latent_vector = torch.mul(epsilon, logvar) + mu
        
        return latent_vector
    
    def encode(self, x):
        '''
        The first part of the model, which encodes an input into a latent representation.
        '''
        #tensors = list()

        for layer, batchnorm in zip(self.encoder_layers, self.encoder_norms):
                x = batchnorm(self.dropout(self.relu(layer(x))))
                #tensors.append(x)
        x = x.view(-1, int(self.fc_size*self.channels[-1]))
        #x = self.dropout(self.relu(self.encode_fc1(x)))
        #x = self.dropout(self.relu(self.encode_fc2(x)))
        mu = self.encode_mu(x)
        logvar = self.softplus(self.encode_logvar(x))
        
        return self.reparameterize(mu, logvar), mu, logvar
    
    def decode(self, x):
        '''
        The second part of the model, where the latent representation is decoded.
        '''

        x = self.latent(x)
        #x = self.dropout(self.relu(self.decode_fc2(x)))
        #x = self.dropout(self.relu(self.decode_fc1(x)))
        x = x.view(-1, self.channels[-1], int(self.fc_size))
        if len(self.channels) > 1:
                for layer, batchnorm in zip(self.decoder_layers, self.decoder_norms):
                        x = batchnorm(self.dropout(self.relu(layer(x))))
        
        return self.output_layer(x)
    
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
        ce_loss = torch.nn.functional.cross_entropy(reconstructed_x+1e-10, x.argmax(1))/math.log(4)
       
        KLD_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean())
        total_loss = ce_loss + KLD_loss *(self.alpha/self.latent_size)
        return total_loss, ce_loss, KLD_loss
















