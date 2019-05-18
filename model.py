import torch
import numpy as np
import math
#from torch.jit_internal import weak_script, List

class CVAE(torch.nn.Module):
	def __init__(self, latent_size, alpha, dropout, channels, kernels, strides, padding, cuda_flag, gumbel, fragment_size):
		##Initialization
		super(CVAE, self).__init__()
		self.dropout = torch.nn.Dropout(p=dropout)
		self.relu = torch.nn.LeakyReLU()
		self.softplus = torch.nn.Softplus()
		self.softmax = torch.nn.Softmax(dim=1)
		self.alpha = alpha
		self.cuda_flag = cuda_flag
		self.gumbel = gumbel

		self.latent_size = latent_size	
		self.channels = channels
		self.kernels = kernels
		self.strides = strides
		self.padding = padding

		self.fc_size = fragment_size
		for i in range(len(self.channels)):
			self.fc_size = ((self.fc_size - self.kernels[i]+2*padding[i])/self.strides[i])+1
		self.fc_size = int(self.fc_size)
		print(self.fc_size)


		##Encoder_layers
		self.encoder_layers = torch.nn.ModuleList()
		self.encoder_norms = torch.nn.ModuleList()

		##Insert layers to Encoder_layers. Specified by user
		layer_count = 0
		for parameters in zip(self.channels, self.kernels, self.strides, self.padding):
			if layer_count == 0:
				self.encoder_layers.append(torch.nn.Conv1d(in_channels=4, out_channels=parameters[0], kernel_size=parameters[1], stride=parameters[2], padding=parameters[3]))
				self.encoder_norms.append(torch.nn.BatchNorm1d(parameters[0]))
			else:
				self.encoder_layers.append(torch.nn.Conv1d(in_channels=self.channels[layer_count-1], out_channels=parameters[0], kernel_size=parameters[1], stride=parameters[2], padding=parameters[3]))
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
		for parameters in zip(self.channels[::-1], self.kernels[::-1], self.strides[::-1], self.padding[::-1]):
			if len(self.channels) > 1+layer_count:
				self.decoder_layers.append(torch.nn.ConvTranspose1d(in_channels=parameters[0], out_channels=self.channels[::-1][layer_count+1], kernel_size=parameters[1], stride=parameters[2], padding=parameters[3]))
				self.decoder_norms.append(torch.nn.BatchNorm1d(self.channels[::-1][layer_count+1]))
			layer_count += 1
		self.output_layer = torch.nn.ConvTranspose1d(in_channels=self.channels[0], out_channels=4, kernel_size=self.kernels[0], stride=self.strides[0], padding=self.padding[0])

	#@torch._jit_internal.weak_script    
	def gumbel_softmax(self, logits, tau=1, hard=False, eps=1e-10, dim=-1):
	# https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
		if eps != 1e-10:
			warnings.warn("`eps` parameter is deprecated and has no effect.")

		gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
		gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
		y_soft = (gumbels+eps).softmax(dim) #maybe log should be removed here...

		if hard:
			# Straight through.
			index = y_soft.max(dim, keepdim=True)[1]
			y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
			ret = y_hard - y_soft.detach() + y_soft
		else:
			# Reparametrization trick.
			ret = y_soft
		return ret
	#@torch._jit_internal.weak_script
	def _sample_gumbel(self, logits, shape, eps=1e-10, out=None):
		# type: (List[int], float, Optional[Tensor]) -> Tensor
		"""
		Sample from Gumbel(0, 1)
		based on
		https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
		(MIT license)
		"""
		#print(out)
		if out is None:
			U = torch.rand(shape)
		else:
			#U = torch.jit._unwrap_optional(out).resize_(shape).uniform_()
			U = torch.zeros(shape, dtype=logits.dtype)
			U = U.uniform_().cuda()
		return - torch.log(eps - torch.log(U + eps))

	#@torch._jit_internal.weak_script
	def _gumbel_softmax_sample(self, logits, tau=1, eps=1e-10, dim=-1):
	# type: (Tensor, float, float) -> Tensor
		"""
		Draw a sample from the Gumbel-Softmax distribution
		based on
		https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
		(MIT license)
		"""
		dims = logits.dim()
		gumbel_noise = self._sample_gumbel(logits, logits.size(), eps=eps, out=torch.empty_like(logits)).cuda()
		y = logits + gumbel_noise
		#torch.nn.functional.Logsoftmax
		return torch.nn.functional.log_softmax(y / tau, dim=dim) #should this be log?
	def gumbel_softmax_true(self, logits, tau=1., hard=False, eps=1e-10, dim=-1):
		# type: (Tensor, float, bool, float) -> Tensor
		r"""
		Sample from the Gumbel-Softmax distribution and optionally discretize.
		Args:
		logits: `[batch_size, num_features]` unnormalized log probabilities
		tau: non-negative scalar temperature
		hard: if ``True``, the returned samples will be discretized as one-hot vectors,
		but will be differentiated as if it is the soft sample in autograd
		Returns:
		Sampled tensor of shape ``batch_size x num_features`` from the Gumbel-Softmax distribution.
		If ``hard=True``, the returned samples will be one-hot, otherwise they will
		be probability distributions that sum to 1 across features
		Constraints:
		- Currently only work on 2D input :attr:`logits` tensor of shape ``batch_size x_num_features``
		Based on
		https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
		(MIT license)
		"""
		y_soft = self._gumbel_softmax_sample(logits, tau=tau, eps=eps, dim=dim)
		if hard:
			_, k = y_soft.max(-1)
			# this bit is based on
			# https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
			y_hard = torch.zeros(shape, dtype=logits.dtype, device=logits.device).scatter_(-1, k.view(-1, 1), 1.0)
			# this cool bit of code achieves two things:
			# - makes the output value exactly one-hot (since we add then
			#   subtract y_soft value)
			# - makes the gradient equal to y_soft gradient (since we strip
			#   all other gradients)
			y = y_hard - y_soft.detach() + y_soft
		else:
			y = y_soft.cuda()
		return y

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

	
	def calc_loss(self, reconstructed_x, x, mu, logvar, tau=1):
		'''
		Function for calculating ce_loss and KLD_loss.
		'''
		if self.gumbel:
			gumb = self.gumbel_softmax_true(reconstructed_x, tau=tau, hard=False, dim=1)
			ce_loss = torch.nn.functional.nll_loss(gumb, x.argmax(1))/math.log(4)
		else:
			ce_loss = torch.nn.functional.cross_entropy(reconstructed_x+1e-10, x.argmax(1))/math.log(4)
		KLD_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean())
		total_loss = ce_loss + KLD_loss *(self.alpha/self.latent_size)
		return total_loss, ce_loss, KLD_loss
