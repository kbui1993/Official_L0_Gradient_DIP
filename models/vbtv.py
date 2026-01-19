import torch
import torch.nn as nn
from math import sqrt
import torch.nn.init
import kornia
from .common import *
from models import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

class DIP(nn.Module):

	def __init__(self,input_depth, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(DIP,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.input_depth=input_depth

	def forward(self, input):

		return self.net(input)


class VectorialTotalVariation(nn.Module):	

	def __init__(self,input_depth, pad, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride', need_sigmoid=True):
		super(VectorialTotalVariation,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=need_sigmoid, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth=input_depth

	def forward(self, input):

		output = self.net(input)

		differential = kornia.filters.SpatialGradient()(output)

		
		differential_squared = torch.mul(differential,differential)
		# print(differential_squared.shape)
		norm_squared = torch.sum(torch.sum(differential_squared,dim=1),dim=1)

		norm_squared_regularized = norm_squared+0.00001*torch.ones((1,self.height,self.width)).type(torch.cuda.FloatTensor)
		norm = torch.sqrt(norm_squared_regularized)
		return output, norm