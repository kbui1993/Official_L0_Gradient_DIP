import os
import numpy as np
from models import *
from models.vbtv import *
import argparse
import torch
import torch.optim
import time

from utils.denoising_utils import *

import uuid

filename = str(uuid.uuid4())
input_filename = filename+'.npy'
output_filename = filename+'_output.npy'

torch.cuda.empty_cache()

# Set up parser
parser = argparse.ArgumentParser(description='[DIP] Image Segmentation')

parser.add_argument('--fname', type=str, default='image1_copy2.png', help='image file name')
parser.add_argument('--root_path', type=str, default='data/segment/', help='root path of the file')
parser.add_argument('--Beta', type=float, default=1.5, help='beta value')
parser.add_argument('--Lambda', type=float, default=0.05, help='lambda value')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--exp_weight', type=float, default=0.90, help='exp weight')
parser.add_argument('--inner_iter', type=int, default=50, help='number of iterations')
parser.add_argument('--num_iter', type=int, default=99, help='number of iterations')

args = parser.parse_args()

# Load clean image
fname = args.root_path + args.fname
img_pil = get_image(fname,-1)[0]


# Setup loss
mse = torch.nn.MSELoss().type(dtype)
mae = torch.nn.L1Loss().type(dtype)
ml2 = torch.nn.MSELoss(reduction='sum').type(dtype)


def prox(g, w, beta, Lambda):
	"""
	This function solves the following L0-gradient minimization problem using L0 Region Fusion:

	min_v (2*lambda/beta)*\|\grad v\|_0 + \|v-(g-w^t/beta)\|_2^2.

	Input:
		@g: output of deep image prior
		@w: dual variable
		@beta: ADMM penalty parameter
		@Lambda: L0 gradient regularizaiton parameter
	"""

    # convert g to numpy array and save
	g1 = (g-w/beta).squeeze(0).permute(1,2,0).detach().cpu().numpy()
	np.save(input_filename, g1.astype('double'))

    # run l0 region fusion on command terminal in separate process
	reg = 2*Lambda/beta
	cmd = f'python l0_region_fusion.py --np_input {input_filename} --reg {reg} --np_output {output_filename}'
	os.system(cmd)

    # load result and convert to tensor
	l0_img = np.load(output_filename)
	l0_img_torch =torch.tensor(np.transpose(l0_img,(2,0,1))).unsqueeze(0).type(dtype)
	return l0_img_torch


def closure(img_pil, args):
	"""
	This function runs L0-gradient regularized DIP using Region_Fuse.
	Input:
		@img_pil: input image in PIL format
		@args: set of arguments initialized by argument parser
	Output:
		@out_img_avg: weighted average output from DIP
		@v_mat: auxiliary variable for DIP output created from Region_Fuse
	"""
	# create save directory
	fname = args.fname[:args.fname.find(".")]
	save_dir = "./" + args.root_path + fname + '/'
	os.makedirs(save_dir, exist_ok=True)

	# Set parameters
	Lambda = args.Lambda
	Beta = args.Beta
	exp_weight = args.exp_weight
	lr = args.lr
	inner_iter = args.inner_iter
	num_iter = args.num_iter

	# convert to relevant data type
	img_np = pil_to_np(img_pil)
	img_torch = np_to_torch(img_np).type(dtype)

	# for weighted averaging of outputs
	out_img_avg = np.copy(img_np)

	# Setup input and network
	INPUT = 'noise'
	pad = 'reflection'	
	OPT_OVER='net'
	input_depth = 32
	net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
	full_net = VectorialTotalVariation(input_depth, pad, height=img_pil.size[1], width=img_pil.size[0], upsample_mode='bilinear' ).type(dtype)

	#initialize auxiliary variable v and lagrange multiplier w
	v_mat = torch.rand((1,3,img_pil.size[1],img_pil.size[0]), device='cuda', requires_grad=False)
	w_mat = torch.rand((1,3,img_pil.size[1],img_pil.size[0]), device='cuda', requires_grad=False)

	# define optimizer
	parameters = get_params(OPT_OVER, full_net, net_input, input_depth)
	optimizer  = torch.optim.Adam(parameters, lr)

	old_net_output = 0

	# Run DIP
	for j in range(num_iter):
		
		#update theta
		for k in range(inner_iter):
			optimizer.zero_grad()
			net_output, _ = full_net(net_input)

			# compute loss
			loss_dataterm = ml2(net_output,img_torch) #fidelity term
			loss_regularizer = torch.sum(torch.mul(w_mat, v_mat - net_output)) + Beta/2*ml2(v_mat,net_output) #additional terms by ADMM
			total_loss = loss_dataterm + loss_regularizer 
			total_loss.backward(retain_graph=True)

			#update network
			optimizer.step()

		#obtain updated output after updating theta
		with torch.no_grad():
			net_output, _ = full_net(net_input)

		#update v
		v_mat = prox(net_output.detach(), w_mat.detach(), Beta, Lambda)

		#update w
		w_mat = w_mat + Beta*(torch.sub(v_mat,net_output.detach()))

		# weighted average of output
		out_img_avg = out_img_avg * exp_weight + net_output.detach().cpu().numpy()[0] * (1 - exp_weight)

		# save results every 50 iterations
		if ((j+1)%50 == 0) or (j == num_iter-1):

			out_img_avg_pil = np_to_pil(out_img_avg)
			v_img = np_to_pil(v_mat.detach().cpu().numpy()[0])
			fname_save = f"{save_dir}_seg_{fname}_lam{args.Lambda}_B{args.Beta}_lr{args.lr}_i{str(j+1)}_j{args.inner_iter}_Region_Fuse.png"
			out_img_avg_pil.save(fname_save)

			fname_save = f"{save_dir}_seg_v_{fname}_lam{args.Lambda}_B{args.Beta}_lr{args.lr}_i{str(j+1)}_j{args.inner_iter}_Region_Fuse.png"
			v_img.save(fname_save)
		
		print('Number of Iterations: %d' %j)

	return out_img_avg, v_mat


# set up timer 
curr_time = time.time()

# run L0-regularized DIP
out_img_avg, v_mat = closure(img_pil, args)

# end timer
end_time = time.time()

#return run time
print(f"The run time is: {(end_time-curr_time)} seconds")

os.remove(input_filename)
os.remove(output_filename)