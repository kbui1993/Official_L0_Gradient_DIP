import l0_module
import argparse
import numpy as np
parser = argparse.ArgumentParser()


parser.add_argument('--np_input', type=str, default = 'check.npy')
parser.add_argument('--reg', type = float, default = 0.05)
parser.add_argument('--np_output', type=str, default = 'output.npy')

args = parser.parse_args()


np_input_file = args.np_input
reg = args.reg

np_input = np.load(np_input_file)

l0_result = l0_module.l0_norm_float(np_input, reg, 32, 50, True)

np.save(args.np_output, l0_result)
