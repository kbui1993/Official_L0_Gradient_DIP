# Official_L0_Gradient_DIP

Official code for the paper "Deep Image Prior with L0 Gradient Regularizer for Image Smoothing" (Arxiv link will be posted soon.). The official version will be published at ICASSP 2026.

# Python Version Note
The code only works for Python 3.8 because the cpython module was created under Python 3.8 environment. If you would like it to work for different Python version, please go to the following link and generate a new cpython module using your current Python version: https://github.com/nhatthanhtran/l0_min_by_fuse_region. To do so, perform the following steps:

1. Clone the repository.
2. Run the sh script ./wrapper_builder/cmake.sh.
3. "build" subdirectory should be made. The cpython file should be ./wrapper_builder/python/l0_module.cpython-[PythonVersion]-x86_64-linux-gnu.

# Running the Algorithm

To run the algorithm, type the following, for example, in the command line:

```
CUDA_VISIBLE_DEVICES=0 python DIP_L0_Region_Fuse.py --Lambda 0.025 --Beta 2.00 --root_path './data/clipart/corrupted/' --fname clip1.jpg --num_iter 100 --inner_iter 25 --lr 1e-03
```

Check out the scripts folder for more examples.
