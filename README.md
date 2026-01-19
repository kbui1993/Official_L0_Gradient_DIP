# Official_L0_Gradient_DIP

Official code for the paper "Deep Image Prior with L0 Gradient Regularizer for Image Smoothing" (Arxiv link will be posted soon.). The official version will be published at ICASSP 2026.

# Python Version Note
The code only works for Python 3.8 because the cpython module was created under Python 3.8 environment. If you would like it to work for different Python version, please go to the following link and generate a new cpython module using your current Python version: https://github.com/nhatthanhtran/l0_min_by_fuse_region. To do so, perform the following steps:

1. Clone the repository.
2. Run the sh script ./wrapper_builder/cmake.sh.
3. "build" subdirectory should be made. The cpython file should be ./wrapper_builder/python/l0_module.cpython-[PythonVersion]-x86_64-linux-gnu.

