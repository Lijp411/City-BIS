#!/bin/bash

# Compile cpp subsampling
cd /content/gdrive/MyDrive/KPConv_3/KPConv-PyTorch-master/cpp_wrappers/cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd /content/gdrive/MyDrive/KPConv_3/KPConv-PyTorch-master/cpp_wrappers/cpp_neighbors
python3 setup.py build_ext --inplace
cd ..