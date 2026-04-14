#!/bin/bash

echo "🧹 Cleaning up..."
rm -rf build/ depth_diff_gaussian_rasterization_min.egg-info
find . -name '*.so' -delete

# echo "📦 Installing in editable mode..."
# pip install -e . --force-reinstall

# echo "🔧 Building CUDA/C++ extensions..."
# python setup.py build_ext --inplace

# echo "✅ Done."
