#!/bin/bash
echo "======================================"
echo "Running Fortran Wrapper..."
echo "======================================"
python run_ebm_wrapper.py

echo ""
echo "======================================"
echo "Running Native Python Version..."
echo "======================================"
python ebm_python_native.py

echo ""
echo "======================================"
echo "Running JAX Version..."
echo "======================================"
python ebm_python_jax.py

echo ""
echo "======================================"
echo "All runs completed successfully!"
echo "Check the figures directory for the plots."
echo "======================================"
