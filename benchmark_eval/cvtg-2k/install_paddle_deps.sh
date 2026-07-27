set -e

echo "=== Activating textcrafter_eval environment ==="
# Robust conda activation: works for anaconda/miniconda and arbitrary prefixes
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    # fallback to common locations
    for cpath in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$cpath" ]; then
            . "$cpath"
            break
        fi
    done
fi
conda activate textcrafter_eval

SETUPTOOLS_CONSTRAINT="$(mktemp)"
echo "setuptools<81" > "$SETUPTOOLS_CONSTRAINT"
export PIP_CONSTRAINT="$SETUPTOOLS_CONSTRAINT"
pip install "setuptools<81"

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Verifying basic environment ==="
echo "Python version: $(python --version)"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"
echo "OpenCV version: $(python -c 'import cv2; print(cv2.__version__)')"

echo "=== Installing PaddlePaddle GPU version (CUDA 11.8) ==="
python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

echo "=== Installing PaddleOCR (using --no-deps to avoid dependency conflicts) ==="
pip install paddleocr==2.10.0 --no-deps

echo "=== Installing opencv-contrib-python (handling possible conflicts) ==="
pip install opencv-contrib-python --no-deps || echo "opencv-contrib-python installation failed, continuing..."

echo "=== Setting environment variables ==="
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

echo "=== Testing all dependency imports ==="
python -c "
import sys
print('=== Import Test Results ===')

# Test PyTorch
try:
    import torch
    print('✓ PyTorch imported successfully, version:', torch.__version__)
    print('  CUDA available:', torch.cuda.is_available())
except Exception as e:
    print('✗ PyTorch import failed:', e)

# Test PaddlePaddle  
try:
    import paddle
    print('✓ PaddlePaddle imported successfully, version:', paddle.__version__)
except Exception as e:
    print('✗ PaddlePaddle import failed:', e)

# Test OpenCV
try:
    import cv2
    print('✓ OpenCV imported successfully, version:', cv2.__version__)
except Exception as e:
    print('✗ OpenCV import failed:', e)

# Test PaddleOCR
try:
    import paddleocr
    print('✓ PaddleOCR imported successfully')
except Exception as e:
    print('✗ PaddleOCR import failed:', e)

# Test other critical dependencies
try:
    from PIL import Image
    print('✓ PIL imported successfully')
except Exception as e:
    print('✗ PIL import failed:', e)

try:
    import open_clip
    print('✓ OpenCLIP imported successfully')
except Exception as e:
    print('✗ OpenCLIP import failed:', e)

try:
    import t2v_metrics
    print('✓ T2V-metrics imported successfully')
except Exception as e:
    print('✗ T2V-metrics import failed:', e)
"

echo "=== Installation completed ==="
echo "If all the above dependencies show ✓, then installation is successful!"
echo "If PaddleOCR still has issues, please run the following command to add environment variables to your shell configuration:"
echo "echo 'export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc"
