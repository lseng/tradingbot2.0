#!/bin/bash
# RunPod Training Setup Script
# Paste this entire script into RunPod terminal

set -e  # Exit on error

echo "=== TradingBot 2.0 Training Setup ==="
echo ""

# 1. Clone repository
echo "[1/5] Cloning repository..."
cd /workspace
git clone https://github.com/lseng/tradingbot2.0.git
cd tradingbot2.0

# 2. Install Git LFS and pull data
echo "[2/5] Installing Git LFS and downloading data (273MB)..."
apt-get update -qq && apt-get install -y git-lfs -qq
git lfs install
git lfs pull

# 3. Verify data
echo "[3/5] Verifying data files..."
ls -lh data/historical/MES/
if [ ! -f "data/historical/MES/MES_1s_2years.parquet" ]; then
    echo "ERROR: Data file not found!"
    exit 1
fi

# 4. Install Python dependencies
echo "[4/5] Installing Python dependencies..."
pip install -q -r src/ml/requirements.txt

# 5. Check GPU
echo "[5/5] Checking GPU..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start training, run:"
echo "  python src/ml/train_scalping_model.py --model lstm --epochs 100"
echo ""
echo "To run backtest after training:"
echo "  python scripts/run_backtest.py --walk-forward --output ./results/walkforward"
echo ""
