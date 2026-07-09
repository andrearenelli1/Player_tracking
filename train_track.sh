#!/bin/bash 

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $ROOT_DIR

# activate the virtual enviroment
if [ -d "venv" ]; then
    echo "Virtual environment found"
else
    echo "ERROR: venv not found"
    echo "Creating a new virtual enviroment..."
    python3 -m venv venv
    pip install -r requirements.txt
fi

echo "Activating virtual enviroment"

source venv/bin/activate

cd src

echo "---------------- Executing file: prepare_dataset.py ----------------" 

python3 prepare_dataset.py

echo "---------------- Executing file: train.py ----------------"

python3 train.py

echo "---------------- Executing file: tracking_2d.py ----------------"

python3 tracking_2d.py

echo "---------------- Executing file: evaluate_2d.py ----------------"

python3 evaluate_2d.py

echo "---------------- Executing file: visualize_2d.py ----------------"

python3 visualize_2d.py