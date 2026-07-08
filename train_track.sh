#!/bin/bash 

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $ROOT_DIR

# activate the virtual enviroment
if [ -d "venv" ]; then
    echo "Virtual environment found"
    echo "Activating virtual enviroment"
else
    echo "ERROR: venv not found"
    echo "Creating a new virtual enviroment..."
    python3 -m venv venv
    pip install -r requirements.txt
fi

source venv/bin/activate

cd src

python3 prepare_dataset.py

python3 train.py

python3 tracking_2d.py

python3 evaluate_2d.py

python3 visualize_2d.py