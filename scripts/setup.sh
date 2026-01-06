python3 -m venv .venv
source .venv/bin/activate

mkdir -p outputs models checkpoints

pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
