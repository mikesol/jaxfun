sudo apt-get update
sudo apt-get install joe screen libsndfile-dev awscli python3.8-venv git-lfs -y
aws s3 cp s3://meeshkan-datasets/unsilenced/ ~/data/ --exclude "*" --include "day[12]/nt1_*" --include "day[12]/67_*" --recursive
git clone https://github.com/mikesol/jaxfun && cd jaxfun
python3 -m venv .venv && source .venv/bin/activate
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements-tpu.txt
screen -S train
# clear && git pull && python train_faux_rnn.py
# clear && git pull && python train_rnn.py
clear && git pull && git lfs pull && python train_tcn.py
# clear && git pull && python train_mixer.py