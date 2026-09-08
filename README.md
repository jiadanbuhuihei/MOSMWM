1. pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
检验：
python3 - <<EOF
import torch
import torchvision
import torchaudio

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)

print(torch.cuda.get_device_name(0))
EOF

2. pip install xformers==0.0.29.post1

检验： 
python3 - <<EOF
import torch
import xformers

print(torch.__version__)
print(xformers.__version__)
EOF

3. pip install ninja packaging
pip install flash-attn==2.7.3 --no-build-isolation

检验：
python3 - <<EOF
import flash_attn
print("flash-attn OK")
EOF

4. pip install -r requirements-rest.txt