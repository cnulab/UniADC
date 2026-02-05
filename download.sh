mkdir ckpt
cd ckpt
pip install gdown

gdown --folder https://drive.google.com/drive/folders/1dQeSFqpQg_NSFLhd3ChuSJCZ0zCSquh8
gdown --folder https://drive.google.com/drive/folders/1hCYIjeRGx3Zk9WZtQf0s3nDGfeiwqTsN
gdown --folder https://drive.google.com/drive/folders/1KPFFYblnovk4MU74OCBfS1EZU_jhBsse

huggingface-cli download ZhengPeng7/BiRefNet --local-dir ./birefnet --resume-download
wget https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit_20m_4xe.pth
