cd /root
git clone http://Naplessss:xxx@github.com/Naplessss/MT.git
cd MT
echo "copy images from blob..."
cp /mnt/epblob/zhgao/MT/bms-molecular-translation.zip .
cp /mnt/epblob/zhgao/MT/tokenizer.pth .
cp /mnt/epblob/zhgao/MT/train.pkl .
echo "unzip images ..."
unzip -q bms-molecular-translation.zip
echo "install packages..."
cd src
apt update
apt install -y libgl1-mesa-dev
pip install -r requirements.txt
# echo "preprocess..."
# python preprocess.py
# echo "save..."
# cp ../tokenizer.pth /mnt/epblob/zhgao/MT
# cp ../train.pkl /mnt/epblob/zhgao/MT

python -m torch.distributed.launch --nproc_per_node 1 baseline_ep.py --model_name tf_efficientnet_b7 --batch_size 4 --size 448 --debug 1 --meta_info fordebug