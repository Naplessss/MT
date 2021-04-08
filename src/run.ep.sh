git clone http://Naplessss:GZF1233gzf@github.com/Naplessss/MT.git
cd MT
pip install -r requirements.txt
echo "copy images from blob..."
cp /mnt/epblob/zhgao/MT/bms-molecular-translation.zip .
echo "unzip images ..."
unzip -q bms-molecular-translation.zip
echo "preprocess..."
python preprocess.py
echo "save..."
cp tokenizer.pth /mnt/epblob/zhgao/MT
cp train.pkl /mnt/epblob/zhgao/MT

