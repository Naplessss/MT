# TNT model运行指南
conda 安装rdkit并激活环境
安装以下库(直接pip安装即可):timm,scikit-learn,matplotlib,torch,python-Levenshtein
运行过程需要在服务器2或1中下载指定的四个预处理好的文件(df_train.more.csv.pickle, tokenizer.stoi.pickle, df_fold.csv, test_orientation.csv) /home/covpreduser/MT

bms.py需指定data_dir整体数据文件目录

python run_train.py即可
