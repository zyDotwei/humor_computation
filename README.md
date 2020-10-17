# humor_computation
CCL2020——幽默计算

## 名次
* 初赛: 第二名
* 复赛: 第六名

## 目录结构 
```
. 
├── README.md
├── configs                             # 配置文件
├── cross_validation.py                 # K折验证
├── logging                             # 日志文件
├── loss                                # 损失函数文件
├── models                              # 模型文件
├── pretrain_models                     # 预训练文件
│   ├── ERNIE                           # ch
│   └── bert-base-uncased               # en
├── processors                          #数据加载和处理
├── requirements.txt
├── results
├── run.py                              # 主程序入口
├── train_dev_data                      # data
└── utils                               # utils
```

## 快速开始
``` python
# 以加载base_cn_config.py为例
python run.py --config base_cn_config
```
