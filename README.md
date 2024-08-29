# 两阶段中文AMR解析方法

**中文** | [English](https://github.com/PKUnlp-icler/Two-Stage-CAMRP/blob/main/README_en.md)

论文 "[A Two-Stage Method for Chinese AMR Parsing](https://arxiv.org/abs/2209.14512)" @ CAMRP-2022 & CCL-2022 的模型及训练代码。

这里有一份讲解PPT可以参考[google drive](https://drive.google.com/file/d/1wrqXTRL2z95LKqcGOdPSj2YLFBdEdtdw/view?usp=share_link)

欢迎issue有关结果复现过程中的任何问题 ~~



<br>

<div align=center>
<b>中文AMR解析的一个例子</b>
<br>

<img width="800" src="./camr.PNG"/>

</div>

<br><br>


<div align=center>
<b>两阶段模型示意图</b>
<br>
<img width="800" src="./model.PNG"/> 


</div>


## 📕准备工作

python: 3.7.11

```shell
conda create -n camrp python=3.7
pip install -r requirement.txt # 确保你的torch和cuda版本匹配, 我们使用的是 torch-1.10.1+cu113
```

### 重要事项
你需要把transformers库中的一些文件替换为我们修改后的版本
- 将 `/miniconda3/envs/camrp/lib/python3.7/site-packages/transformers/models/bert/modeling_bert.py` 换成 `./src/modeling_bert.py`
- 将 `/miniconda3/envs/camrp/lib/python3.7/site-packages/transformers/modeling_outputs.py` 换成 `./src/modeling_outputs.py`
- 将 `/miniconda3/envs/camrp/lib/python3.7/site-packages/transformers/trainer.py` 换成  `./src/trainer.py`


### 复现论文中的结果

如果想要复现论文中的结果，可以直接跳转到 ''[推理](https://github.com/PKUnlp-icler/Two-Stage-CAMRP/blob/main/README.md#%E6%8E%A8%E7%90%86)'' 部分(不需要数据集准备、预处理和训练)


## 📕数据集准备

在开始之前，您需要从CAMRP-2022收集CAMR和CAMR元组数据，并将它们放在 `. /datasets` 下。

`./datasets/vocabs` 由我们提供

数据结构如下

```bash
/Two-Stage-CAMRP/datasets
├── camr
│   ├── camr_dev.txt
│   └── camr_train.txt
├── camr_tuples
│   ├── tuples_dev.txt
│   └── tuples_train.txt
└── vocabs
    ├── concepts.txt
    ├── eng_predicates.txt
    ├── nodes.txt
    ├── predicates.txt
    ├── ralign.txt
    └── relations.txt
```

## 📕预处理

首先，我们需要在两阶段方法中将中文AMR标注转换为5个不同的子任务。

任务包括
1. Surface Tagging 
2. Normalization Tagging
3. Non-Aligned Concept Tagging
4. Relation Classification 
5. Relation Aligment Classification

`./scripts/preprocess.py` 将自动为五个任务生成预处理数据，你需要设置脚本中CAMR和CAMR元组数据的路径

（推荐）我们准备了处理好的数据在 `./preprocessed` 下，你可以直接使用他。

完全处理好的数据应该长下面的样子：

```bash
/Two-Stage-CAMRP/preprocessed
├── non_aligned_concept_tagging
│   ├── dev.extra_nodes
│   ├── dev.extra_nodes.tag
│   ├── dev.sent
│   ├── train.extra_nodes
│   ├── train.extra_nodes.tag
│   ├── train.sent
│   └── train.tag.extra_nodes_dict
├── normalization_tagging
│   ├── dev.p_transform
│   ├── dev.p_transform_tag
│   ├── dev.sent
│   ├── train.p_transform
│   ├── train.p_transform_tag
│   └── train.sent
├── relation_classification
│   ├── dev.4level.relations
│   ├── dev.4level.relations.literal
│   ├── dev.4level.relations_nodes
│   ├── dev.4level.relations_nodes_no_r
│   ├── dev.4level.relations.no_r
│   ├── relation_alignment_classification
│   │   ├── dev.4level.ralign.relations
│   │   ├── dev.4levelralign.relations.literal
│   │   ├── dev.4level.ralign.relations_nodes
│   │   ├── train.4level.ralign.relations
│   │   ├── train.4levelralign.relations.literal
│   │   └── train.4level.ralign.relations_nodes
│   ├── train.4level.relations
│   ├── train.4level.relations.literal
│   ├── train.4level.relations_nodes
│   ├── train.4level.relations_nodes_no_r
│   └── train.4level.relations.no_r
└── surface_tagging
    ├── dev.sent
    ├── dev.tag
    ├── train.sent
    └── train.tag

5 directories, 33 files
```

## 📕训练

```bash


export CUDA_VISIBLE_DEVICES=0
# train following tasks individually. It takes about 1 day to train all tasks on a single A40 GPU

cd scripts/train

python train_surface_tagging.py
python train_normalization_tagging.py
python train_non_aligned_tagging.py


python train_relation_classification.py
python train_relation_alignment_classification.py

# the trained models will be saved under /Two-Stage-CAMRP/models

```


## 📕推理

要复现我们的结果，你需要下载五个模型，从 [Google Drive](https://drive.google.com/drive/folders/1zkWlgyVYvoZTcXzZsC4GABFRKXHr9pUa?usp=sharing) 或者[北大网盘](https://disk.pku.edu.cn:443/link/B9082FEBBE8AF0210486969549C65D52) 或者通过上面的脚本进行训练. 在得到五个模型后，将模型的文件夹们放到 `./models/trained_models` 中. 

下载完后，数据结构应该如下
```bash
/Two-Stage-CAMRP/models
└─trained_models
    ├─non_aligned_tagging
    │  └─checkpoint-1400
    ├─normalization_tagging
    │  └─checkpoint-650
    ├─relation_align_cls
    │  └─checkpoint-33000
    ├─relation_cls
    │  └─checkpoint-32400
    └─surface_tagging
        └─checkpoint-125200
```


然后运行下面的命令，就可以得到最终在testA测试集上的[结果](https://github.com/chenllliang/Two-Stage-CAMRP/blob/main/result/testA.with_r.with_extra.relation.literal.sync_with_no_r.with_func_words.camr_tuple),结果保存在 `./results` 文件夹下。

```bash
export CUDA_VISIBLE_DEVICES=0

cd scripts/eval

python inference_surface_tagging.py ../../models/trained_models/surface_tagging/checkpoint-125200 ../../test_A/test_A_with_id.txt ../../result/testA

python inference_normalization_tagging.py ../../models/trained_models/normalization_tagging/checkpoint-650 ../../test_A/test_A_with_id.txt ../../result/testA

python inference_non_aligned_tagging.py ../../models/trained_models/non_aligned_tagging/checkpoint-1400 ../../test_A/test_A_with_id.txt ../../result/testA


bash inference.sh ../../result/testA.surface ../../result/testA.norm_tag ../../result/testA.non_aligned ../../test_A/test_A.txt ../../result/testA ../../models/trained_models/relation_cls/checkpoint-32400 ../../models/trained_models/relation_align_cls/checkpoint-33000

```




## 📕获得AlignSmatch分数

AlignSmatch 工具来自 [CAMRP 2022](https://github.com/GoThereGit/Chinese-AMR/tree/main/tools)

```
cd ./Chinese-AMR/tools

python Align-smatch.py -lf ../data/test/test_A/max_len_testA.txt -f ../../result/testA.with_r.with_extra.relation.literal.sync_with_no_r.with_func_words.camr_tuple ../../test_A/gold_testa.txt --pr

# Result for the provided model
# Precision: 0.78  Recall: 0.76  F-score: 0.77
```


## 📕引用我们的工作

```
@article{陈亮:18,
author = {陈亮},
author = {高博飞},
author = {常宝宝},
author = {张亦驰},
title = {基于概念预测和关系预测的AMR解析与对齐方法},
publisher = {中文信息学报},
year = {2024},
journal = {中文信息学报},
volume = {38},
number = {7},
eid = {18},
pages = {18-30},
keywords = {语义解析|抽象语义表示|中文自然语言处理},
url = {http://jcip.cipsc.org.cn/CN/Y2024/V38/I7/18},
doi = {null},
}



@misc{Chen2022ATM,
  title={A Two-Stage Method for Chinese AMR Parsing},
  author={Liang Chen and Bofei Gao and Baobao Chang},
  year={2022},
  journal = {arXiv}
}
```
