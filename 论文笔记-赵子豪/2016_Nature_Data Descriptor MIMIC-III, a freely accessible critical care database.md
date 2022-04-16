# Data Descriptor: MIMIC-III, a freely accessible critical care database

Authors: Alistair E.W. Johnson,  Tom J. Pollard,  Lu Shen,  Li-wei H. Lehman,  Mengling Feng,  Mohammad Ghassemi,  Benjamin Moody,  Peter Szolovits,  Leo Anthony Celi & Roger G. Mark 

Nature-scientific data'16

麻省理工学院医学工程与科学研究所计算生理学实验室，波士顿白斯•以色列狄肯尼斯医学中心，新加坡信息通信研究所数据分析系，麻省理工学院计算机科学和人工智能实验室

论文链接：https://www.nature.com/articles/sdata201635

[toc]

## 0. 总结

本文公开了一个大规模医疗信息数据库MIMIC-III (Medical Information Mart for Intensive Care，重症监护医疗信息数据集) 

## 1. 研究背景

- 电子病例迅速普及，美国使用电子病历的医院比例从2008年的9.4%上升到2014年的75.5%。

- 关于医疗健康的科学研究由于缺乏公开数据集，可复现性很差。

## 2. 数据库介绍

- 数据来自以色列医院，美国顶级医疗机构，哈佛医学院主要教学医院之一。

- 数据包括生命体征、药物、实验室检测结果、护理记录、体液平衡、手术代码、诊断代码、成像报告、住院时间、存活数据等。
- 包含2001-2012年之间5万+成年病人（16岁+）进入ICU的医疗记录，以及2001-2008年间7870名新生儿的医疗记录
- 根据数据格式，患者数据可以划分为结构化数据（实验室测量、生命体征等）和非结构化数据（医嘱信息等）。

下表是一个病人的医疗记录和生命体征记录：

![image-20220415190643223](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415190642908-409376499.png)

下表是数据库中包含的26张表：

<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415192721155-109944072.png" alt="image-20220415192721808" style="zoom:150%;" />

26张表的详细解释，参考 [MIMIC-III数据集介绍](https://blog.csdn.net/qq_43787862/article/details/105028846)

![在这里插入图片描述](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415194854139-23459304.png)

- 这张表用主要记录患者的入院情况,用的比较多的可能有**患者的人口统计学信息**

![在这里插入图片描述](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415195054029-1374809644.png)

- CHARTEVENTS是最重要的一张表,记录的大部分是患者生命体征的数据,如心率,血压,体温等等
- *两张INPUTEVENTS应组合起来使用.提供比如说患者给药的速率(如葡萄糖输入的速率),给药途径,给药部位.基于这两个表可以做一些关于给药,药物干预方面的研究*
- NOTEEVENTS大部分是患者的医嘱,如患者的既往史和现病史等,再比如患者体温波动的情况等,都是通过文本形式给出的.

- *OUTPUTEVENTS主要记录了患者的出量信息,比如说患者的尿量等信息,可以作为患者生命情况的表示.*
- *PROCEDUREEVENTS记录诸如手术开始时间,结束时间,手术操作等信息*

![在这里插入图片描述](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415195311873-1037250027.png)

- **DIAGNOSES_ICD**表中记录了患者的**ICD-9诊断编码,**比如说想做一些疾病诊断或疾病预测的研究时会用到.一个患者可能会对应多个诊断,所以是一个序列格式的表.可能会认为第一个是患者的主病.
- DRGCODES表中记录了患者的诊断类别和诊断编码
- LABEVENTS表中是患者的化验项目,有比如像白细胞,红细胞这种指标值.LABEVENTS,CHARTEVENTS和OUTPUTSEVENTS表合起来基本上可以代表患者进入ICU后生理指标的大部分特征.
- **PRESCRIPTIONS**中是患者的用药记录,和前面的INPUTEVENTS综合起来可以作为用药干预的研究
- PROCEDURE_ICD表记录的是病人的手术记录.
  

![在这里插入图片描述](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415195539300-586967380.png)

