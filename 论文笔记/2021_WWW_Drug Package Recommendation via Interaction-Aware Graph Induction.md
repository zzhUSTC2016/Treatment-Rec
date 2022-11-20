# Drug Package Recommendation via Interaction-Aware Graph Induction

Authors: Zhi Zheng, Chao Wang, **Tong Xu\***, Dazhong Shen,  Penggang Qin,  Baoxing Huai, **Tongzhu Liu**, **Enhong Chen**

Keywords: Drug Recommendation, Package Recommendation, Graph Neural Network  

- 关键词：药物推荐，包推荐，图神经网络

WWW’21  中国科大，华为，中国科大一附院（安徽省立医院）

刘同柱，原科大一附院党委书记，2020.12起任安徽省卫健委党组书记

论文链接：https://dl.acm.org/doi/pdf/10.1145/3442381.3449962

------

[TOC]

## 0. 总结

- 基于NCF和GNN，运用协同过滤技术来进行药包推荐
- 使用了科大一附院数据集APH
- 其中药包是指现有患者使用过的治疗方案，无法推荐出新的药物组合
- 药包中只包含药物名称，无法对药物剂量和使用频次等信息给出推荐方案

## 1.研究目标

利用历史诊疗信息和额外的药物互斥信息，对输入的诊断，给出对应的药物推荐。

## 2.问题背景

药物之间的相互关系是非常重要的，这种相关关系又与病人的身体情况有关，这是不被传统的推荐系统所考虑的。

## 3. 问题定义

### 3.1 数据集与预处理

#### 3.1.1 安徽省立医院的医疗记录数据 (electronic medical records, EMRs)

- Demographics 基本信息
  - 药物与年龄、体重等身体状况密切相关
  - 转化为字典格式
- Laboratory results 检验结果
  - 转化为字典格式
- Admission Notes 入院记录
  - 主诉、体查情况等
  - 去除标点和无效字符，并填充或者截断为固定长度的语句。（中文数据集？）
- Drugs  住院时使用的药物记录

<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220414214826315-783608010.png" alt="image-20220414214826012" style="zoom: 50%;" />

#### 3.1.2 药物相互作用数据

[DrugBank](https://go.drugbank.com/releases/latest)  和  [YaoZhi](https://db.yaozh.com/interaction)(药智数据)

以下是药智数据的例子：

![image-20220414221728589](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220414221728375-1301989803.png)

<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220414221810108-745717550.png" alt="image-20220414221810287" style="zoom: 80%;" />

- 将药物相互作用分为三类
  - No Interaction  无
  - Synergism  协作、促进
  - Antagonism  对抗、互斥
- ![image-20220414221345972](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220414221345782-443117701.png)



<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220414221307873-1696888425.png" alt="image-20220414221308030" style="zoom:67%;" />

## 3.2 问题定义

根据患者信息和候选药物集合，以及药物之间的关系矩阵，给出每个患者与每个药物之间的个性化评分。

## 4. 方法

![image-20220415225344153](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415225343396-1019745644.png)

- Pre-training
  - NCF and BPR Loss
  - 得到患者和药物的embedding

- Package Graph Construction
  - Graph of the drug package of each patient  每个患者的药包图
  - 引入药物之间的相互关系
  - 为学习药包的表示做准备
- Package Recommendation
  - 首先，结合患者信息，学习边的表示
    $$
    c_{v u}=\operatorname{Tanh}\left(a^{\top} M L P\left(\left[\hat{\mathbf{d}}_{u} \| \hat{\mathbf{d}}_{v}\right\rceil\right)\right),
    $$
  
  - 1层GNN，更新节点（药物）的表示
    $$
    \begin{gathered}
    \mathbf{m}_{v u}^{(l)}=W_{1}^{(l-1)} \mathbf{h}_{v}^{(l-1)} \\
    \mathbf{M}_{u}^{(l)}=\sum_{v \in \mathcal{N}(u)} G R U\left(\hat{\mathbf{e}}_{v u} \mathbf{m}_{v u}^{(l)}, \mathbf{h}_{u}^{(l-1)}\right) \\
    \mathbf{h}_{u}^{(l)}=M L P\left(W_{0}^{(l-1)} \mathbf{h}_{u}^{(l-1)}+\mathbf{M}_{u}^{(l)}\right)
    \end{gathered}
    $$
  
  - 将每个节点的表示聚合为药包的表示
    $$
    \mathrm{g}=\sum_{v \in V} \sigma\left(M L P\left(\left[\mathbf{d}_{v} \| \mathbf{h}_{v}\right]\right)\right) \odot\left(M L P\left(\left[\mathbf{d}_{v} \| \mathbf{h}_{v}\right]\right)\right)
    $$
  
  - 在这个阶段，item由药物（drug）变化为药包（drug package）

<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415225418835-812818593.png" alt="image-20220415225419677" style="zoom: 33%;" />

<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415225434132-224989198.png" alt="image-20220415225434979" style="zoom: 33%;" />

## 5. 实验

### 5.1 Baselines

- NCF：推荐出K个药物作为药包，K为药包平均大小
- NN（Nearest Neighbor）：基于NCF学出embedding，再选择跟当前test患者最接近的患者的药包
- Package2vec：对Item2vec做改进，学习药包的表示。
- LDA：基于LDA学习药包表示。
- BR：一种包推荐方法。SIGIR'17，Bundle Rec
- DAM：IJCAI'19，Attention network for bundle rec

- GNN：本方法的简化版本，不使用边表示，只进行节点信息聚合

数据集划分 8:1:1

### 5.2 Performance

<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415225508939-704501446.png" alt="image-20220415225509785" style="zoom:50%;" />

- 本文提出的方法效果最好。

- NCF性能最差，说明只用协同过滤方法不合适，会更多推荐流行药物，无法整体考虑药物组合等信息。

- GNN性能也不错，说明构建药包图，并学习药包表示是很有效的



