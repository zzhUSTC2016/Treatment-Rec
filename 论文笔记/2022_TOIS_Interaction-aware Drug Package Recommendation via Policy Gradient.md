# Interaction-aware Drug Package Recommendation via Policy Gradient

Authors: ZHI ZHENG, CHAO WANG, **TONG XU\***, DAZHONG SHEN, PENGGANG QIN, XIANGYU ZHAO,  BAOXING HUAI,   XIAN WU,   **ENHONG CHEN**

Keywords: drug recommendation, package recommendation, graph neural network, reinforcement learning  

- 关键词：药物推荐，组合推荐，图神经网络，强化学习

TOIS'22 中国科大，香港城市大学，华为，腾讯

论文链接：https://dl.acm.org/doi/pdf/10.1145/3511020

[toc]

## 0. 总结

这篇文章使用了他们自己收集的数据集APH，以及MIMIC-3

用GNN建模药物之间的关系，用RNN生成预测结果，基于强化学习来训练

## 1. 研究目标

基于历史病例数据，生成多个药物组合进行推荐。

## 2. 研究背景

前序研究只能推荐现有的药物组合，

## 3. Preliminaries

### 3.1 数据集和预处理方法

#### 3.1.1 APH 安徽省立

**治疗记录：**

- Demographics 基本信息
  - 药物与年龄、体重等身体状况密切相关
  - 转化为字典格式
- Laboratory results 检验结果
  - 转化为字典格式
- Admission Notes 入院记录
  - 主诉、体查情况等
  - 去除标点和无效字符，并填充或者截断为固定长度的语句。（中文数据集？）
- Drugs  住院时使用的药物记录

**药物互斥信息：**

[DrugBank](https://go.drugbank.com/releases/latest)  和  [YaoZhi](https://db.yaozh.com/interaction)(药智数据)

#### 3.1.2 MIMIC-3

从中提取患者基本信息、检验结果和药物记录，即上述1、2、4项

将诊断代码和相关数据转化为入院记录

药物互斥信息同上

### 3.2 问题定义

WWW'21：对现有的药包进行排序

![image-20220415202200863](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415202200078-745865523.png)

TOIS'22：**生成**新的药包，选取最合适的

![image-20220415202212239](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415202211452-1358839937.png)

系统给出不只一种推荐结果，可以给医生更多选择和参考，尤其是包含一些不常用的药物。

推荐结果应该有如下特性：

- 准确性：推出的首选治疗方案应尽可能准确
- 综合性：C中的药物应包含正在使用的所有药物
- 多样性：C中的药物应尽可能多样以提供更多参考

## 4. 方法

![image-20220415203608061](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415203607372-693979963.png)

### 4.1 Message Passing on Drug Interaction Graph



用边分类任务来训练药物和边的embedding。
$$
\begin{gathered}
\mathbf{e}_{v u}^{(l)}=M L P^{(l)}\left(\left[\mathbf{h}_{u}^{(l)} \| \mathbf{h}_{v}^{(l)}\right]\right) \\
\mathbf{m}_{v u}^{(l)}=W_{1}^{(l-1)} \mathbf{e}_{v u}^{(l-1)} \\
\mathbf{M}_{u}^{(l)}=\sum_{v \in \mathcal{N}(u)} \mathbf{m}_{v u}^{(l)} \\
\mathbf{h}_{u}^{(l)}=M L P\left(W_{0}^{(l-1)} \mathbf{h}_{u}^{(l-1)}+\mathbf{M}_{u}^{(l)}\right)
\end{gathered}
$$

- 节点通过MLP表示生成边表示
- 进一步生成聚合信息
- 所有邻居的聚合信息
- 节点表示更新

$$
L_{g r a p h}=-\sum_{u, v \in G} \ln \left(\operatorname{softmax}\left(\hat{\mathbf{e}}_{v u}^{\top} \mathbf{Q}\right)_{\mathcal{R}_{u v}}\right)
$$

边分类任务作为监督信号（正/负/无相互作用）

### 4.2 Patient Encoder

结构化数据表示：
$$
\mathbf{m}_{w}=M L P(\mathcal{W})
$$
非结构化数据用char-GRU得到其表示$\mathbf{h}_{q}$

连接起来作为病人的表示：
$$
\mathbf{u}=\left[\mathbf{m}_{w} \| \mathbf{h}_{q}\right]
$$
还需要一个mask向量：
$$
\mathbf{m}=\sigma(M L P(\mathbf{u}))
$$

### 4.3 Drug Package Generation

#### 4.3.1 Drug Package Generation with Maximum Likelihood Estimation

药物按照出现频率降序排列。

为了捕捉历史推荐结果中的互斥信息，利用4.2中的MLP生成对应的边表示。没有构建图，是为了减少计算复杂度（O(t)）
$$
\mathbf{i}_{t}=M L P\left(\sum_{k=1}^{t-1} \mathbf{m} \odot M L P_{i n t e r}\left(\left[\mathbf{d}_{k} \| \mathbf{d}_{t}\right]\right)\right)
$$
GRU：
$$
\begin{aligned}
\mathbf{r}_{t} &=\sigma\left(\mathbf{W}_{d r}\left[\mathbf{d}_{t} \| \mathbf{i}_{t}\right]+\mathbf{W}_{h r} \mathbf{h}_{t-1}+\mathbf{b}_{r}\right) \\
\mathbf{z}_{t} &=\sigma\left(\mathbf{W}_{d z}\left[\mathbf{d}_{t} \| \mathbf{i}_{t}\right]+\mathbf{W}_{h z} \mathbf{h}_{t-1}+\mathbf{b}_{z}\right) \\
\tilde{\mathbf{h}}_{t} &=\tanh \left(\mathbf{W}_{d h}\left[\mathbf{d}_{t} \| \mathbf{i}_{t}\right]+\mathbf{W}_{h h}\left(\mathbf{r}_{t} \odot \mathbf{h}_{t-1}\right)+\mathbf{b}_{h}\right) \\
\mathbf{h}_{t} &=\left(1-\mathbf{z}_{t}\right) \odot \mathbf{h}_{t-1}+\mathbf{z}_{t} \odot \tilde{\mathbf{h}}_{t}
\end{aligned}
$$
为了不出现重复药物：
$$
d_{t+1} \sim \operatorname{softmax}\left(\mathbf{W}_{o} \mathbf{h}_{t}+\mathbf{M}_{t}\right),
$$
$$
\left(M_{t}\right)_{i}= \begin{cases}-\infty & \text { if the } i \text {-th drug has been predicted. } \\ 0 & \text { otherwise. }\end{cases}
$$

Loss：
$$
L_{M L E}=-\sum_{t=1}^{T} \log \left(\operatorname{softmax}\left(\mathbf{W}_{o} \mathbf{h}_{t-1}+\mathbf{M}_{t-1}\right)_{d_{t}^{*}}\right) .
$$

#### 4.3.2 Drug Package Generation with Policy Gradient

为了避免药物顺序带来的影响，采用强化学习

基于强化学习的训练方式：
$$
L(\theta)=-\mathbb{E}_{\mathbf{y} \sim p_{\theta}(\mathbf{u})}[r(\mathbf{y})] \approx-r(\tilde{\mathbf{y}}),
$$
基于F1 score的reward function：
$$
\begin{gathered}
\operatorname{Precision}(\mathrm{y}, \mathcal{P})=\frac{|\mathrm{y} \cap \mathcal{P}|}{|\mathrm{y}|}, \\
\operatorname{Recall}(\mathrm{y}, \mathcal{P})=\frac{|\mathrm{y} \cap \mathcal{P}|}{|\mathcal{P}|}, \\
r(\mathrm{y})=F_{1}(\mathrm{y}, \mathcal{P})=\frac{2 * \operatorname{Precision}(\mathrm{y}, \mathcal{P}) * \operatorname{Recall}(\mathrm{y}, \mathcal{P})}{\operatorname{Precision}(\mathrm{y}, \mathcal{P})+\operatorname{Recall}(\mathrm{y}, \mathcal{P})},
\end{gathered}
$$
Policy gradient优化参数。
$$
\begin{aligned}
L_{R L} &=(r(\hat{\mathbf{y}})-r(\tilde{\mathbf{y}})) \log p_{\theta}(\tilde{\mathbf{y}} \mid \mathbf{u}) \\
&=(r(\hat{\mathbf{y}})-r(\tilde{\mathbf{y}})) \sum_{t=1}^{T} \log p\left(d_{t} \mid d_{1}, \ldots, d_{t-1}, \mathbf{u}\right) \\
&=(r(\hat{\mathbf{y}})-r(\tilde{\mathbf{y}})) \sum_{t=1}^{T} \log \left(\operatorname{softmax}\left(\mathbf{W}_{o} \mathbf{h}_{t-1}+\mathbf{M}_{t-1}\right)_{d_{t}}\right) .
\end{aligned}
$$

### 4.4 Training and Testing Strategies

#### 4.4.1 Training Strategies

先用MLE Loss预训练：
$$
L_{\text {pretrain }}=L_{M L E}+\alpha * L_{g r a p h}+\lambda_{1} *\|\Theta\|_{2}^{2}
$$
再用强化学习Loss训练：
$$
L_{D P G}=L_{R L}+\beta * L_{g r a p h}+\lambda_{2} *\|\Theta\|_{2}^{2},
$$

## 5. 实验

### 5.1 数据统计

数据集总体情况统计：

- \# of records 病人数量
- \# of drugs 药物数量
- \# of words in disease document 结构化数据（基本信息+实验室检验数据）
- The average size of durg packages  平均每个药包大小
- \# of aligned drugs  出现在药物相互作用数据库中的药物数量？
- 最后两行占第一行的比例很大，说明药物相互作用普遍存在
- 最后一行比例接近一半，背后原因值得探究
  - 有些药物相互作用是温和的，可以接受
    - 药物相互作用进一步分级
  - 有些药物存在相互作用，但是没有一起使用
  - 确实存在互斥药物同时使用的情况

<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415215729329-1671927410.png" alt="image-20220415215730086" style="zoom: 80%;" />

APH中药物出现频率分布：

- 呈长尾分布，与商品推荐系统中的分布相似
  - 推荐结果是否有popularity bias？
  - 是否可以用debias方法加以研究？

<img src="https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415222122589-1213227709.png" alt="image-20220415222123419" style="zoom: 50%;" />

### 5.2 Baselines and Evaluation Metrics

#### 5.2.1 Baselines：

- traditional recommendation:
  - NCF：推荐出K个药物作为药包，K为药包平均大小
  - NN（Nearest Neighbor）：基于NCF学出embedding，再选择跟当前test患者最接近的患者的药包
- package recommendation:
  - Package2vec：对Item2vec做改进，学习药包的表示。
  - LDA：基于LDA学习药包表示。
  - BR：一种包推荐方法。SIGIR'17，Bundle Rec
  - DAM：IJCAI'19，Attention network for bundle rec
  - DPR：他们WWW’21提出的方法
- drug recommendation：
  - GRU-MLE:   本文方法变体，不采用强化学习
  - GRU-F：本文方法变体，由F1 score给出reward
  - GRU-DPR：本文方法变体，由DPR给出reward
  - CGAN：基于ICCV‘17的GAN技术，GRU为生成器，DPR为判别器
  - KG-MIML-Net ：ACML'18，multi-instance multi-label learning task，encoder-decoder model
  - GAMENet：AAAI’19，LEAP同组，GCN捕捉药物相关关系，Attention?
  - CompNet：CIKM‘19，Relational-GCN，RL

#### 5.2.2 Evaluation Metrics

Set Precision, Set Recall and Set F1-value (S-Precision, S-Recall, S-F1)  

![image-20220415224219080](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415224218372-1156464567.png)

ablation study:

推出的n个药包覆盖率有多少，越大越好
$$
Coverage $=\frac{\left|\left(\mathcal{P}_{1} \cup \mathcal{P}_{2} \cup \cdots \cup \mathcal{P}_{n}\right) \cap \mathcal{P}_{g}\right|}{\left|\mathcal{P}_{g}\right|},
$$
推出的n个药包多样性如何，越大越好
$$
Diversity =\frac{1}{n \times(n-1)} \sum_{\substack{1 \leq i, j \leq n \\ i \neq j}}\left(1-\frac{\left|\mathcal{P}_{i} \cap \mathcal{P}_{j}\right|}{\left|\mathcal{P}_{i} \cup \mathcal{P}_{j}\right|}\right).
$$
![image-20220415224240211](https://img2022.cnblogs.com/blog/2671585/202204/2671585-20220415224239368-178579618.png)
