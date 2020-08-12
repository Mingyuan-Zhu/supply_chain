# Deep Reinforcement Learning for VRP

```
车辆路径规划问题（Vehicle Routing Problem, VRP）是物流领域最经典的优化问题之一
- 指一定数量的客户各自有不同数量的货物需求，配送中心向客户提供并输送货物，其行车路线在满足客户需求的前提下，需达到诸如路程最短、成本最小、耗时最少等目标。
车辆路径规划问题是典型的NP-hard问题， 同样难以找到多项式的算法， 由于NP-Hard放宽了限定条件，它将有可能比所有的NPC问题的时间复杂度更高从而更难以解决。
VRP属于网络优化问题，到VRP的本质就是序列决策问题，深度学习技术在VRP上最近也有很多应用

```

![](https://github.com/wouterkool/attention-learn-to-route/raw/master/images/cvrp_0.png)

常见问题的变形

- ***\*CVRP：\****Capacitated VRP, 限制配送车辆的承载体积、重量等。
- ***\*VRPTW：\****VRP with Time Windows, 客户对货物的送达时间有时间窗要求。
- ***\*VRPPD：\****VRP with Pickup and Delivery, 车辆在配送过程中可以一边揽收一边配送，在外卖O2O场景中比较常见。
- ***\*MDVRP：\**** Multi-Depot VRP, 配送网络中有多个仓库，同样的货物可以在多个仓库取货。
- ***\*OVRP：\****Open VRP, 车辆完成配送任务之后不需要返回仓库。
- ***\*VRPB：\**** VRP with backhauls, 车辆完成配送任务之后回程取货。



下图所示的为各类问题的关系



![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WynZ0MPNicyoSlB2PKm4AblaiaWG7kwa7EXRQqzohibQBmcgKs5bIDJcuj68A9BUatpW6Roic8OqibCQHQ/640?wx_fmt=png)



```
常见的经典算法：

- 精确解算法：
  - 最基本的方法为列生成、分支定价算法，Branch-and-Cut以及Branch-Cut-and-Price

- 启发式算法
  - 通过一系列启发式的规则来构造和改变解，如模拟退火、禁忌搜索，Wright
```



```
现实生活中的应用实例：

- 菜鸟仓配 自适应大规模邻域搜索（Adaptive Large Neighborhood Search, ALNS） 核心core model
他人的解析为：   https://blog.csdn.net/b0Q8cpra539haFS7/article/details/89166730

```

### Attention

Attention机制的出现，使得NLP的发展取得了重大突破， 例如transformer 和bert model。类比于生物领域脑科学来说，倾向于人因工程的全局扫描后，关注于model特定的部分- 更专注于数据重要的部分。

![figure1 ](https://upload-images.jianshu.io/upload_images/13590053-1f06d6f5548bcf51.png?imageMogr2/auto-orient/strip|imageView2/2/w/746/format/webp)

****

****

***NLP*** 应用：

“Attention is all you nedd”, 第一次提出了注意力转移机制， 带来了相应的NLP领域的革新。例如基于注意力转移机制的transformer 以及bert 模型，获得了远超过传统RNN以及LSTM的效果， 它引入一个上下文向量来表示解码上下文，使用具有带确定性贪心基准值(greedy rollout baseline)的深度强化学习算法对模型进行训练

同时， pytroch-based 的易于实现，使得模型省去的大量的步骤在于指定n_head方面， 特别是与tensorflow1相比较而言。 这里给我我DT尝试过的transformer模型，来进行文本分析的实现。[https://github.com/Mingyuan-Zhu/supply_chain/blob/master/%E2%80%9C%E2%80%9C%E2%80%9Cpytorch_transorformer_test_glove_embdding300_Carl_ipynb%E2%80%9D%E2%80%9D%E7%9A%84%E5%89%AF%E6%9C%AC%20(1).ipynb](https://github.com/Mingyuan-Zhu/supply_chain/blob/master/"""pytorch_transorformer_test_glove_embdding300_Carl_ipynb""的副本 (1).ipynb)



### **Attention, learn to solve routing problems**

源码：

*https://github.com/wouterkool/attention-learn-to-route*

其他人求解CVRP的源码：

*https://github.com/echofist/AM-VRP*

从论文原文和实验结果均可以看出，这种完全端到端求解的深度强化学习方法相比LKH3启发式搜索方法最大的优势在于端到端神经网络的求解速度快(尤其在使用greedy策略时)；而相比同类型的完全端到端深度强化学习方法，本文使用的基于transformer的多头Attention模型具有更好地传递VRP中节点与节点之间信息的作用，它相比非多头注意力的embedding结合LSTM的RL模型具有提升求解质量的效果。



结果而言， DNN 在求解acc以及speed方面远超过LKH3的启发式搜索。 并且基于Multi-head teenetion的transformer模型来说，结果是好于传统的信息传递效果，例如对其进行vector pre-trend embedding 的长短期记忆以及CNN,RNN等。这与在NLP领域的结果大致一致，纵使基于“No free lunch”， 大部分情况下都会取得更好的效果。预计在transformer embedding处理 后会取得更好的效果，使用与训练的embedding并进行固定，会节约大量的训练时间。

实验代码基于pytorch框架~



~![paper code](https://github.com/wouterkool/attention-learn-to-route/raw/master/images/tsp.gif)

如图片所示， **VRP的目标是总成本最小的一组路径优化，每条路径中车辆从指定的仓库出发并最终回到仓库**，路径上的总需求不能超过车辆的承载能力。带容量限制的车辆路径规划问题(CVRP)则限定为，单辆特定容量限制的车辆负责，当车辆容量不足以满足任何客户点的需求时，必须返回仓库将货物装满。该问题的解决方案可以看作一组路径序列，每个路径序列的起点和终点都在车站，中间经过的都是客户节点。

**求解VRP的算法可分为精确算法和启发式算法**。精确算法提供了最优的保证解，但由于计算复杂度高，无法处理大规模算例，而启发式算法往往速度快，但由于没有精确理论保证往往只能得到次优解。考虑到最优性和计算代价之间的权衡，启发式算法可以在大规模算例可接受的运行时间内找到次优解。

原版transformer结构如下,Encoder含Multi-Head Attention layer，全连接层（Feed Forward）



![](https://raw.githubusercontent.com/ldy8665/Material/master/image/Blog/Attention_transformer_architecture.png)

这篇文章介绍的完全端到端的NN，自动地学习到隐含的启发式信息，尽可能地达到与启发式算法相近的效果，model结构类比于transformer然,修改了其结构，作为encoder和decoder。提出了一个基于注意力层的模型，该模型具有优于Pointer Network的优势，并且展示了如何进行训练.

- 解决的问题:旅行商问题TSP（Travelling Salesman Problem）

- Encoder： GraphAttentionEncoder（类名）

  在解码器的每一步解码中，为得到综合当前已有信息的Context node embedding，首先将从编码器得到的图嵌入信息与每一步解码需要增加的信息嵌入,

  对于TSP的求解，增加的信息为起点和当前点的embedding

  

  ![](https://raw.githubusercontent.com/ldy8665/Material/master/image/Blog/Attention_code_encoder.png)

  `generate_vrp_data(dataset_size, vrp_size)`，意思是多少个实例s，每个s是多少个结点node

  对一每一个问题实例s来说，s内包括n个结点（nodes，第i个结点的特征xi,对于TSP，是坐标，然后xi之间是全连通的。这一个实例s就可以当做一个图形（Graph）作为输入。

  ![](https://raw.githubusercontent.com/ldy8665/Material/master/image/Blog/Attention_input.png)

  ![](https://raw.githubusercontent.com/ldy8665/Material/master/image/Blog/Attention_generate_data3.png)

  第一个元素是是结点坐标，第二个是每个结点的demand，第三个是车的capacity，这就是我们的输入了

  

- Decoder： AttentionModel（类名）

  

![](https://raw.githubusercontent.com/ldy8665/Material/master/image/Blog/Attention_decoder.png)



