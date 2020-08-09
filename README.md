# Deep Reinforcement Learning for VRP
```
车辆路径规划问题（Vehicle Routing Problem, VRP）是物流领域最经典的优化问题之一
- 指一定数量的客户各自有不同数量的货物需求，配送中心向客户提供并输送货物，其行车路线在满足客户需求的前提下，需达到诸如路程最短、成本最小、耗时最少等目标。
车辆路径规划问题是典型的NP-hard问题， 同样难以找到多项式的算法， 由于NP-Hard放宽了限定条件，它将有可能比所有的NPC问题的时间复杂度更高从而更难以解决。
VRP属于网络优化问题，到VRP的本质就是序列决策问题，深度学习技术在VRP上最近也有很多应用

```

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

Attention机制的出现，使得NLP的发展取得了重大突破， 例如transformer 和bert model。

***NLP*** 应用：

“Attention is all you nedd”, 第一次提出了注意力转移机制