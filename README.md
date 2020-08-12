# Deep Reinforcement Learning for VRP(ICLR2019)

```
车辆路径规划问题（Vehicle Routing Problem, VRP）是物流领域最经典的优化问题之一
- 指一定数量的客户各自有不同数量的货物需求，配送中心向客户提供并输送货物，其行车路线在满足客户需求的前提下，需达到诸如路程最短、成本最小、耗时最少等目标。
车辆路径规划问题是典型的NP-hard问题， 同样难以找到多项式的算法， 由于NP-Hard放宽了限定条件，它将有可能比所有的NPC问题的时间复杂度更高从而更难以解决。
VRP属于网络优化问题，到VRP的本质就是序列决策问题，深度学习技术在VRP上最近也有很多应用

```

![asd](https://github.com/wouterkool/attention-learn-to-route/raw/master/images/cvrp_0.png)

常见问题的变形

- ***CVRP***Capacitated VRP, 限制配送车辆的承载体积、重量等。
- ***VRPTW***VRP with Time Windows, 客户对货物的送达时间有时间窗要求。
- ***VRPPD***VRP with Pickup and Delivery, 车辆在配送过程中可以一边揽收一边配送，在外卖O2O场景中比较常见。
- ***MDVRP*** Multi-Depot VRP, 配送网络中有多个仓库，同样的货物可以在多个仓库取货。
- ***OVRP***Open VRP, 车辆完成配送任务之后不需要返回仓库。
- ***VRPB*** VRP with backhauls, 车辆完成配送任务之后回程取货。



下图所示的为各类问题的关系



![s](https://github.com/Mingyuan-Zhu/supply_chain/blob/master/p)



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

同时， pytroch-based 的易于实现，使得模型省去的大量的步骤在于指定n_head方面， 特别是与tensorflow1相比较而言。 这里给我我DT尝试过的transformer模型，来进行文本分析的实现。

```

```



[Code_transformer_in Job description ](https://github.com/Mingyuan-Zhu/supply_chain/blob/master/%E2%80%9C%E2%80%9C%E2%80%9Cpytorch_transorformer_test_glove_embdding300_Carl_ipynb%E2%80%9D%E2%80%9D%E7%9A%84%E5%89%AF%E6%9C%AC%20(1).ipynb)

```
class JDclassification(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, batch_size, device):
        super(JDclassification,self).__init__()
        ## define the components used to construct a nn network

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.embedding.weight = nn.Parameter(torch.from_numpy(weights_matrix).float())

        
        self.batch_size=batch_size
        self.num_encoder=2
        self.num_head=16
        self.max_length=500
        self.dropout=0.5
        self.hidden=100
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc_predict = nn.Linear(300, num_classes)

#        self.init_weights()



    def forward(self, text, mask, seq_length_idx):
        ## construct the nn; define the data processing steps

        batch_size=text.size()[0]
        x =self.embedding(text)*mask[:,:,None] #[batch,seq,embed]
        x=self.transformer_encoder(x)*mask[:,:,None]
        x=x.permute(0,2,1)
        x=F.relu(F.avg_pool1d(x,x.size(2)).squeeze(2))
        x=self.fc_predict(x)
        return x

```

### **Attention, learn to solve routing problems**(ICRL2019)



"Greddy"算法作为baseline来指导Pointer Network学习求解组合优化问题。用贪婪算法作为baseline比传统增强学习的值函数能够更加有效。本文所提方法能够对节点数目在100以下的TSP（旅行商问题）（测试集为20，50，80，100）问题有着较好的效果，同时本文还在相同超参数的情况下验证了该算法在两类VRP（车辆路径规划问题）问题上的求解效果

DNN用于传统的组合优化问题求解的文章NIPS2018就有一篇用增强学习求解VRP问题的文章。组合优化问题多都属于NP-hard问题，传统的基于纯运筹学的求解思路很难高效准确的得到这类问题的最优解，DNN是一个很好的研究思路。

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

  

  每个node的特征（在TSP类是2维坐标），通过`self.init_embed = nn.Linear(node_dim, embed_dim) 线性投影到高维空间，然后通过`MultiHeadAttentionLayer这个类，得到最后的node embeddings：h和graph embedding：h`. h 的均值（h_means就是） graph embedding。

  最后将最后结点的embeddings的平均作为图形的嵌入（graph embedding），并且将结点的embdedding和graph embedding作为decoder的输入。本文的MHA用了8层（M=8），然后最后全连接层用了一层512的隐藏层，Relu激活函数。

  ```
          # node为所有点的的初始embedding
          node = self.embedding(sss)  # 客户点embedding坐标加容量需求
          node[:, 0, :] = self.embedding_p(s[:, 0, :])  # 仓库点只embedding坐标
          x111 = node  # [batch x seq_len x embedding_dim]
          
          在前向传播部分随机初始化embedding服从Normal distribution, 并不固定embedding，而是进行更新（与NLP的不进行预训练，而是用attention进行更新）类似
  ```



使用了N个transformer模型中的注意力层，每个注意力层包含两个子层：多头注意力层(Multi-Head Attention layer, ***MHA***)和全连接前馈层(fully connected Feed-Forward layer, ***FF***)，每个子层都使用了残差连接(Skip connection)和批归一化(Batch Normalization, ***BN***)。

```

        ##################################################################################
        # encoder部分
        #####################################################
        # 第一层MHA
        query1 = self.wq1(node)
        query1 = t.unsqueeze(query1, dim=2)
        query1 = query1.expand(batch, city_size, city_size, embedding_size)
        key1 = self.wk1(node)
        key1 = t.unsqueeze(key1, dim=1)
        key1 = key1.expand(batch, city_size, city_size, embedding_size)
        value1 = self.wv1(node)
        value1 = t.unsqueeze(value1, dim=1)
        value1 = value1.expand(batch, city_size, city_size, embedding_size)
        x = query1 * key1
        x = x.view(batch, city_size, city_size, M, -1)
        x = t.sum(x, dim=4)  # u=q^T x k
        x = x / (dk ** 0.5)
        x = F.softmax(x, dim=2)
        x = t.unsqueeze(x, dim=4)
        x = x.expand(batch, city_size, city_size, M, 16)
        x = x.contiguous()
        x = x.view(batch, city_size, city_size, -1)
        x = x * value1
        x = t.sum(x, dim=2)  # MHA :(batch, city_size, embedding_size)
        x = self.w1(x)  # 得到一层MHA的结果

        x = x + x111
        #####################
        # 第一层第一个BN
        x = x.permute(0, 2, 1)
        x = self.BN11(x)
        x = x.permute(0, 2, 1)
        # x = t.tanh(x)
        #####################
        # 第一层FF
        x1 = self.fw1(x)
        x1 = F.relu(x1)
        x1 = self.fb1(x1)

        x = x + x1
```



<svg xmlns="http://www.w3.org/2000/svg" width="17.554ex" height="6.354ex" role="img" focusable="false" viewBox="0 -1562.5 7759 2808.5" style="vertical-align: -2.819ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><g data-mml-node="math"><g data-mml-node="msubsup"><g data-mml-node="mi"><path data-c="68" d="M137 683Q138 683 209 688T282 694Q294 694 294 685Q294 674 258 534Q220 386 220 383Q220 381 227 388Q288 442 357 442Q411 442 444 415T478 336Q478 285 440 178T402 50Q403 36 407 31T422 26Q450 26 474 56T513 138Q516 149 519 151T535 153Q555 153 555 145Q555 144 551 130Q535 71 500 33Q466 -10 419 -10H414Q367 -10 346 17T325 74Q325 90 361 192T398 345Q398 404 354 404H349Q266 404 205 306L198 293L164 158Q132 28 127 16Q114 -11 83 -11Q69 -11 59 -2T48 16Q48 30 121 320L195 616Q195 629 188 632T149 637H128Q122 643 122 645T124 664Q129 683 137 683Z"></path></g><g data-mml-node="TeXAtom" transform="translate(576, 530.4) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mo"><path data-c="28" d="M94 250Q94 319 104 381T127 488T164 576T202 643T244 695T277 729T302 750H315H319Q333 750 333 741Q333 738 316 720T275 667T226 581T184 443T167 250T184 58T225 -81T274 -167T316 -220T333 -241Q333 -250 318 -250H315H302L274 -226Q180 -141 137 -14T94 250Z"></path></g><g data-mml-node="mi" transform="translate(389, 0)"><path data-c="4E" d="M234 637Q231 637 226 637Q201 637 196 638T191 649Q191 676 202 682Q204 683 299 683Q376 683 387 683T401 677Q612 181 616 168L670 381Q723 592 723 606Q723 633 659 637Q635 637 635 648Q635 650 637 660Q641 676 643 679T653 683Q656 683 684 682T767 680Q817 680 843 681T873 682Q888 682 888 672Q888 650 880 642Q878 637 858 637Q787 633 769 597L620 7Q618 0 599 0Q585 0 582 2Q579 5 453 305L326 604L261 344Q196 88 196 79Q201 46 268 46H278Q284 41 284 38T282 19Q278 6 272 0H259Q228 2 151 2Q123 2 100 2T63 2T46 1Q31 1 31 10Q31 14 34 26T39 40Q41 46 62 46Q130 49 150 85Q154 91 221 362L289 634Q287 635 234 637Z"></path></g><g data-mml-node="mo" transform="translate(1277, 0)"><path data-c="29" d="M60 749L64 750Q69 750 74 750H86L114 726Q208 641 251 514T294 250Q294 182 284 119T261 12T224 -76T186 -143T145 -194T113 -227T90 -246Q87 -249 86 -250H74Q66 -250 63 -250T58 -247T55 -238Q56 -237 66 -225Q221 -64 221 250T66 725Q56 737 55 738Q55 746 60 749Z"></path></g></g><g data-mml-node="TeXAtom" transform="translate(576, -356.7) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mo"><path data-c="28" d="M94 250Q94 319 104 381T127 488T164 576T202 643T244 695T277 729T302 750H315H319Q333 750 333 741Q333 738 316 720T275 667T226 581T184 443T167 250T184 58T225 -81T274 -167T316 -220T333 -241Q333 -250 318 -250H315H302L274 -226Q180 -141 137 -14T94 250Z"></path></g><g data-mml-node="mi" transform="translate(389, 0)"><path data-c="67" d="M311 43Q296 30 267 15T206 0Q143 0 105 45T66 160Q66 265 143 353T314 442Q361 442 401 394L404 398Q406 401 409 404T418 412T431 419T447 422Q461 422 470 413T480 394Q480 379 423 152T363 -80Q345 -134 286 -169T151 -205Q10 -205 10 -137Q10 -111 28 -91T74 -71Q89 -71 102 -80T116 -111Q116 -121 114 -130T107 -144T99 -154T92 -162L90 -164H91Q101 -167 151 -167Q189 -167 211 -155Q234 -144 254 -122T282 -75Q288 -56 298 -13Q311 35 311 43ZM384 328L380 339Q377 350 375 354T369 368T359 382T346 393T328 402T306 405Q262 405 221 352Q191 313 171 233T151 117Q151 38 213 38Q269 38 323 108L331 118L384 328Z"></path></g><g data-mml-node="mo" transform="translate(866, 0)"><path data-c="29" d="M60 749L64 750Q69 750 74 750H86L114 726Q208 641 251 514T294 250Q294 182 284 119T261 12T224 -76T186 -143T145 -194T113 -227T90 -246Q87 -249 86 -250H74Q66 -250 63 -250T58 -247T55 -238Q56 -237 66 -225Q221 -64 221 250T66 725Q56 737 55 738Q55 746 60 749Z"></path></g></g></g><g data-mml-node="mo" transform="translate(2081.8, 0)"><path data-c="3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path></g><g data-mml-node="mfrac" transform="translate(3137.6, 0)"><g data-mml-node="mn" transform="translate(270, 676)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"></path></g><g data-mml-node="mi" transform="translate(220, -686)"><path data-c="6E" d="M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z"></path></g><rect width="800" height="60" x="120" y="220"></rect></g><g data-mml-node="munderover" transform="translate(4344.3, 0)"><g data-mml-node="mo"><path data-c="2211" d="M60 948Q63 950 665 950H1267L1325 815Q1384 677 1388 669H1348L1341 683Q1320 724 1285 761Q1235 809 1174 838T1033 881T882 898T699 902H574H543H251L259 891Q722 258 724 252Q725 250 724 246Q721 243 460 -56L196 -356Q196 -357 407 -357Q459 -357 548 -357T676 -358Q812 -358 896 -353T1063 -332T1204 -283T1307 -196Q1328 -170 1348 -124H1388Q1388 -125 1381 -145T1356 -210T1325 -294L1267 -449L666 -450Q64 -450 61 -448Q55 -446 55 -439Q55 -437 57 -433L590 177Q590 178 557 222T452 366T322 544L56 909L55 924Q55 945 60 948Z"></path></g><g data-mml-node="TeXAtom" transform="translate(148.2, -1087.9) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><path data-c="69" d="M184 600Q184 624 203 642T247 661Q265 661 277 649T290 619Q290 596 270 577T226 557Q211 557 198 567T184 600ZM21 287Q21 295 30 318T54 369T98 420T158 442Q197 442 223 419T250 357Q250 340 236 301T196 196T154 83Q149 61 149 51Q149 26 166 26Q175 26 185 29T208 43T235 78T260 137Q263 149 265 151T282 153Q302 153 302 143Q302 135 293 112T268 61T223 11T161 -11Q129 -11 102 10T74 74Q74 91 79 106T122 220Q160 321 166 341T173 380Q173 404 156 404H154Q124 404 99 371T61 287Q60 286 59 284T58 281T56 279T53 278T49 278T41 278H27Q21 284 21 287Z"></path></g><g data-mml-node="mo" transform="translate(345, 0)"><path data-c="3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path></g><g data-mml-node="mn" transform="translate(1123, 0)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"></path></g></g><g data-mml-node="TeXAtom" transform="translate(509.9, 1150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><path data-c="6E" d="M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z"></path></g></g></g><g data-mml-node="msubsup" transform="translate(5954.9, 0)"><g data-mml-node="mi"><path data-c="68" d="M137 683Q138 683 209 688T282 694Q294 694 294 685Q294 674 258 534Q220 386 220 383Q220 381 227 388Q288 442 357 442Q411 442 444 415T478 336Q478 285 440 178T402 50Q403 36 407 31T422 26Q450 26 474 56T513 138Q516 149 519 151T535 153Q555 153 555 145Q555 144 551 130Q535 71 500 33Q466 -10 419 -10H414Q367 -10 346 17T325 74Q325 90 361 192T398 345Q398 404 354 404H349Q266 404 205 306L198 293L164 158Q132 28 127 16Q114 -11 83 -11Q69 -11 59 -2T48 16Q48 30 121 320L195 616Q195 629 188 632T149 637H128Q122 643 122 645T124 664Q129 683 137 683Z"></path></g><g data-mml-node="TeXAtom" transform="translate(576, 530.4) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mo"><path data-c="28" d="M94 250Q94 319 104 381T127 488T164 576T202 643T244 695T277 729T302 750H315H319Q333 750 333 741Q333 738 316 720T275 667T226 581T184 443T167 250T184 58T225 -81T274 -167T316 -220T333 -241Q333 -250 318 -250H315H302L274 -226Q180 -141 137 -14T94 250Z"></path></g><g data-mml-node="mi" transform="translate(389, 0)"><path data-c="4E" d="M234 637Q231 637 226 637Q201 637 196 638T191 649Q191 676 202 682Q204 683 299 683Q376 683 387 683T401 677Q612 181 616 168L670 381Q723 592 723 606Q723 633 659 637Q635 637 635 648Q635 650 637 660Q641 676 643 679T653 683Q656 683 684 682T767 680Q817 680 843 681T873 682Q888 682 888 672Q888 650 880 642Q878 637 858 637Q787 633 769 597L620 7Q618 0 599 0Q585 0 582 2Q579 5 453 305L326 604L261 344Q196 88 196 79Q201 46 268 46H278Q284 41 284 38T282 19Q278 6 272 0H259Q228 2 151 2Q123 2 100 2T63 2T46 1Q31 1 31 10Q31 14 34 26T39 40Q41 46 62 46Q130 49 150 85Q154 91 221 362L289 634Q287 635 234 637Z"></path></g><g data-mml-node="mo" transform="translate(1277, 0)"><path data-c="29" d="M60 749L64 750Q69 750 74 750H86L114 726Q208 641 251 514T294 250Q294 182 284 119T261 12T224 -76T186 -143T145 -194T113 -227T90 -246Q87 -249 86 -250H74Q66 -250 63 -250T58 -247T55 -238Q56 -237 66 -225Q221 -64 221 250T66 725Q56 737 55 738Q55 746 60 749Z"></path></g></g><g data-mml-node="TeXAtom" transform="translate(576, -293.8) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><path data-c="69" d="M184 600Q184 624 203 642T247 661Q265 661 277 649T290 619Q290 596 270 577T226 557Q211 557 198 567T184 600ZM21 287Q21 295 30 318T54 369T98 420T158 442Q197 442 223 419T250 357Q250 340 236 301T196 196T154 83Q149 61 149 51Q149 26 166 26Q175 26 185 29T208 43T235 78T260 137Q263 149 265 151T282 153Q302 153 302 143Q302 135 293 112T268 61T223 11T161 -11Q129 -11 102 10T74 74Q74 91 79 106T122 220Q160 321 166 341T173 380Q173 404 156 404H154Q124 404 99 371T61 287Q60 286 59 284T58 281T56 279T53 278T49 278T41 278H27Q21 284 21 287Z"></path></g></g></g></g></g></svg>



- Decoder： AttentionModel（类名）

  Encoder的graph embedding和结点的embedding当做是Decoder的输入。并且将Graph embedding和input symbol（上一层的输入的隐藏层h和第一次输出的隐藏层h）拼接到一起

  然后根据MHA计算出新的context node embedding。然后每一个MHA的self_attention（也就是每一个head）

![](https://raw.githubusercontent.com/ldy8665/Material/master/image/Blog/Attention_decoder.png)



```
        ##################################################################################
        # decoder部分
        for i in range(city_size * 2):  # decoder输出序列的长度不超过city_size * 2
            flag = t.sum(dd, dim=1)  # dd:(batch, city_size)
            f1 = t.nonzero(flag > 0).view(-1)  # 取得需求不全为0的batch号
            f2 = t.nonzero(flag == 0).view(-1)  # 取得需求全为0的batch号

            if f1.size()[0] == 0:  # batch所有需求均为0
                pro[:, i:] = 1  # pro:(batch, city_size*2)
                seq[:, i:] = 0  # swq:(batch, city_size*2)
                temp = dis.view(-1, city_size, 1)[
                    index + mask_size]  # dis:任意两点间的距离 (batch, city_size, city_size, 1) temp:(batch, city_size,1)
                distance = distance + temp.view(-1)[mask_size]  # 加上当前点到仓库的距离
                break

            ind = index + mask_size
            tag[ind] = 0  # tag:(batch*city_size)
            start = x.view(-1, embedding_size)[ind]  # (batch, embedding_size)，每个batch中选出一个节点

            end = rest[:, :, 0]  # (batch, 1)
            end = end.float()  # 车上剩余容量

            graph = t.cat([avg, start, end], dim=1)  # 结合图embedding，当前点embedding，车剩余容量: (batch,embedding_size*2 + 1)_
            query = self.wq(graph)  # (batch, embedding_size)
            query = t.unsqueeze(query, dim=1)
            query = query.expand(batch, city_size, embedding_size)
            key = self.wk(x)
            value = self.wv(x)
            temp = query * key
            temp = temp.view(batch, city_size, M, -1)
            temp = t.sum(temp, dim=3)  # (batch, city_size, M)
            temp = temp / (dk ** 0.5)

            mask = tag.view(batch, -1, 1) < 0.5  # 访问过的点tag=0
            mask1 = dd.view(batch, city_size, 1) > rest.expand(batch, city_size, 1)  # 客户需求大于车剩余容量的点

            flag = t.nonzero(index).view(-1)  # 在batch中取得当前车不在仓库点的batch号
            mask = mask + mask1  # mask:(batch x city_size x 1)
            mask = mask > 0
            mask[f2, 0, 0] = 0  # 需求全为0则使车一直在仓库
            if flag.size()[0] > 0:  # 将有车不在仓库的batch的仓库点开放
                mask[flag, 0, 0] = 0

            mask = mask.expand(batch, city_size, M)
            temp.masked_fill_(mask, -float('inf'))
            temp = F.softmax(temp, dim=1)
            temp = t.unsqueeze(temp, dim=3)
            temp = temp.expand(batch, city_size, M, 16)
            temp = temp.contiguous()
            temp = temp.view(batch, city_size, -1)
            temp = temp * value
            temp = t.sum(temp, dim=1)
            temp = self.w(temp)  # hc,(batch,embedding_size)

            query = self.q(temp)
            key = self.k(x)  # (batch, city_size, embedding_size)
            query = t.unsqueeze(query, dim=1)  # (batch, 1 ,embedding_size)
            query = query.expand(batch, city_size, embedding_size)  # (batch, city_size, embedding_size)
            temp = query * key
            temp = t.sum(temp, dim=2)
            temp = temp / (dk ** 0.5)
            temp = t.tanh(temp) * C  # (batch, city_size)

            mask = mask[:, :, 0]
            temp.masked_fill_(mask, -float('inf'))
            p = F.softmax(temp, dim=1)  # 得到选取每个点时所有点可能被选择的概率

            indexx = t.LongTensor(batch).to(DEVICE)
            if train != 0:
                indexx[f1] = t.multinomial(p[f1], 1)[:, 0]  # 按sampling策略选点
            else:
                indexx[f1] = (t.max(p[f1], dim=1)[1])  # 按greedy策略选点

            indexx[f2] = 0
            p = p.view(-1)
            pro[:, i] = p[indexx + mask_size]
            pro[f2, i] = 1
            rest = rest - (dd.view(-1)[indexx + mask_size]).view(batch, 1, 1)  # 车的剩余容量
            dd = dd.view(-1)
            dd[indexx + mask_size] = 0
            dd = dd.view(batch, city_size)

            temp = dis.view(-1, city_size, 1)[index + mask_size]
            distance = distance + temp.view(-1)[indexx + mask_size]

            mask3 = indexx == 0
            mask3 = mask3.view(batch, 1, 1)
            rest.masked_fill_(mask3, l)  # 车回到仓库将容量设为初始值

            index = indexx
            seq[:, i] = index[:]

        if train == 0:
            seq = seq.detach()
            pro = pro.detach()
            distance = distance.detach()

        return seq, pro, distance  # 被选取的点序列,每个点被选取时的选取概率,这些序列的总路径长度


```







***PS:***  欠缺部分，1 源码的train function过分的难读   2 伪码理解较为困难

