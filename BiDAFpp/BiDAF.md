假设Context有T个单词，Query有J个单词

1. 词向量用预训练的glove(维度用d1表示)，字符向量用 `1D-CNN` 计算（最后得到的向量维度等于你用的卷积核的个数，这里用d2表示，字符向量的作用是可以弥补当训练BiDAF时遇到不在gloVe字典中的单词）
2. 拼接词向量和字符向量，得到的向量维度用d表示（d=d1+d2），现在我们有了两个矩阵，维度分别为 $d*T$ 和 $d*J$ ，分别用于表示Context和Query
3. 通过`highway network` （`highway network` 和`resnet` 很像，它的作者甚至认为何凯明是抄袭了他的思想），用公式表示就是 $y=t\bigodot g(Wx+b)+(1-t)\bigodot x​$ ，其中 $\bigodot​$ 表示`element-wise multiply` ，g表示激活函数，t的值本身也是由另一个线性层加一个激活函数算出来，取值范围是0到1。可以认为 `highway network` 的作用是调整词向量和字符向量的相对比重。这步过后，两个矩阵的维度不变
4. 单层双向的LSTM。这步过后，两个矩阵的维度分别为 $2d*T$ 和 $ 2d*J$ ，分别命名为H和U
5. 接下来的 `Attention Flow Layer` 是个模型的重点，提出了一个相似度矩阵S，维度是 $T*J$ ，表示每个上下文单词和每个问句单词的相似度，S是这么来的：对于上下文a和问句b， $S_{ab}=w^T[a;b;a\bigodot b]$ ，`;`表示上下拼接，w是可训练的向量，维度是 $6d*1$ 。得到S后，可以进行下面的两个过程 `Context-to-Query Attention` 和 `Query-to-Context attention `
6. `Context-to-Query Attention` ： 取出S中的一行 $1*J​$ ，做softmax，得到的结果即视为权重，与U中的每一列做加权求和，得到一个 $2d*1​$ 的向量。遍历S中的每一行重复上述动作，得到矩阵 $\check{U}​$ ，维度为 $2d*T​$
7.  `Query-to-Context attention` ：和上面的做法并不一样，先取出S中每一行的最大值，得到一个列向量 $T*1$ ，做softmax，用矩阵H和这个列向量做矩阵乘法，得到一个 $ 2d*1 $ 的向量，然后直接把这个向量拷贝T次，得到矩阵 $\check{H}$ ，维度为 $2d*T$
8. 这步要把 $H$ ，$\check{H}$ ， $\check{U}$ 组合在一起得到一个大矩阵 $G$ 。是这样： $G_t=\beta (H_t,\check{U}_t, \check{H}_t)$ ，下标t表示列号，其中 $\beta(a,b,c)=[a;b;a\bigodot b;a\bigodot c]$ ，$G$ 的维度是 $8d*T$
9. 双层双向的LSTM。第一层双向的LSTM过后，得到矩阵M1，维度为 $2d*T$ ，第二层双向的LSTM过后，得到矩阵M2，,维度也为 $2d*T$
10. 可以预测答案的开始位置和结束位置了，$p1=softmax(w^T_{p1}[G;M1])$ ，$p2=softmax(w^T_{p2}[G;M2])$， w维度都是$1*10d$ ，p纬度都是$1*T$ ，预测的时候取T个值中最大的那个；训练的时候损失函数为$L(\theta)=-\frac{1}{N}\sum\limits_i^N{[log(p_{y1})+log(p_{y2})]}$ ，其中$y1$ 和$y2$ 表示正确答案



Reference:

1. https://towardsdatascience.com/modeling-and-output-layers-in-bidaf-an-illustrated-guide-with-minions-f2e101a10d83

