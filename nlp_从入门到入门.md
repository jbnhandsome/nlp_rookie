## NLP笔记 

### 文本处理

![Q_XFO~A6__SLLD_E_3_N7AJ.png](https://i.loli.net/2021/10/13/kNwMunyoH9XJzmt.png)

1.读入文本

2.对文本进行tokenization，就是将一句话处理成一个一个单词

3.然后我们将上一步处理完的tokens进行encoding，把每一个单词，都用一个数字来表示。

4.因为每句话的长度不一样，所以我们需要规定一个长度，比如说20个词，如果超过就截取，不够就补0。



![5TM4GM3@~1_R_9_A1__55UJ.png](https://i.loli.net/2021/10/13/9q2MgOR8BI4XyYz.png)

5.然后我们把句子输入到Embedding层，变成20*8的向量，这里的8表示每个单词用一个8维的向量表示。Embedding层有一个参数10000 * 8， 第一个表示单词的个数，第二个表示词向量维度

6.然后把这20 * 8的向量展开，变成一个160的向量，进行逻辑回归，这里有161的参数向量，因为多一个1维的bias

### RNN

序列化数据，适用于输入输出的长度都不固定的数据

![_V2ATU_ZGPMD6W_~50_P__8.png](https://i.loli.net/2021/10/13/EMgAaIvKHJy1T53.png)

用h保存阅读过的信息，例如h2，包含了the cat sat的信息。

整个RNN只有一个参数A

![2H_1_W__1GW_GDOYQ70EXMR.png](https://i.loli.net/2021/10/13/a9VeN6Du3EflCSH.png)

1. 我们首先把h在t-1时刻的输出，和t时刻的输入x拼接在一起，乘上参数A
2. 然后我们把上一步得到的结果矩阵用tanh(双曲正切函数，如右下角的图)作用一下，就得到t时刻的输出h了。

我们为什么需要tanh激活函数呢？

![_Y6TM2S_3CJ@714SM_00WEB.png](https://i.loli.net/2021/10/13/svZcqnwQHphOoGf.png)

假设我们的x词向量都是0，并且有100个X，再把tanh去掉.

如果矩阵A的特征值略小于1，这样的话矩阵A的100次方激活接近0

如果矩阵A的特征值略大于1，这样矩阵A的100次方就会无穷大，参数爆炸。



矩阵A的大小shape(h) * [shape(h)+shape(x)]



#### 用RNN做电影情感评价

![X90FJF8@_W19@9K_BTC_Z12.png](https://i.loli.net/2021/10/13/R6umFkjJeg8ClI2.png)

1.我们搭建一个神经网络，最底层是word_embedding，把词映射成词向量，词向量的维度由你自己选择，最好用cross validation 来选择最优的维度，这里我们设置X的维度是32

2.然后搭建一层simple RNN layer, 输出的是状态h，h的维度也是由你自己确定，同上。我们依旧设置h的维度32，h=x这里只是巧合，通常不一样。h1 包含i love 的信息。 h_t包含了所有的信息。

3.我们只是用h_t，把之前的全部丢掉，h_t大小0-1，表示正负值。

![RPWXA__F7_H_UZ4M1XD_WCS.png](https://i.loli.net/2021/10/13/BZw1a3IudrMUN7x.png)

上图是参数个数表

500表示 每句话有500个词

32表示每个词用32维的词向量表示

2080 表示矩阵A的大小，同时还要加32个偏移量

![~S5Z5QMJ1EEFKSM6P2YB7PN.png](https://i.loli.net/2021/10/13/7PXCfoyFDH2QTrR.png)

这里我们使用全部状态，所以还要加一个flatten层，把状态矩阵变成一个向量，然后把向量作为输入，判断电影是正面的还是负面的。

### LSTM

原理和RNN差不多

![`3V_XGW@J__PFAR2_VWFMS7.png](https://i.loli.net/2021/10/16/GzAiB38ItMaR4Dj.png)



LSTM最重要的信息是上面的传输带,过去的信息记作C_t-1,直接送到下一个时刻不会发生太大的变化，LSTM就是依靠传输带避免梯度消失的问题

![PFTCII88E__BK8B_1~N_`D3.png](https://i.loli.net/2021/10/16/aBpjAVlWmxubqRU.png)

LSTM里面有许多gate

#### Forget Gate

![ZPTZ3_BC5`M_87__1_C__3C.png](https://i.loli.net/2021/10/16/7ObHEBDT9xtrapC.png)

输入向量a，经过一个sigmoid函数，把a中的每一个元素都压缩成0-1的数值，输入的a和f维度相同

![2TS7B_P_FL8__0UK40LB_J6.png](https://i.loli.net/2021/10/16/KTu8sQMNlHYvqtG.png)

下面我们把a和c做点乘得到下一时刻状态，如果c中向量对应的f元素是0，那这个状态就不能通过，如果对应的元素是1，那个状态全部通过



![_5QP5@_O_OHF___ZRZ9__H6.png](https://i.loli.net/2021/10/16/hsxc1XyD9kfOG6K.png)



f值的计算类似于RNN

同样是把x_t和h_t-1(上一时刻**状态**)合起来

![2Z__ZK~Q1K8G_Q93BXE~29I.png](https://i.loli.net/2021/10/16/tZr3fyYxuEz7jMs.png)

W_f 是需要通过反向传播来进行学习

**遗忘门的作用：**

思考一个具体的例子，假设一个语言模型试图基于前面所有的词预测下一个单词，在这种情况下，每个 cell 状态都应该包含了当前主语的性别（**保留信息**），这样接下来我们才能正确使用代词。但是当我们又开始描述一个新的主语时，就应该把旧主语的性别给忘了才对（**忘记信息**）



#### Input gate

输入们i_t的计算类似遗忘门

![7_8__6HCHGW_S3___YMHS_U.png](https://i.loli.net/2021/10/16/rpH8JRNMsEbBKma.png)

i_t的每一个元素都介于0-1之间

输入们也有自己的输入矩阵W_i



还需要算New value

也是把旧状态和x_t拼接在一起，区别是激活函数是tanh(-1 - +1)

![43Q_FI@_L~3@3_41_Z~_J53.png](https://i.loli.net/2021/10/16/6GmLeD5FlON3EYh.png)



这样我们就可以计算全部的输出了

![JCKQ@ZT_R_LLC_25IUW`4HR.png](https://i.loli.net/2021/10/16/l2iHr5AXp9tyaGY.png)

i_t 和 f_t用来控制输入和遗忘数量

#### output gate

思考一个具体的例子，假设一个语言模型试图基于前面所有的词预测下一个单词，在这种情况下，每个 cell 状态都应该包含了当前主语的性别（**保留信息**），这样接下来我们才能正确使用代词。但是当我们又开始描述一个新的主语时，就应该把旧主语的性别给忘了才对（**忘记信息**）

最后一步是计算LSTM的输出，也就是h_t，t时刻的状态

这里我们计算输出们o_t, 计算方法和之前类似，也是把x_t和h_t-1拼接在一起

现在我们计算h_t

我们用tanh激活c_t，再和o_t相乘

这里我们要copy两份这个输出，一个是LSTM的输出，另一个是传给下一个状态

![XW7@U6BMZR1BUQB_QSI_P_6.png](https://i.loli.net/2021/10/17/FE5L9xSMYcQundv.png)



我们来计算LSTM的参数

有四个参数矩阵W，矩阵的行数是h的维度，列数是h+x的维度

所以参数的数量是 4* shape(h) * [shape(h) + shape(x)]



### Making RNN more effective

#### stacked RNN

堆叠多层RNN

![T03~2QM0E1_8_A__BALUL81.png](https://i.loli.net/2021/10/17/8DkZLu4HiEdBvoj.png)

最低层的RNN输入是词向量，后面的RNN输入时上一层的状态向量H

我们一共用了三层RNN，最上面的状态h时stacked RNN 的输出

keras实现：

```python
model = sequential( )
model.add(mbedding (vocabulary,embedding_dim，input_length=word_num) model.add(LSTM(state_dim, return_sequences=True,dropout=0.2))
model.add(LSTM(state_dim,return_sequences=True,dropout=0.2))
model.add(LsTM(state_dim, return_sequences=False,dropout=0.2))#可以把前面所有的状态向量全部扔掉
mode1.add(Dense( 1, activation='sigmoid'))

```

![_2@__J5M5C_5XIJ_ET7F1_J.png](https://i.loli.net/2021/10/17/TjVUyeY9xqJHrMd.png)

这四个数字都是32只是巧合，最后一个RNN只输入最后一个状态向量

### Bidirectional RNN

对RNN来说，从前往后和从后往前没有太大的区别。

可以训练两条完全独立的RNN

![__FXVJTU9_NE67WWOWYOG@G.png](https://i.loli.net/2021/10/17/wADUQe7aIMPH54i.png)

对于A和A‘ 我们把他俩结合起来，作为下一个RNN的输入。

如果只有两层RNN，那我们可以把y都丢掉，只保留两条链最后的状态向量,然后把两条RNN的输出h做concatenation

下面是双向LSTM的实现

```python
from keras.models import sequential
from keras.layers import LSTM，Embedding,Dense,Bidirectional

vocabulary = 10000
embedding_dim = 32word_num = 500
state_dim = 32
model = sequential( )
model.add (Embeddina(vocabulary，embedding_dim,input_length=word_num))
model.add(Bidirectional(I,STM(state_dim,return_sequences=False,dropout=0.2)))
mode1.add (Dense( i, activation='sigmoid ' ) )
model.summary ( )

```

下图是结构参数图，状态向量的维度是32，所以Bidirection的参数是64

![OH6UP87W_28XT_@731V`N_5.png](https://i.loli.net/2021/10/17/Ve4p5COkdsrnSgw.png)



### Pretraining

因为参数实在是太多了，所以我们可以对embedding层进行预训练

#### 步骤

1. 我们需要先找到一个很大的数据集，或许是情感分析的任务，让学出来的词向量带一些情感。这个任务越和我们需要的任务相似，transfer的效果越好。这个神经网络的结构是什么都可以，甚至可以不是RNN。

2. 我们只留下Embedding层，其他层都丢掉
3. 然后再在上面搭建我们自己的RNN网络。我们要把Embedding层固定住，训练其他层，不训练这个层。

### Text General

#### RNN 文本预测 

![`_Z9RONP_KO5OO1FQAFLLCV.png](https://i.loli.net/2021/10/17/hHzvGwbiKZglmeB.png)

把字符转成one-hot向量输入RNN,然后我们取最后一个的状态输入，然后再放入Softmax输出预测到的字符，我们可能预测到的字符是t

那下一次RNN的输入就变成 the cat sat on the mat

或许下一次预测值是**.** 我们这样不断输入，可能最后会生成一个完整的话

#### How to train 

![2D_0SOYSRO`_F0AX_ZP2S1X.png](https://i.loli.net/2021/10/17/bHasOgo6CpuV3ln.png)

我们首先取40长度的句子作为我们的输入，然后下一个字符作为我们想要预测的标签，步长设置为3

假如这篇文章有3000个字符，我们会得到1000个样本

![JNAAPP_3ZS~4Q_XL76_6_U5.png](https://i.loli.net/2021/10/17/nKmyFSANeh4qgfE.png)

其实这就是一个多分类的问题，每个类别对应一个特殊的字符

用什么样本，生成的文本就是什么风格

**实际操作**

1. **准备数据**

![PHVQEN5QQ9AH@2M_WL_IXYW.png](https://i.loli.net/2021/10/17/C8RKV51DF4cJO6q.png)

2. **字符转换成向量**

一个token 是一个字符，我们把token转换成数字，然后我们就可以把每个正整数用one-hot向量表示

![ZJML3R_LJC_X4E_WN_NJCXB.png](https://i.loli.net/2021/10/17/JFYSjifbtGnoCuX.png)

我们一共有57个字符，我们设置每句话的长度是60，所以每个字符可以用57*1的one-hot向量表示，步长设置为1

![_9R2AE_NK_5Z6392H_`P0ZC.png](https://i.loli.net/2021/10/17/Pj4THRCxwKOXJWg.png)

3. **我们建立RNN**

![0A8L2S6WYCM8DRR_Z@SI1@B.png](https://i.loli.net/2021/10/17/FChlQsm2wOeAN5t.png)

LSTM输入是60 * 57的矩阵 LSTM状态向量的维度设置为128

Dense起到的作用是一个分类器

4. **训练**

![L3``1J26VC2OYK_U@SR__HJ.png](https://i.loli.net/2021/10/17/KTb1pZIefMkSO9s.png)

X表示 第i个slice是60*57的向量矩阵

y表示第i个slice是哪一个字符

5. **预测**

![TP4KKQ39_VG_FX@WO_VG71K.png](https://i.loli.net/2021/10/17/qQ9nvtLyDHR8gjS.png)

我们如何选择下一个字符呢

Option 1: greedy selection.

```
 next_index = np.argmax(pred). lt is determipistic.
```

选择概率最大的字符

Empirically not good.

Option 2: sampling from the multinomial distribution.. 

Option 3: adjusting the multinomial distribution.

```
pred = pred ** ( 1 / temperature)
pred = pred / np.sum(pred)
Sample according to pred.
Between greedy and multinomial (controlled temperature).
```

temperature介于0-1

概率值做这种变换，大的概率值会变大，小的会变小，介于两者之间

如下图所示：

![__CL_9_CF43ATUYA0~_ZGSB.png](https://i.loli.net/2021/10/17/VEKXg1wR47Ly6ar.png)

temperature

 越接近确定性选择

### Seq2Seq

#### ![TY__NLPBFU6Z_4ZJF_9B_IF.png](https://i.loli.net/2021/10/17/Pt8b1uSYl4GwgLz.png) 

这里主要分成一个解码器一个编码器

编码器传给解码器的是Encoder的LSTM的最后一个状态

Decoder通过这个状态可以得知Encoder输入的句子是go away

Decoder**第一个输入**必须是起始符 我们用/t表示

我们用字母的one-hot向量来作为标签y，通过预测p和y我们计算CrossEntropy，然后做反向传播，不断更新参数，直到预测完成

#### Seq2Seq Model in Keras

![_DPP_7H_3AE5E3WRS_3OZLW.png](https://i.loli.net/2021/10/17/STAFvKdBczWsl8P.png)

这里简单记一下，主要看Pytorch实现的部分

对于Decoder 我们首先输入的是\t 还有Encoder传入的状态，我们输出概率分布，再抽样抽样，得到字符m的可能性最大，我们再把这里输出的状态和m作为新的输入。