## Transformer

和Bert很有关系



## Transformer

Transformer就是一个,Sequence-to-sequence的model,他的缩写,我们会写做Seq2seq,那Sequence-to-sequence的model,又是什麼呢

#### Encoder

**Encoder** 做的事情,就是**输入一个 Vector Sequence**,**输出另外一个 Vector Sequence**

输入一个句子，输出一个任意长度的句子

例如语音识别，机器翻译

![UGRNO9K55Y@ZAWWS9AR8_Q3.png](https://i.loli.net/2021/10/02/daR4GLlFtCZOk67.png)

每一个Block展开是右边的部分

右边的自注意力机制也不是单纯的注意力机制，还需要加入一个残差网络

![WY_`P0__2SZS8E_99TV79HA.png](https://i.loli.net/2021/10/02/H5IsMqwxJ2idLKn.png)

layer normalization 计算这个向量不同的维度的均值和标准差

![@R@B_TI__B2XF_GO1CY_EF9.png](https://i.loli.net/2021/10/02/UIsCrEZOpmu8e9G.png)

当然在全连接网络的时候我们也需要加上残差网络和layer normalization 

![_D4WTYNF1L_YPW`~V5G`~_I.png](https://i.loli.net/2021/10/02/Cw4J5hqMrDTLcsp.png)

![W`6J4I4VIQGC_BE9H4ALX2N.png](https://i.loli.net/2021/10/02/oyXZbOx8zs1DuAK.png)

这里的Add&Norm就是残差网络和layer 归一化

bert就是transformer的encoder

#### Decoder

产生输出

Decoder 做的事情,就是**把 Encoder 的输出先读进去**,至於怎麼读进去,这个我们等一下再讲 我们先,你先假设 Somehow 就是有某种方法,把 Encoder 的输出读到 Decoder 裡面,这步我们等一下再处理

![7G_IB9Y0_K@NH28_HJK~TBI.png](https://i.loli.net/2021/10/03/XGOpqQfVzxg8Jyk.png)

在这个机器学习裡面,假设你要处理 NLP 的问题,**每一个 Token,你都可以把它用一个 One-Hot 的 Vector 来表示**,One-Hot Vector 就其中一维是 1,其他都是 0,所以 **BEGIN 也是用 One-Hot Vector 来表示**,其中一维是 1,其他是 0

接下来Decoder 会**吐出一个向量,这个 Vector 的长度很长,跟你的 Vocabulary 的 Size 是一样的**

你就先想好说,你的 Decoder **输出的单位**是什麼,假设我们今天做的是中文的语音辨识,我们 Decoder 输出的是中文,你这边的 Vocabulary 的 Size ,可能就是中文的方块字的数目

**不同的字典,给你的数字可能是不一样的**,常用的中文的方块字,大概两 三千个,一般人,可能认得的四 五千个,在更多都是罕见字 冷僻的字,所以你就看看说,你要让你的 Decoder,输出哪些可能的中文的方块字,你就把它列在这边

举例来说,你觉得这个 Decoder,能够输出常见的 3000 个方块字就好了,就把它列在这个地方,不同的语言,它输出的单位 不见不会不一样,这个取决於你对个语言的理解

比如说英文,你可以选择输出**字母的 A 到 Z**,输出英文的字母,但你可能会觉得字母这个单位太小了,有人可能会选择输出**英文的词汇**,英文的词汇是用空白作為间隔的,但如果都用词汇当作输出,又太多了

所以你会发现,刚才在助教的投影片裡面,助教说他是用 Subword 当作英文的单位,就有一些方法,可以把英文的字首字根切出来,拿字首字根当作单位,如果中文的话,我觉得就比较单纯,通常今天你可能就用中文的这个方块字,来当作单位

每一个中文的字,都会对应到一个数值,因為在**產生这个向量之前,你通常会先跑一个 Softmax**,就跟做分类一样,所以这一个向量裡面的分数,它是一个 Distribution,也就是,**它这个向量裡面的值,它全部加起来,总和 会是 1**

分数最高的中文字 就是最终的输出

![__ZX2OQSF9F6RJMY__GDL36.png](https://i.loli.net/2021/10/03/7SFVls2DMyZmRWw.png)

Decoder 会把自己的上一个输出，当作当前时刻的输入一步一步输出

但是如果上一步输出的是错误怎么解决呢，在最后会给出答案