## Bert

BERT 全称为 **Bidirectional Encoder Representation from Transformer**，是 Google 以无监督的方式利用大量**无标注**文本「炼成」的语言模型，其架构为 Transformer 中的 Encoder（BERT=Encoder of Transformer）

![W`8X_S02HBS7DB1QC@ZCUGE.png](https://i.loli.net/2021/10/17/XeQjMGYN9IzuUSs.png)

Bert在训练过程中的任务主要分为两个，如下图所示

![1MYZ4_CP_2JHUHYB3CJGBQN.png](https://i.loli.net/2021/10/17/rsRWjqHMLwQSGlB.png)

1. #### **Masked Language Model**

   在 BERT 中，Masked LM（Masked Language Model）构建了语言模型，简单来说，就是**随机遮盖或替换**一句话里面的任意字或词，然后让模型通过上下文预测那一个被遮盖或替换的部分，之后**做 Loss 的时候也只计算被遮盖部分的 Loss**，这其实是一个很容易理解的任务，实际操作如下：

   1. 随机把一句话中 15% 的 token（字或词）替换成以下内容：
      1. 这些 token 有 80% 的几率被替换成 `[MASK]`，例如 my dog is hairy→my dog is [MASK]
      2. 有 10% 的几率被替换成任意一个其它的 token，例如 my dog is hairy→my dog is apple
      3. 有 10% 的几率原封不动，例如 my dog is hairy→my dog is hairy
   2. 之后让模型**预测和还原**被遮盖掉或替换掉的部分，计算损失的时候，只计算在第 1 步里被**随机遮盖或替换**的部分，其余部分不做损失，其余部分无论输出什么东西，都无所谓

   这样做的好处是，BERT 并不知道 [MASK] 替换的是哪一个词，而且任何一个词都有可能是被替换掉的，比如它看到的 apple 可能是被替换的词。这样强迫模型在编码当前时刻词的时候不能太依赖当前的词，而要考虑它的上下文，甚至根据上下文进行 "纠错"。比如上面的例子中，模型在编码 apple 时，根据上下文 my dog is，应该把 apple 编码成 hairy 的语义而不是 apple 的语义

例如 The [mask] sat on the mat

到最后我希望um经过一个softmax 得到一个p，希望p中cat的概率最大

![A_MBCVJ8GHDOLMR_JSKCAQS.png](https://i.loli.net/2021/10/17/QKMeHBajL3hks6v.png)

这里我们用e表示cat的one-hot，用p表示masked位置的概率

所以loss = CrossEntropy(e,p)

用梯度下降去优化，我们只关心被遮盖住的向量的损失，其他的向量我们并不关心。

2. #### **Next Sentence Prediction**

给一个句子 “ calculus is a branch of math ”

预测 "it was developed by newton and leibniz" 是否是他的下一句话

我们怎么做这个事情呢？

 我们首先拿到属于上下文的一对句子，也就是两个句子，之后我们要在这两个句子中加一些特殊的 token：`[CLS]上一句话[SEP]下一句话[SEP]`。也就是在句子开头加一个 `[CLS]`，在两句话之间和句末加 `[SEP]`，  可以看到，上图中的两句话明显是连续的。如果现在有这么一句话 `[CLS]我的狗很可爱[SEP]企鹅不擅长飞行[SEP]`，可见这两句话就不是连续的。在实际训练中，我们会让这两种情况出现的数量为 **1:1** 

![V13@N____S0GS~L0_`_P_Y7.png](https://i.loli.net/2021/10/17/oApuFSQgTjEm3JY.png)

Token Embeddings 是pytorch的nn.Embedding，是正常的词向量

Position Embedding 和Transformer不一样，是学出来的，不再是定死的。

Segment Embeddings 是用来告诉bert哪些是这一句话，那些是下一句话，帮助bert判断上下句的起止位置。

EX:

```
[CLS]我的狗很可爱[SEP]企鹅不擅长飞行[SEP]
 0   0 0 0 0 0 0 0  1 1 1 1 1 1 1 1
```



![RB_4_IDYFRG3EYXR8BU_XKY.png](https://i.loli.net/2021/10/17/ketzD93mbWcyv4Y.png)

开头加个cls 中间加个sep，然后把他俩通过上面的两层去做一个二分类

![8WAC21ZH_72RV_M2_K9W6_9.png](https://i.loli.net/2021/10/17/j49tsQbfiVDp6hz.png)

我们用这个二分类判断他俩1是否是下一句话。



所以总的来说，BERT它学会了如何填空。BERT的神奇之处在于，在你训练了一个填空的模型之后，它还可以**用于其他任务**。这些任务**不一定与填空有关**，也可能是完全不同的任务，但BERT仍然可以用于这些任务，这些任务是BERT实际使用的任务，它们被称为Downstream Tasks(下游任务)，以后我们将谈论一些Downstream Tasks 的例子。

所谓的 "Downstream Tasks  "是指，你真正关心的任务。但是，当我们想让BERT学习做这些任务时，我们仍然**需要一些标记的信息**。

总之，BERT只是学习填空，但是，以后可以用来做各种你感兴趣的Downstream Tasks 。它就像胚胎中的干细胞,它有各种无限的潜力，虽然它还没有使用它的力量,它只能填空,但以后它有能力解决各种任务。我们只需要给它一点数据来激发它，然后它就能做到。

BERT分化成各种任务的功能细胞，被称为Fine-tune(微调)。所以，我们经常听到有人说，他对BERT进行了微调，也就是说他手上有一个BERT，他对这个BERT进行了微调，使它能够完成某种任务，与微调相反，在微调之前产生这个BERT的过程称为预训练。

#### GLUE

通常我们训练好了Bert，我们会在多个任务上测试它，看看它在每个任务上的准确性，然后我们取其平均值，得到一个总分。这种不同任务的集合，，我们可以称之为任务集。任务集中最著名的基准被称为GLUE，它是General Language Understanding Evaluation的缩写。 

![GY5852SY~_2_F_J_JS1_~N0.png](https://i.loli.net/2021/10/18/CDB9qEHTeyVZ7iL.png)



下面我们会用四个案例来说明怎么使用和训练Bert

#### Case 1: Sentiment analysis

第一个案例是这样的，我们假设我们的Downstream Tasks 是输入一个序列，然后输出一个class，这是一个分类问题。

比如说Sentiment analysis情感分析，就是给机器一个句子，让它判断这个句子是正面的还是负面的。

![I4_Y_E_4_3Y`JE__`3___TK.png](https://i.loli.net/2021/10/18/VxSDXng8flaL5Ae.png)

 这里李宏毅老师有一点没讲到，就是为什么要用第一个位置，即 `[CLS]` 位置的 output。这里我看了网上的一些博客，结合自己的理解解释一下。因为 BERT 内部是 Transformer，而 Transformer 内部又是 Self-Attention，所以 `[CLS]` 的 output 里面肯定含有整句话的完整信息，这是毋庸置疑的。但是 Self-Attention 向量中，自己和自己的值其实是占大头的，现在假设使用 *[Math Processing Error]w1* 的 output 做分类，那么这个 output 中实际上会更加看重 *[Math Processing Error]w1*，而 *[Math Processing Error]w1* 又是一个有实际意义的字或词，这样难免会影响到最终的结果。但是 `[CLS]` 是没有任何实际意义的，只是一个占位符而已，所以就算 `[CLS]` 的 output 中自己的值占大头也无所谓。当然你也可以将所有词的 output 进行 concat，作为最终的 output 。

这里的话我们的Bert模型的参数不再是随机初始化了

在训练的时候，Linear transform和BERT模型都是利用Gradient descent来更新参数的。

- Linear transform的参数是**随机初始化**的
- 而BERT的参数是由**学会填空的BERT初始化**的。

#### Case 2: POS tagging

这里我们输入一个序列，然后输出另一个序列，输入输出长度相同，例如Pos tagging。

POS tagging的意思是词性标记。你给机器一个句子，它必须告诉你这个句子中每个词的词性，即使这个词是相同的，也可能有不同的词性。

你只需向BERT输入一个句子。之后，对于这个句子中的每一个标记，它是一个中文单词，有一个代表这个单词的相应向量。然后，这些向量会依次通过Linear transform和Softmax层。最后，网络会预测给定单词所属的类别，例如，它的词性。

这里的BERT部分，即网络的Encoder部分，其参数不是随机初始化的。在**预训练过程中，它已经找到了不错的参数**。 

![YMCAH_5R`K_IXUYHE___0HK.png](https://i.loli.net/2021/10/18/Iov6rOLHwSPVqQe.png)

#### Case 3 Natural Language Inference

 第三个案例以两个句子为输入，输出一个类别，什么样的任务采取这样的输入和输出？ 最常见的是Natural Language Inference ，它的缩写是NLI 

这里我们要输入一个前提和一个假设，我们需要判断这个前提和这个假设是否是矛盾的

![5_T@K_2~IEP__RK5E_3X_N2.png](https://i.loli.net/2021/10/18/VTUZqgimtQuFoSD.png)

机器要做的是判断，是否有可能**从前提中推断出假设**。这个前提与这个假设相矛盾吗？或者说它们不是相矛盾的句子？

在这个例子中，我们的前提是，一个人骑着马，然后他跳过一架破飞机，这听起来很奇怪。但这个句子实际上是这样的。这是一个基准语料库中的例子。

这里的**假设**是，这个人在一个餐馆。所以**推论**说这是一个**矛盾**。

所以机器要做的是，把两个句子作为输入，并输出这两个句子之间的关系。这种任务很常见。它可以用在哪里呢？例如，舆情分析。给定一篇文章，下面有一个评论，这个消息是同意这篇文章，还是反对这篇文章？该模型想要预测的是每条评论的位置。事实上，有很多应用程序接收两个句子，并输出一个类别。就像是上面的上下文判断判断

![OV_Q_8RJ2O8_X1FKHXNOFPH.png](https://i.loli.net/2021/10/18/bQSg6PM2xTKXwso.png)

#### Case 4 Extraction-based Question Answering

这里是 一个问题回答系统。也就是说，在机器读完一篇文章后，你问它一个问题，它将给你一个答案。 

![_9X04AQ6SEC_T5P~BFUAFK6.png](https://i.loli.net/2021/10/18/yt1bzrqITw5LQOE.png)

在这个任务中，一个输入序列包含**一篇文章**和**一个问题**，文章和问题都是一个**序列**。对于中文来说，每个d代表一个汉字，每个q代表一个汉字。你把d和q放入QA模型中，我们希望它输出**两个正整数s和e**。根据这两个正整数，我们可以直接从文章中**截取一段**，它就是答案。这个片段就是正确的答案。