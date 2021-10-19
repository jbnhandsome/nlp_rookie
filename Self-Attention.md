## Self-Attention

为了解决序列到序列模型记忆长序列能力不足的问题，一个非常直观的想法是，当我们要生成一个目标语言是，不光考虑前一个时刻的状态和已经生成的单词，还要考虑当前生成的单词和源语言句子中哪些单词更相关，更关注源语言的词，这种做法就叫做注意力机制。

a1表示输入向量

![`1W_LSF31E0K_UE~28_E_S0.png](https://i.loli.net/2021/10/01/5HKUpIQxLk7BWVR.png)

α表示每一个向量和a1的关联程度

那我们如何计算α呢？

![_8_EWU9XL_`F_C2~3K_YV8X.png](https://i.loli.net/2021/10/01/i4sh7ZmCBeWXdMc.png)

分别乘上Wq和Wk再做不同的计算，有非常多的方法来计算α

![_U9`_M0TTB__T@Q88~KA_B4.png](https://i.loli.net/2021/10/01/jC5MXAneytRFmgl.png)

如何计算attention score : α

ai*Wq 表示这是查询ai和其他向量的相关性qi (自己和自己也要计算相关性！)

aj*Wk表示这是aj提供的键值来计算相关性 kj

αi,j 表示 ai 和 aj的相关性 qi * kj

![1J2K_GK@~O_3S6VQCI_Y_6O.png](https://i.loli.net/2021/10/01/dY1FvCnwZ675eJu.png)

softmax 可以被替代为别的函数

![V1P5__T_OG6FU25JD_I_9N7.png](https://i.loli.net/2021/10/01/qxj95o7ifkTUYSC.png)

这里我们再把每一个向量乘上新的参数矩阵Wv，这样之后再和计算出来的α'相乘，这样，哪个向量的相关性最强，他俩相乘得到的值最大，最后相加的和也最接近。

![37I_2F35GMJ__77_NY~L_6Y.png](https://i.loli.net/2021/10/01/MfGjNOAcbP5Rl8U.png)

同理，我们也可以得到其他的b1,b2,b3,b4,

这些是被同时计算出来的，如下图所示：

![H18Z_X_7_Y@XV0_6JL1T_XH.png](https://i.loli.net/2021/10/01/Ht53RcYwQOI9Asn.png)

把a1,a2,a3,a4份别乘上不同的矩阵就得到了q,k,v

下一步是每一个q和其他k做dot product

![__XI_0__QN0UHUZ1MO_3GMN.png](https://i.loli.net/2021/10/01/Q8KmesGNVdcnLtg.png)

把q所形成的矩阵和k所形成的矩阵相乘

A = Kt * Q

![H_A2~31ZM_B2XFW_4XP2E5H.png](https://i.loli.net/2021/10/01/Hg4iI8CwlZjLE7M.png)

A'表示通过softmax的A

![__0HPWX2P2X@5T@I3MZI3AS.png](https://i.loli.net/2021/10/02/QfrdYatPeLjVR3q.png)

下一步再用A'计算O

其实这就是一连串的矩阵乘法，唯一需要学的参数就是Wq,Wk,Wv了，这三个矩阵里的参数是未知的

![53JWJ_YV~WWZ__@VPNZR_`X.png](https://i.loli.net/2021/10/02/mjzGpRwnSBr96aF.png)

### Multi-head Self-attention

![JQ4_DD85R1_ENKIO47ATHHA.png](https://i.loli.net/2021/10/02/2mYjtk9zwXoGAdP.png)

这里面q^i,1 表示第i个输入向量的第一个表示空间？

这里的不同的表示空间是ai乘上不同的Wq,Wk,Wv得到的

然后qi,1只针对向量空间相同的进行计算

![C~_QS_SWY5_~L0KF___N_4R.png](https://i.loli.net/2021/10/02/rqkxIwFb8hTNtJW.png)

下一步就是把算出来的两个向量空间接起来

![XJE__O__Y`JC_Z1T6TW6KZQ.png](https://i.loli.net/2021/10/02/GrJ7ZObV8tcapMl.png)

图片源自 [Multi-Head Attention - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/266448080) ， 表示整体的过程

#### Position Encoding

完全没有位置信息，天涯若比邻，所有的位置都一样

如果位置的资讯比较重要怎么办？

每一个位置有一个不同的位置向量ei

把e加到a上

![_`_KG_3OAV1NSD_KJVWUM9Q.png](https://i.loli.net/2021/10/02/WckTjGoAt8hPYSO.png)

这里论文告诉了我们如何去求这个位置向量.

### Self-attention VS RNN

 ![](https://i.bmp.ovh/imgs/2021/10/a44ba0926e6cd854.png) 

RNN 它今天 input 一排是 vector,output 另外一排 vector 的时候,它没有办法一次处理,没有办法平行处理所有的 output

但 Self-attention 有一个优势,是它可以平行处理所有的输出,你今天 input 一排 vector,再 output 这四个 vector 的时候,**这四个 vector 是平行产生的,并不需要等谁先运算完才把其他运算出来**,output 的这个 vector,里面的 output 这个 vector sequence 里面,每一个 vector 都是同时产生出来的

所以在运算速度上,Self-attention 会比 RNN 更有效率