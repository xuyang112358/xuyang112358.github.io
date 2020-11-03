---
title: Keras常用函数汇总
date: 2020-10-28 18:24:38
mathjax: true
tags:
  - Keras
categories:
  - 机器学习
---

# tf.keras.Input函数

`tf.keras.Input`函数用于构建网络的第一层——输入层，向模型中输入数据，该层会告诉网络我们的输入的尺寸是什么，并指定数据的形状、数据类型等信息，会返回一个Model对象，通过该对象可以调用model.compile和model.fit函数，非常方便。

首先给出 `tf.keras.Input` 的函数定义：

```python
tf.keras.Input(
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=False,
    tensor=None,
    **kwargs
)
```

<!--more-->

其中各个参数的含义为（前四个是常用的）：

- `shape`：输入的形状，一个形状元组(由整数组成)，其中并不指定batch size，只是定义输入的数据的形状。比如`shape=(32, )`和`shape=32`是等价的，表示输入都为32维的向量。
- `batch_size`: 声明输入的batch_size大小，一般会在预测时候用，训练时不需要声明，会在fit时声明，即dataset类型数据声明了batch_size。
- `name`：可选参数，字符串形式表示当前层的名字。如果没有这个参数的话，会自动生成。
- `dtype`：数据类型，在大多数时候，我们需要的数据类型为tf.float32，因为在精度满足的情况下，float32运算更快。

- `sparse`：一个布尔值，指示创建的占位符是否是稀疏的。
- `tensor`：将现有张量wrap到Input层中，如果设置了的话，Input层将不会创建占位符张量(可以理解为张量是已有的，所以不需要创建新的占位符)
- `**kwargs`：当前并不支持的参数

以下为一个实例：

```python
# this is a logistic regression in Keras
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

# keras.layers.LSTM函数

```python
(units,  # 输出维度，指的是每一个lstm单元的hidden layer的神经元数量
activation='tanh',  # 要使用的激活函数
recurrent_activation='sigmoid',  # 用于循环时间步的激活函数
use_bias=True,  # 布尔值，该层是否使用偏置向量
kernel_initializer='glorot_uniform',  # 权值矩阵的初始化器，用于输入的线性转换
recurrent_initializer='orthogonal',  # 权值矩阵的初始化器，用于循环层状态的线性转换
bias_initializer='zeros',  # 偏置向量的初始化器
unit_forget_bias=True,  # 布尔值。如果为 True，初始化时，将忘记门的偏置加1。将其设置为 True 同时还会强制 bias_initializer="zeros"
kernel_regularizer=None,  # 运用到 kernel 权值矩阵的正则化函数
recurrent_regularizer=None,  # 运用到 recurrent_kernel 权值矩阵的正则化函数
bias_regularizer=None,  # 运用到偏置向量的正则化函数
activity_regularizer=None,  # 运用到层输出（它的激活值）的正则化函数 
kernel_constraint=None,  # 运用到 kernel 权值矩阵的约束函数
recurrent_constraint=None,  # 运用到 recurrent_kernel 权值矩阵的约束函数
bias_constraint=None,  # 运用到偏置向量的约束函数
dropout=0.,  # 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
recurrent_dropout=0.,  # 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于循环层状态的线性转换
implementation=2,  # 实现模式，1 或 2。 模式 1 将把它的操作结构化为更多的小的点积和加法操作， 而模式 2 将把它们分批到更少，更大的操作中。 这些模式在不同的硬件和不同的应用中具有不同的性能配置文件
return_sequences=False,  # 布尔值。是返回输出序列中的最后一个输出，还是全部序列
return_state=False,  # True:输出状态，以供下一个lstm单元使用。false:不输出state
go_backwards=False,  # 布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列
stateful=False,  # 布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品的最后状态 将用作下一批次中索引 i 样品的初始状态
time_major=False,
unroll=False,  # 布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，但它往往会占用更多的内存。 展开只适用于短序列
**kwargs)
```

# RepeatVector函数

```python
keras.layers.RepeatVector(n)
```

将输入重复n次。

**例：**

```
model = Sequential()
model.add(Dense(32, input_dim=32))
# 现在：model.output_shape == (None, 32)
# 注意：`None` 是批表示的维度

model.add(RepeatVector(3))
# 现在：model.output_shape == (None, 3, 32)
```

**参数**

- n：整数，重复次数

**输入尺寸**

2D张量，尺寸为（num_samples, features）

**输出尺寸**

3D张量，尺寸为（num_samples, n, features）