---
title: Keras常用函数汇总
mathjax: true
tags:
  - Keras
categories:
  - 机器学习
abbrlink: 24767
date: 2020-10-28 18:24:38
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

**参数：**

- n：整数，重复次数

**输入尺寸：**

2D张量，尺寸为（num_samples, features）

**输出尺寸：**

3D张量，尺寸为（num_samples, n, features）

# 回调Callbacks

- **ReduceLROnPlateau**
- **EarlyStopping**
- **ModelCheckpoint**
- **完整代码示例**

## **ReduceLROnPlateau**

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

当标准评估已经停止时，降低学习速率。

当学习停止时，模型总是会受益于降低 2-10 倍的学习速率。 这个回调函数监测一个数据并且当这个数据在一定「有耐心」的训练轮之后还没有进步， 那么学习速率就会被降低。

**例子：**

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

**参数：**

- **monitor**: 被监测的数据。
- **factor**: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
- **patience**: 没有进步的训练轮数，在这之后训练速率会被降低。
- **verbose**: 整数。0：安静，1：更新信息。
- **mode**: {auto, min, max} 其中之一。如果是 `min` 模式，学习速率会被降低如果被监测的数据已经停止下降； 在 `max` 模式，学习塑料会被降低如果被监测的数据已经停止上升； 在 `auto` 模式，方向会被从被监测的数据中自动推断出来。
- **epsilon**: 对于测量新的最优化的阀值，只关注巨大的改变。
- **cooldown**: 在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
- **min_lr**: 学习速率的下边界。

## **EarlyStopping**

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
```

当被监测的数量不再提升，则停止训练。

**参数：**

- **monitor**: 被监测的数据。
- **min_delta**: 在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。
- **patience**: 没有进步的训练轮数，在这之后训练就会被停止。
- **verbose**: 详细信息模式。
- **mode**: {auto, min, max} 其中之一。 在 `min` 模式中， 当被监测的数据停止下降，训练就会停止；在 `max` 模式中，当被监测的数据停止上升，训练就会停止；在 `auto` 模式中，方向会自动从被监测的数据的名字中判断出来。

## **ModelCheckpoint**

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

在每个训练期之后保存模型。

`filepath` 可以包括命名格式选项，可以由 `epoch` 的值和 `logs` 的键（由 `on_epoch_end` 参数传递）来填充。

例如：如果 `filepath` 是 `weights.{epoch:02d}-{val_loss:.2f}.hdf5`， 那么模型被保存的的文件名就会有训练轮数和验证损失。

**参数：**

- **filepath**: 字符串，保存模型的路径。
- **monitor**: 被监测的数据。
- **verbose**: 详细信息模式，0 或者 1 。
- **save_best_only**: 如果 `save_best_only=True`， 被监测数据的最佳模型就不会被覆盖。
- **mode**: {auto, min, max} 的其中之一。 如果 `save_best_only=True`，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。 对于 `val_acc`，模式就会是 `max`，而对于 `val_loss`，模式就需要是 `min`，等等。 在 `auto` 模式中，方向会自动从被监测的数据的名字中判断出来。
- **save_weights_only**: 如果 True，那么只有模型的权重会被保存 (`model.save_weights(filepath)`)， 否则的话，整个模型会被保存 (`model.save(filepath)`)。
- **period**: 每个检查点之间的间隔（训练轮数）。

**【注】**

loss是训练集的损失值，val_loss是测试集的损失值

以下是loss与val_loss的变化反映出训练走向的规律总结：

train loss 不断下降，test loss不断下降，说明网络仍在学习;（最好的）

train loss 不断下降，test loss趋于不变，说明网络过拟合;（max pool或者正则化）

train loss 趋于不变，test loss不断下降，说明数据集100%有问题;（检查dataset）

train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;（减少学习率）

train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。（最不好的情况）

## **完整代码示例**

```python
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.datasets import mnist

# 加载mnist数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape图片并归一化
X_train = X_train.reshape(-1, 28, 28, 1) / 255
X_test = X_test.reshape(-1, 28, 28, 1) / 255
# 转换成one-hot格式
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# 构建线性模型
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
# 把上一层的输出展开为向量用于后续的全连接操作
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))
adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=1)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   period=1)
reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
model.fit(X_train,
          y_train,
          epochs=5,
          batch_size=64,
          validation_data=(X_test, y_test),
          callbacks=[early_stopping, model_checkpoint, reduce_lr_on_plateau])
```

<br>

<br>

<br>

<img src="../../../images/地波雷达与自动识别系统（AIS）目标点迹最优关联算法/HDU_LOGO.png" alt="HDU_LOGO" style="zoom:50%;" />