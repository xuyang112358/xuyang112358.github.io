---
title: numpy常用函数汇总
abbrlink: 60994
date: 2020-11-06 14:30:15
mathjax: true
tags:
  - Python
categories:
  - 机器学习
---

## 函数np.append(arr, values, axis=None)

**作用：**

为原始array添加一些values

**参数：**

- arr:需要被添加values的数组
- values:添加到数组arr中的值（array_like，类数组）
- axis:可选参数，如果axis没有给出，那么arr，values都将先展平成一维数组。**注：如果axis被指定了，那么arr和values需要有相同的shape，否则报错：ValueError: arrays must have same number of dimensions**

补充对axis的理解

- axis的最大值为数组arr的维数-1，如arr维数等于1，axis最大值为0；arr维数等于2，axis最大值为1，以此类推。
- 当arr的维数为2(理解为单通道图)，axis=0表示沿着行方向添加values；axis=1表示沿着列方向添加values
- 当arr的维数为3(理解为多通道图)，axis=0，axis=1时同上；axis=2表示沿着深度方向添加values

**返回：**

添加了values的新数组

<!--more-->