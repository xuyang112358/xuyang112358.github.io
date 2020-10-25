---
title: 深入理解GRU
mathjax: true
tags:
  - 自然语言处理
  - 深度学习
categories:
  - 循环神经网络
abbrlink: 35841
date: 2020-10-24 16:54:37
---

# 简介

GRU 是 gated recurrent units 的缩写，由 Cho在 2014 年提出。GRU 和 LSTM 最大的不同在于 GRU 将遗忘门和输入门合成了一个"更新门"，同时网络不再额外给出记忆状态，而是将输出结果作为记忆状态不断向后循环传递，网络的输人和输出都变得特别简单。