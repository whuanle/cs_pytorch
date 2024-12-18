# 入门神经网络和 Torch

本章节介绍神经网络基础知识。

1，神经网络基础知识、线性模型、全连接层，内容比较简单，不深入讨论，

2，了解 Pytorch 怎么搭建神经网络，基础流程，内容要非常简单，不深入讨论。

3，使用一个示例简单入门，下还有参数优化的影响、，不深入讨论，各种卷积神经网络上的优化。

4，启动社区上已经有的模型，

5，了解神经网络的基础骨架





 包括数据流水线、模型、损失函数和小批量随机梯度下降优化器



然后是如何使用神经网络 Pytorch 框架，在 C# 中的写法。

然后简单使用 《破解深度网络》中的线性回归和分类问题。



模型训练中的泛化（generalization）是指机器学习模型在未见过的数据上的表现能力。也就是说，一个模型不仅能在训练数据上表现良好，还能在新的、未见过的数据上有效地进行预测或分类。

以下是一些关键点来解释泛化的意义：

1. **训练误差和测试误差**：训练误差指模型在用来训练的已知数据集上的误差；而测试误差指模型在未见过的测试数据集上的误差。一个有良好泛化能力的模型，训练误差和测试误差应该都比较低，且两者差距不大。

2. **防止过拟合**：过拟合（overfitting）是指模型在训练数据上表现非常好，但在测试数据上表现很差。这通常是因为模型过于复杂，捕捉到了训练数据中的噪音。泛化不好的模型通常会过拟合。

3. **防止欠拟合**：欠拟合（underfitting）是指模型不能很好地捕捉训练数据中的模式，因而在训练数据和测试数据上都表现不好。这通常是因为模型过于简单。

4. **正则化方法**：为了提高模型的泛化能力，可以使用正则化技术，比如L1正则化（Lasso）、L2正则化（Ridge），或者使用Dropout等方法，来减少模型的复杂度，防止过拟合。

5. **交叉验证（Cross-validation）**：这是评估模型泛化能力的一种方法，通过在多个训练和验证数据集上进行训练和评估，来保证模型不是仅仅在某一个特定的数据集上表现良好。

总之，泛化能力是评价一个机器学习模型质量的关键指标，良好的泛化能力意味着模型在处理实际应用中的新数据时也能保持较高的准确性和可靠性。





训练模型时，需要验证学习率等各方面的参数大小配置。

如果图片不是 28x28，怎么缩放大小。

如果图片是彩色的，这么转换为黑白。

一边训练一边验证数据集。