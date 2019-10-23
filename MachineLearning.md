# 0.模型性能评估

对于分类器或者说分类算法，评价指标主要有[precision](https://en.wikipedia.org/wiki/Precision_and_recall#Precision)，[recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)，[F1 score](https://en.wikipedia.org/wiki/F1_score)等

## 0.1. ROC

ROC和AUC通常是用来评价一个二值分类器的好坏

### ROC曲线

![image-20190411110145800](/Users/jianfengyuan/Library/Application Support/typora-user-images/image-20190411110145800.png)

曲线坐标上：

- X轴是FPR（表示假阳率-预测结果为positive，但是实际结果为negitive，FP/(N)）
- Y轴式TPR（表示真阳率-预测结果为positive，而且的确真实结果也为positive的,TP/P）

那么平面的上点(X,Y)：

- (0,1)表示所有的positive的样本都预测出来了，分类效果最好

- (0,0)表示预测的结果全部为negitive

- (1,0)表示预测的结果全部分错了，分类效果最差

- (1,1)表示预测的结果全部为positive

	> 

![image-20190411105825492](/Users/jianfengyuan/Library/Application Support/typora-user-images/image-20190411105825492.png)

- **正确率**([Precision](https://en.wikipedia.org/wiki/Precision_and_recall#Precision))：

	$Precision=\frac{TP}{TP+FP}$

- **真阳性率**(True Positive Rate，[TPR](https://en.wikipedia.org/wiki/Sensitivity_and_specificity))，**灵敏度**([Sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Sensitivity))，**召回率**([Recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall))：

	$Sensitivity=Recall=TPR=\frac{TP}{TP+FN}$

- **真阴性率**(True Negative Rate，[TNR](https://en.wikipedia.org/wiki/Sensitivity_and_specificity))，**特异度**([Specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Specificity))：

	$Specificity=TNR=\frac{TN}{FP+TN}$

- **假阴性率**(False Negatice Rate，[FNR](https://en.wikipedia.org/wiki/False_positives_and_false_negatives#False_positive_and_false_negative_rates))，**漏诊率**( = 1 - 灵敏度)：

	$FNR=\frac{FN}{TP+FN}$

- **假阳性率**(False Positice Rate，[FPR](https://en.wikipedia.org/wiki/False_positive_rate))，**误诊率**( = 1 - 特异度)：

	$FPR=\frac{FP}{FP+TN}$ 

#### ROC曲线建立

一般默认预测完成之后会有一个概率输出p，这个概率越高，表示它对positive的概率越大。
现在假设我们有一个threshold，如果p>threshold，那么该预测结果为positive，否则为negitive，按照这个思路，我们多设置几个threshold,那么我们就可以得到多组positive和negitive的结果了，也就是我们可以得到多组FPR和TPR值了
将这些(FPR,TPR)点投射到坐标上再用线连接起来就是ROC曲线了

#### **如何判断ROC曲线的好坏？**

FPR表示模型虚报的响应程度， 而TPR你表示模型预测响应的覆盖程度，所以我们希望虚报的越少越好，覆盖的越多越好，所以TPR越高，同时FRP越低，那么模型的性能越好

#### ROC曲线无视样本不平衡

因为TPR和FPR分别是基于实际表现1和0出发的，也就是说他们分别在实际的正样本和负样本中来观察相关概率问题，因此无论样本中正反样例比例是否平衡，ROC都不会被影响

![img](https://pic1.zhimg.com/50/v2-432e490292db97f258ecf04a7b17ef1c_hd.gif)



## 0.2 AUC(曲线下的面积,Area under curve)

首先AUC值是一个概率值，当你随机挑选一个正样本以及一个负样本，当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值。当然，AUC值越大，当前的分类算法越有可能将正样本排在负样本前面，即能够更好的分类。

如果连接对角线，AUC面积刚好为0.5,对角线的实际含义为**随机判断响应与不响应，正负样本的覆盖率都为50%，表示随机效果。**因为ROC曲线越抖越好，因此理想形状是一个正方形，ROC面积为1，所以一般AUC的值一般在0.5-1之间



# 1.Linear Regression and Logistic Regression

**LR**回归是一个线性的二分类模型，主要是**计算在某个样本特征下事件发生的概率**，比如根据用户的浏览购买情况作为特征来计算它是否会购买这个商品，抑或是它是否会点击这个商品。然后**LR**的最终值是根据一个线性和函数再通过一个**sigmoid**函数来求得，这个线性和函数权重与特征值的累加以及加上偏置求出来的，所以在训练**LR**时也就是在训练线性和函数的各个权重值**w**。	
$$
\begin{align}
h_w(x)=\frac{1}{1+e^{−(w^Tx+b)}}
\end{align}
$$

## 推导

关于这个权重值**w**一般使用最大似然法来估计,假设现在有样本$\{x_i,y_i\}$,其中$x_i$表示样本的特征，$y_i∈\{0,1\}$表示样本的分类真实值，$yi=1$的概率是$p_i$,则$y_i=0$的概率是$1−p_i$，那么观测概率为:


$$
\begin{align}
正例可能性：p(y_i = 1) = p^{yi}_i\\反例可能性：p(y_i = 0) = (1−p_i)^{1−y_i}\\
p(y_i)=p^{yi}_i∗(1−p_i)^{1−y_i}
\end{align}
$$

​										假设$p(y_i)\approx{h_w(x_i)}$

则最大似然函数为:
$$
∏(h_w(x_i)y_i∗(1−h_w(x_i))^{1−y_i})
$$

对这个似然函数取对数之后就会得到的表达式

$$
\begin{align}
L(w)=∑_i(y_i∗logh_w(x_i)+(1−y_i)∗log(1−h_w(x_i)))\\\\

=∑_i(y_i∗(w^Tx_i)+log(1+e	^{w^Tx_i})))
\end{align}
$$
估计这个$L(w)$的极大值就可以得到$w$的估计值。

所以求解问题就变成了这个最大似然函数的最优化问题，这里通常会采用**随机梯度下降法**和**拟牛顿迭代法**来进行优化

# 2. SVM

SVM 损失函数：

![L_i=\displaystyle\sum_{j\not =y_i}[max(0,w^T_jx_i-w^T_{y_i}x_i+\Delta)]](https://www.zhihu.com/equation?tex=L_i%3D%5Cdisplaystyle%5Csum_%7Bj%5Cnot+%3Dy_i%7D%5Bmax%280%2Cw%5ET_jx_i-w%5ET_%7By_i%7Dx_i%2B%5CDelta%29%5D)

对应正确分类的W的行向量的梯度:

![\displaystyle\nabla_{w_{y_i}}L_i=-(\sum_{j\not=y_i}1(w^T_jx_i-w^T_{y_i}x_i+\Delta>0))x_i](https://www.zhihu.com/equation?tex=%5Cdisplaystyle%5Cnabla_%7Bw_%7By_i%7D%7DL_i%3D-%28%5Csum_%7Bj%5Cnot%3Dy_i%7D1%28w%5ET_jx_i-w%5ET_%7By_i%7Dx_i%2B%5CDelta%3E0%29%29x_i)

$j\not =y_i$行的梯度是

![\displaystyle\nabla_{w_j}L_i=1(w^T_jx_i-w^T_{y_i}x_i+\Delta>0)x_i](https://www.zhihu.com/equation?tex=%5Cdisplaystyle%5Cnabla_%7Bw_j%7DL_i%3D1%28w%5ET_jx_i-w%5ET_%7By_i%7Dx_i%2B%5CDelta%3E0%29x_i)



# 3.Softmax

**实操事项：数值稳定。**编程实现softmax函数计算的时候，中间项![e^{f_{y_i}}](https://www.zhihu.com/equation?tex=e%5E%7Bf_%7By_i%7D%7D)和![\sum_j e^{f_j}](https://www.zhihu.com/equation?tex=%5Csum_j+e%5E%7Bf_j%7D)因为存在指数函数，所以数值可能非常大。除以大数值可能导致数值计算的不稳定，所以学会使用归一化技巧非常重要。如果在分式的分子和分母都乘以一个常数![C](https://www.zhihu.com/equation?tex=C)，并把它变换到求和之中，就能得到一个从数学上等价的公式：

![\frac{e^{f_{y_i}}}{\sum_je^{f_j}}=\frac{Ce^{f_{y_i}}}{C\sum_je^{f_j}}=\frac{e^{f_{y_i}+logC}}{\sum_je^{f_j+logC}}](https://www.zhihu.com/equation?tex=%5Cfrac%7Be%5E%7Bf_%7By_i%7D%7D%7D%7B%5Csum_je%5E%7Bf_j%7D%7D%3D%5Cfrac%7BCe%5E%7Bf_%7By_i%7D%7D%7D%7BC%5Csum_je%5E%7Bf_j%7D%7D%3D%5Cfrac%7Be%5E%7Bf_%7By_i%7D%2BlogC%7D%7D%7B%5Csum_je%5E%7Bf_j%2BlogC%7D%7D)

![C](https://www.zhihu.com/equation?tex=C)的值可自由选择，不会影响计算结果，通过使用这个技巧可以提高计算中的数值稳定性。通常将![C](https://www.zhihu.com/equation?tex=C)设为![logC=-max_jf_j](https://www.zhihu.com/equation?tex=logC%3D-max_jf_j)。该技巧简单地说，就是应该将向量![f](https://www.zhihu.com/equation?tex=f)中的数值进行平移，使得最大值为0。代码实现如下：

```python
f = np.array([123, 456, 789]) # 例子中有3个分类，每个评分的数值都很大
p = np.exp(f) / np.sum(np.exp(f)) # 不妙：数值问题，可能导致数值爆炸

# 那么将f中的值平移到最大值为0：
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # 现在OK了，将给出正确结果
```

# 4.决策树

决策树的损失函数：

关于决策的损失函数，可以先从决策树的目的是通过属性来分割样本，然后得到准确的分类，则最准确的分类就是当所有叶子都只有一个一个样本时，次数决策树的正确率最高，此时每个叶子中的交叉熵为0，符合决策树通过不断分裂，降低叶子的交叉熵，从而更好的对样本进行分类，因此，在离散决策树中，使用交叉熵来作为决策树的经验损失

# 优化方法

https://zhuanlan.zhihu.com/p/32230623

### 1.梯度下降

**LR**的损失函数为:
$$
J(w)=-\frac{1}{N}\sum_{i}(y_i∗log(h_w(x_i))+(1−y_i)∗log(1−h_w(x_i)))
$$

这样就变成了求$ min(J(w))$
其更新w的过程为
$$
w:=w−α∗▿J(w)\\
w:=w−α∗\frac{1}{N}\sum_{i}^{N}((h_w(x_i)-y_i)*x_i)
$$
缺点：

​	1.容易陷入局部最优解

​	2.每次对当前样本计算cost的时候都要遍历全部样本，速度比较慢

因此许多框架会使用随机梯度下降

**普通更新**。最简单的更新形式是沿着负梯度方向改变参数（因为梯度指向的是上升方向，但是我们通常希望最小化损失函数）。假设有一个参数向量**x**及其梯度**dx**，那么最简单的更新的形式是：

```python
# 普通更新
x += - learning_rate * dx
```

其中learning_rate是一个超参数，它是一个固定的常量。当在整个数据集上进行计算时，只要学习率足够低，总是能在损失函数上得到非负的进展。

**动量（Momentum）更新**是另一个方法，这个方法在深度网络上几乎总能得到更好的收敛速度。该方法可以看成是从物理角度上对于最优化问题得到的启发。损失值可以理解为是山的高度（因此高度势能是![U=mgh](https://www.zhihu.com/equation?tex=U%3Dmgh)，所以有![U\propto h](https://www.zhihu.com/equation?tex=U%5Cpropto+h)）。用随机数字初始化参数等同于在某个位置给质点设定初始速度为0。这样最优化过程可以看做是模拟参数向量（即质点）在地形上滚动的过程。

因为作用于质点的力与梯度的潜在能量（![F=-\nabla U](https://www.zhihu.com/equation?tex=F%3D-%5Cnabla+U)）有关，质点**所受的力**就是损失函数的**（负）梯度**。还有，因为![F=ma](https://www.zhihu.com/equation?tex=F%3Dma)，所以在这个观点下（负）梯度与质点的加速度是成比例的。注意这个理解和上面的随机梯度下降（SDG）是不同的，在普通版本中，梯度直接影响位置。而在这个版本的更新中，物理观点建议梯度只是影响速度，然后速度再影响位置：

```python
# 动量更新
v = mu * v - learning_rate * dx # 与速度融合
x += v # 与位置融合
```

在这里引入了一个初始化为0的变量**v**和一个超参数**mu**。说得不恰当一点，这个变量（mu）在最优化的过程中被看做*动量*（一般值设为0.9），但其物理意义与摩擦系数更一致。这个变量有效地抑制了速度，降低了系统的动能，不然质点在山底永远不会停下来。通过交叉验证，这个参数通常设为[0.5,0.9,0.95,0.99]中的一个。和学习率随着时间退火类似，动量随时间变化的设置有时能略微改善最优化的效果，其中动量在学习过程的后阶段会上升。

### Adagrad

**Adagrad**是一个由[Duchi等](https://link.zhihu.com/?target=http%3A//jmlr.org/papers/v12/duchi11a.html)提出的适应性学习率算法

```python
# 假设有梯度和参数向量x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

注意，变量**cache**的尺寸和梯度矩阵的尺寸是一样的，还跟踪了每个参数的梯度的平方和。这个一会儿将用来归一化参数更新步长，归一化是逐元素进行的。注意，接收到高梯度值的权重更新的效果被减弱，而接收到低梯度值的权重的更新效果将会增强。有趣的是平方根的操作非常重要，如果去掉，算法的表现将会糟糕很多。用于平滑的式子**eps**（一般设为1e-4到1e-8之间）是防止出现除以0的情况。Adagrad的一个缺点是，在深度学习中单调的学习率被证明通常过于激进且过早停止学习。

### RMSprop

**RMSprop**是一个非常高效，但没有公开发表的适应性学习率方法。有趣的是，每个使用这个方法的人在他们的论文中都引用自Geoff Hinton的Coursera课程的[第六课的第29页PPT](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/%7Etijmen/csc321/slides/lecture_slides_lec6.pdf)。这个方法用一种很简单的方式修改了Adagrad方法，让它不那么激进，单调地降低了学习率。具体说来，就是它使用了一个梯度平方的滑动平均：

```python
cache =  decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

在上面的代码中，decay_rate是一个超参数，常用的值是[0.9,0.99,0.999]。其中**x+=**和Adagrad中是一样的，但是**cache**变量是不同的。因此，RMSProp仍然是基于梯度的大小来对每个权重的学习率进行修改，这同样效果不错。但是和Adagrad不同，其更新不会让学习率单调变小。

### Adam

[Adam](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1412.6980)是最近才提出的一种更新方法，它看起来像是RMSProp的动量版。简化的代码是下面这样：

```python
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```

注意这个更新方法看起来真的和RMSProp很像，除了使用的是平滑版的梯度**m**，而不是用的原始梯度向量**dx**。论文中推荐的参数值**eps=1e-8, beta1=0.9, beta2=0.999**。在实际操作中，我们推荐Adam作为默认的算法，一般而言跑起来比RMSProp要好一点。但是也可以试试SGD+Nesterov动量。完整的Adam更新算法也包含了一个偏置*（bias）矫正*机制，因为**m,v**两个矩阵初始为0，在没有完全热身之前存在偏差，需要采取一些补偿措施。

# 激励函数

## 1. ReLu

## 2.tanh

## 3.sgmoid

# 正则化(Regularization)

假设一个数据集和一个权重集**W**能够正确地分类每个数据（即所有边界都满足对于所有i都有
$L_i = 0$），问题在于这个**W**并不唯一：可能有很多相似的**W**都能正确地分类所有的数据，该**W**对于每个数据，loss都是0，我们可以乘上一个$\lambda$，当$\lambda>1$时，任何正确分类的损失值都为0，但是对于错误分类，损失值则会变大。我们希望向某些特定的权重**W**添加一些偏好，对于其他权重则不添加，以此来消除模糊性。这个方法是向损失函数增加一个正则化惩罚，最常用的是L2范式。

从直观上来看，L2惩罚倾向于更小更分散的权重向量，这就会鼓励分类器最终将所有维度上的特征都用起来，而不是强烈依赖其中少数几个维度。这一效果将会提升分类器的泛华性能，并避免过拟合。

 

## 梯度消失

*Sigmoid函数饱和使梯度消失*。sigmoid神经元有一个不好的特性，就是当神经元的激活在接近0或1处时会饱和：在这些区域，梯度几乎为0。回忆一下，在反向传播的时候，这个（局部）梯度将会与整个损失函数关于该门单元输出的梯度相乘。因此，如果局部梯度非常小，那么相乘的结果也会接近零，这会有效地“杀死”梯度，几乎就有没有信号通过神经元传到权重再到数据了。还有，为了防止饱和，必须对于权重矩阵初始化特别留意。比如，如果初始化权重过大，那么大多数神经元将会饱和，导致网络就几乎不学习了。

## 无监督学习

### 聚类

#### k均值 (kmeans)

由用户指定k个初始知心，以作为聚类的类别，重复迭代直至算法收敛

算法流程：

1. 选取k个初始质心，将其类别标为质心所对应

#### 最大期望聚类

### 降维

#### 潜语义分析(LSA)

#### 主成分分析(PCA)

#### 奇异值分解(SVD)

### word2vec对物品进行推荐

## 基于内容的推荐算法(Content Base,CB)

### 基于用户的内容推荐算法

根据用户特征信息，给相似的用户推荐物品

### 基于物品的内容推荐算法

根据物品的特征信息，给用户推荐相似的物品



## 基于协同过滤的推荐算法

### 协同过滤(collabrative filtering, CF)

基于用户的行为信息给用户进行推荐

### 基于近邻的协同过滤

- 基于用户的协同过滤(User-CF)



- 基于物品的协同过滤(Item-CF)

### 基于模型的协同过滤

- 奇异值分解(SVD)
- 潜在语义分析(LSA)
- 支持向量机(SVM)