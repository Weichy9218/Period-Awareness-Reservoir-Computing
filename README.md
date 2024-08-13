# Period-Awareness-Reservoir-Computing
PRAC experiment writing thoughts：数据说明要有，且一定要有落脚点。


##### Classification Performance Evaluation

1.加入大类方法的纵向对比。

在对比实验后，不止说我们方法的好，更全面地讲清楚我们的方法，在加入periodicity awareness后，能明显优于Reservoir computing，即将RC优化，超过了网络训练方法即对比方法TimesNet, LSTNet, InceptionTime等方法。

```latex
Statistically, PARC exhibits significant differences from most baselines and consistently wins more in one-to-one comparisons. Across different method categories, we find that existing RC methods (rmESN and ConvMESN) generally lag behind fully trainable neural networks (such as LSTNet and InceptionTime).
Nevertheless, our method effectively bridges this gap, achieving top-performing results and highlighting the benefits of incorporating periodicity awareness.
```

2.时间和计算复杂度放在一起分析，能给出时间复杂度的理论计算最好

```latex
The main factors affecting PARC's complexity include sample size $M$, series length $L$, period limit $\kappa$, and reservoir size $S$.
According to experiments, PARC's runtime is primarily spent on iteration of PerioRes modules and computation of dynamic features, with a complexity of $O(\kappa M(S^{2}L+S^{3}))$.
This means that under specific settings, the runtime is nearly linear with the data complexity (characterized by $M$ and $L$), showcasing exceptional efficiency. 
```

时间的分析要全面，不能简单比较结果，comparison methods要分类，trainable 普遍效率低，但不一定要全部点明。师兄只点明了LSTNet，因为其网络结构非常简单，纵向上看很直观。关键的non-trainable方法一定要点名，和Rocket方法比时间，和Hydra、Minirocket比acc。在比较时间时提及rmESN时间最快，但是精度欠缺。在和Hydra\Minirocket对比时，要分析其时间优势和性能优势的原因。

```latex
Clearly, PARC is faster than all those trainable models, either fully or partially, even against architectures as simple as LSTNet. 
Compared to the non-trainable Rocket, our runtime remains impressively less than half. As for the other methods, RC-based rmESN performs the fastest but also the least accurate.

On the other hand, although Hydra and MiniRocket require shorter runtimes than ours, their accuracy tends to be somewhat lower.
下面的分析很精彩：
Their speed advantage could stem from the simplicity and inherent parallelism of convolutional operations as opposed to the sequential computation paradigm of RNNs and RC models.
关键在于：对不同的对比方法分类，一类方法的分析>单个算法的分析
```



##### Ablation Study

Varying RC Models: 强调的是Periodicity awareness的用处，对不同的ESN结构的适应性很强

```latex
Even with the weakest ESN, PARC remains competitive with the state-of-the-art InceptionTime and surpasses existing RC methods, underscoring its superiority and versatility in adapting to various RC models.
```

Varying Feature Utilization: 强调模型选择的特征的特点，本质上还是PARC模型结构的设计，integrate past and present information(feature)

```
This is because our feature effectively integrates past and present information, encapsulating the inherent temporal dynamics, whereas the other features fail to achieve.
```

Varying Classifiers: 一致性强的classifier分析结果的稳定性+相差较多的classifier分析原因：由于输出特征的muti-scale latent period，复杂的特征结果不易被KNN和Logistics分割。Logistic Regression 假设数据具有线性边界，而KNN对异常值敏感且易受数据规模和维度的影响（维数灾难）。当然也不用说的特别详细

```
RF and SVM are relatively weaker, yet not statistically significant, indicating that the learned features adapt well to various classifiers. 
However, both LR and KNN yield inferior outcomes, suggesting difficulties in integrating features across multi-scale latent periods.
```



##### Impact of Dataset Periodicity

主要研究数据的周期性，从数据本身的层面证明，我们方法的有效性。

1. 数据自相关分数（不知道咋算的）
2. 相对准确率提升

计算二者的皮尔逊系数，证明数据存在period，period awareness有助于Times series classification accuracy的提升。当我们发现数据具有Periodicity时，我们的方法更适用。



##### Hyperparameter Study

不仅要分析趋势，还要给出PARC即我们模型建议使用的参数。同时还要强调如$\kappa$ 即周期保有率对实验结果的影响，以及reservoir size对结果影响不大但是对时间影响很大，热力图谱半径s比p要更大的暗指可能sequential比periodic dynamics更重要。谱半径反应的是模型的记忆性能，谱半径越大代表特征记住的时间越长，可能代表特征约重要。





写作要详细、具体有内容，不要废话，我太爱写废话了。

好的论文句句都有干货。
