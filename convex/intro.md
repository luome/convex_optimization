# 凸优化问题

## 普遍的凸优化问题

1. 凸优化问题

 (a) 回归(Regression)

 1. Least Squares: $$min_{\beta} \sum_i (y_i - x_i^T\beta)^2$$

 2. Least Absolute Deviations: $$min_\beta\sum_i|y_i-x_i^T\beta|$$

 (b) 正则化回归(Regularized Regression)

 1. Lasso: $$min_\beta\sum_i(y_i - x_i^T\beta)^2  s.t. \sum_j|\beta_j|\leq t$$

 (c) 去噪(Denosing)

 1. Total-Variation denosing / Fused Lasso

 (d) 分类(Classification)

 1. Logistic regression

 2. 0-1 Loss

 3. Hinge loss / **SVM**

 (e) 其他问题

 1. Travelling-salesman problem(TSP)

 2. Planning / Discrete optimization

 3. Maximum-likelihood estimation

2. 非优化问题

  (a) Hypothesis testing / p-values 

  (b) Boosting 

  (c) Random Forests

  (d) Cross-validation / bootstrap
----
## 主要的概念

### 凸集和凸函数

凸集: $$C \in {\cal R}^n$$ 是一个凸集如果$$ x, y \in {\cal C} \Rightarrow tx + (1-t)y \in {\cal C}, 其中 0 \leq t \leq 1.$$ 

凸函数: $$ f : {\cal R^n } \rightarrow {\cal R}$$是凸函数如果$${\mit dom(f)} \subseteq {\cal R^n } 是凸的,且f(tx + (1-t)y) \ leq tf(x) + (1-t)f(y), 其中 0 \leq t \leq 1.$$

### 凸优化问题

#### 优化问题:

$$
\begin{split}
min_{x \in D} f(x)\\
subject\ to\ g_i(x) \leq 0, i = 1,...,m\\
        h_j(x)=0, j=1,...,r\\
\end{split}
$$
$$
{\mit where\ D= dom(f)\cap \bigcap_{i=1}^m dom(g_i) \cap \bigcap_{j=1}^p dom(h_j)\ is\ the\ intersection\ of\ all\ the\ domains.}
$$

#### 凸优化问题:

如果上述优化问题满足下列条件,那么它是一个凸优化问题:

1. $${\mit f}$$和$${\mit g_i, i=1,...,m}$$是凸的。

2. $${\mit h_j, j=1,...p}$$是放射的，意味着$$h_j(x) = a_j^Tx + b_j, j=1,...,p$$

#### 局部极小即全局极小

对于一个凸优化问题，如果x是可行的，且最小化f为一个局部区域，

$$f(x) \leq f(y)\ \text{对于所有可行y}, \left \|x-y\right \|_2 \leq \rho,\ \text{那么对于所有所有可行y}, f(x) \leq f(y).
$$

简单的说，在凸优化问题中，**局部极小就是全局极小**。

