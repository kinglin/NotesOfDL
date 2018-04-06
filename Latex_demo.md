# Latex demo with markdown

#### matrix
$\begin{bmatrix} 1 & x & x^2 \\\ 1 & y & y^2 \\\ 1 & z & z^2 \\ \end{bmatrix}$

#### hat
$\hat{y}=P\{y=1|x\}$  
$\hat{y}=\sigma(\omega^T+b)$  
$L(\hat{y},y)=-(ylog\hat{y}+(1-y)log(1-\hat{y}))$  

#### frac & sum
$J(\omega,b)=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y},y)$

#### partial
$\omega:=\omega-\alpha\frac{\partial J(\omega,b)}{\partial\omega}$  

#### big braket
$ReLU:
\begin{equation}
g(z)=
\begin{cases}
0& \text{z<0}\\
1& \text{z>=0}
\end{cases}
\end{equation}$