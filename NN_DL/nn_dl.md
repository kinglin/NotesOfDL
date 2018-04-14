
[TOC]

## week 1
+ different NN use in different area ( eg. CNN for image processing )

![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_1.jpg)

+ why large scale nn becomes popular these days:
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_2.png)

+ The activate function is changed from sigmod function to ReLU, because the training procedure of sigmod function is very slow, and the gradient of ReLU is static when the value is bigger than 0.

##week 2

### Binary classification
+ Notation:
    + m: number of pictures
    + n: dimension of each picture( number of pixels )
    + X: the whole set of pictures, n*m matrix
    + Y: out put of dataset, 1*m matrix

### Logistic regression
given x, and want to know $\hat{y}=P\\{y=1|x\\}$  
output is $\hat{y}=\sigma(\omega^T+b)$  

### Logistic Regression cost function
notaion:   
$x^{(i)}$ means #i record  
logistic regression loss function:  
$L(\hat{y},y)=-(ylog\hat{y}+(1-y)log(1-\hat{y}))$  
so the cost funtion is the average of all records:  
$J(\omega,b)=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y},y)$

>actually, the basic concept of classification is probabilities  
for example,   
if $y=1: p(y|x)=\hat{y}$  
if $y=0: p(y|x)=1-\hat{y}$  
so that $p(y|x)=\hat{y}^y(1-\hat{y})^{1-y}$  
and calculate log for both sides, we can get loss function above

### Gradient descent
>repeat{  
>&emsp;&emsp;$\omega:=\omega-\alpha\frac{dJ(\omega)}{d\omega}$    
>}

if derivative is positive, which means $\frac{dJ(\omega)}{d\omega}>0$, $\omega$ will be subtracted, and in converse, it will increase.
>repeat{  
>&emsp;&emsp;$\omega:=\omega-\alpha\frac{\partial J(\omega,b)}{\partial\omega}$  
> &emsp;&emsp;$b:=b-\alpha\frac{\partial J(\omega,b)}{\partial b}$  
>}

but in code we use $d\omega$ instead of $\frac{\partial J(\omega,b)}{\partial\omega}$, and b instead of $\frac{\partial J(\omega,b)}{\partial b}$

### Derivatives with Computation Graph
make $J=3(a+bc)$, and $u=bc$, $v=a+u$  
then $\frac{dJ}{da}=\frac{dJ}{dv}\cdot\frac{dv}{da}=3\times1=3$   
To simplify the representation in code, we use $da$ instead of $\frac{dJ}{da}$  
the whole example:
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_3.png)

### Logistic regression gradient descent
This is the procedure of back probagation  
We assume that $X$ stands for 2-D parameters, so that we have $\omega_1,\omega_2$ and $b$ three coefficients here, and use the back probagation.  
At first, we can write the calculation procedure down from left to right, and then we calculate the dirivatives one by one from right to left, using the basic calculus mentioned above. Finally, we get $\omega_1:=\omega_1-\alpha d\omega_1=\omega_1-\alpha\cdot x_1\cdot(a-y)$ and the other two formular.
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_4.png)

### Vectorization
***whenever possible, avoid explicit for-loops***  
vectorization is to make your loop more efficient  
the loop in our program is to calculate $\omega_ix_i$ , so we have to write a for loop from 1 to n to calculate. But if we use vectorization, the procedure will become a line time a colume in matrix, just like $\omega^TX$, because the CPU can use parallel thread to do this work. In python, the method is numpy.dot($\omega,x$). Following is the code.
```python3
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print(toc-tic)

#results
#249825.69335160166
#0.0015769004821777344
```

another example
```python3
v = np.array([5,-1,3,2])
print(v)
u = np.exp(v)
print(u)
u = np.maximum(v,0)
print(u)
u = 1/v
print(u)

#result
#[ 5 -1  3  2]
#[148.4131591    0.36787944  20.08553692   7.3890561 ]
#[5 0 3 2]
#[ 0.2        -1.          0.33333333  0.5       ]
```

**Vectorizing Logistic Regression**  

We get these formular:  
$z^{(1)}=w^Tx^{(1)}+b$&emsp;&emsp;$a^{(1)}=\sigma(z^{(1)})$&emsp;&emsp;  
$z^{(2)}=w^Tx^{(2)}+b$&emsp;&emsp;$a^{(2)}=\sigma(z^{(2)})$&emsp;&emsp;  
...  
$z^{(m)}=w^Tx^{(m)}+b$&emsp;&emsp;$a^{(m)}=\sigma(z^{(m)})$&emsp;&emsp;  
In ordinary programming procedure, we have 2 for-loop here: a loop for m, and a loop for n(w*x)  
But with help of numpy, we can make it more efficient.  
At first, we regard all   
$x^{(i)}$ as an $n_x\times m$ matrix $X=[x^{(1)}\ x^{(2)}\ \cdots\ x^{(m)}]$, and  
$w$ as an $n_x\times 1$ column vector $w=[w_1\ w_2\ \cdots \ w_n]^T$, and   
$b$ as an $1\times m$ row vector $b=[b\ b\ \cdots \ b]$  

so we make   
$Z=[z^{(1)}\ z^{(2)}\ \cdots\ z^{(m)}]=w^TX+b=np.dot(w.T, X)+b$  
$b$ here will be broadcasted by python, so it won't trouble if it is a real number actually.

After getting $Z$ calculated by just one line code, we need to calculate $A=[a^{(1)}\ a^{(2)}\ \cdots\ a^{(m)}]=\sigma(Z)$ next.

##week 3
**Add One Hidden Layer**  
In general, if a neural network has one hidder layer and one output layer, we call it a **two layer** neural network  
$W^{[1]}$ means $W$ in layer one  
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_5.png)

Four lines of code to calculate the output of two layers NN:  
$W^{[1]}=[w_1^{[1]}\ w_2^{[1]}\ w_3^{[1]}\ w_4^{[1]}]^T$
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_6.jpg)

To get rid of for-loop for m patterns:  
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_7.png)

We can regard $X$ as $A^{[0]}$, so the four lines code above(in red rectangle) can be simplified into a loop of 2 lines:  
```python3
for i in num_of_layers:
    Z[i+1] = W[i+1]A[i]+b
    A[i+1] = sigmoid(Z[i+1])
```

***To make dimensions more clear***  
$z^{[1]} = W^{[1]}x + b^{[1]}$——>$4*1,4*3,3*1,4*1$  
$a^{[1]} = \sigma (z^{[1]})$——>$4*1,4*1$  
$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$——>$1*1,1*4,4*1,1*1$  
$a^{[2]} = \sigma (z^{[2]})$——>$1*1,1*1$  

To avoid for-loop of 1 to m:  
$Z^{[1]} = W^{[1]}X + b^{[1]}$——>$4*m,4*3,3*m,4*m$  
$A^{[1]} = \sigma (Z^{[1]})$——>$4*m,4*m$  
$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$——>$1*m,1*4,4*m,1*m$  
$A^{[2]} = \sigma (Z^{[2]})$——>$1*m,1*m$  


**Activation Function**  
Activation function can be different in different layers, even in different nodes  
One stategy is that we choose **ReLU** $a=max(0,z)$ or **tanh** $tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$ or instead of $\sigma(z)$ in hidden layers, because the scope of $tanh(z)$ is $[-1,1]$, and $0$ as mean is better than $0.5$  
But in binary classification(which $y=0,1$), the output node is better to use $\sigma(z)$, while the nodes in hidden layers use **ReLU or Leaky ReLU**  
The main reason to use **ReLU** is that the derivatives is much bigger than $tanh$ or $\sigma$ when $z$ is very small or very large. When $z$ is very small, ReLU just makes the derivatives as 0.  
It's very difficult to know in advance about what kind of activation function suits the specific problem best, so trail and error is essential.  

**Why having non-linear activation function?**  
If all the activation function in hidden layers are linear function, then it is of no difference from connecting to output nodes(sigmoid function) directly, because all the $w$ will be combined into one.  
But in the case of regression problems like predicting housing price, we can use linear function in output nodes(actually, ReLU is better)

**Derivatives of activation functions**  
$\sigma: g(z)=\frac{1}{1+e^{-z}}==>g^{'}(z)=g(z)(1-g(z))$  
$tanh: g(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}==>g^{'}(z)=1-g^2(z)$  
$ReLU: g(z)=max(0,z)$  
$\begin{equation}
g^{'}(z)=
\begin{cases}
0& \text{z<0}\\
1& \text{z>=0}
\end{cases}
\end{equation}$  
$Leaky ReLU: g(z)=max(0.01z,z)$  
$\begin{equation}
g^{'}(z)=
\begin{cases}
0.01& \text{z<0}\\
1& \text{z>=0}
\end{cases}
\end{equation}$  

**Formulars for computing derivatives**
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_8.png)

**Random Initialization**  
Why don't use 0 as initial weights?  
No matter how many epoch we process, the hidden units are all symmetric, and there will be no difference between them.(weights will all be the same)
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_9.png)

What should we do?  
Random initialize the weights, and give a relative small factor(like 0.01)  
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_10.png)










