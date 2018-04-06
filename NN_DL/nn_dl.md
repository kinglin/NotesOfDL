
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
given x, and want to know $\hat{y}=P\\{y=1|x\\}$<br>
output is $\hat{y}=\sigma(\omega^T+b)$<br>

### Logistic Regression cost function
notaion: <br>
$x^{(i)}$ means #i record<br>
logistic regression loss function:<br>
$L(\hat{y},y)=-(ylog\hat{y}+(1-y)log(1-\hat{y}))$<br>
so the cost funtion is the average of all records:<br>
$J(\omega,b)=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y},y)$

>actually, the basic concept of classification is probabilities<br>
for example, <br>
if $y=1: p(y|x)=\hat{y}$<br>
if $y=0: p(y|x)=1-\hat{y}$<br>
so that $p(y|x)=\hat{y}^y(1-\hat{y})^{1-y}$<br>
and calculate log for both sides, we can get loss function above

### Gradient descent
>repeat{<br>
>&emsp;&emsp;$\omega:=\omega-\alpha\frac{dJ(\omega)}{d\omega}$    
>}

if derivative is positive, which means $\frac{dJ(\omega)}{d\omega}>0$, $\omega$ will be subtracted, and in converse, it will increase.
>repeat{<br>
>&emsp;&emsp;$\omega:=\omega-\alpha\frac{\partial J(\omega,b)}{\partial\omega}$<br>
> &emsp;&emsp;$b:=b-\alpha\frac{\partial J(\omega,b)}{\partial b}$<br>
>}

but in code we use $d\omega$ instead of $\frac{\partial J(\omega,b)}{\partial\omega}$, and b instead of $\frac{\partial J(\omega,b)}{\partial b}$

### Derivatives with Computation Graph
make $J=3(a+bc)$, and $u=bc$, $v=a+u$<br>
then $\frac{dJ}{da}=\frac{dJ}{dv}\cdot\frac{dv}{da}=3\times1=3$ <br>
To simplify the representation in code, we use $da$ instead of $\frac{dJ}{da}$<br>
the whole example:
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_3.png)

### Logistic regression gradient descent
This is the procedure of back probagation<br>
We assume that $X$ stands for 2-D parameters, so that we have $\omega_1,\omega_2$ and $b$ three coefficients here, and use the back probagation.<br>
At first, we can write the calculation procedure down from left to right, and then we calculate the dirivatives one by one from right to left, using the basic calculus mentioned above. Finally, we get $\omega_1:=\omega_1-\alpha d\omega_1=\omega_1-\alpha\cdot x_1\cdot(a-y)$ and the other two formular.
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_4.png)

### Vectorization
***whenever possible, avoid explicit for-loops***<br>
vectorization is to make your loop more efficient<br>
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

**Vectorizing Logistic Regression**<br>

We get these formular:<br>
$z^{(1)}=w^Tx^{(1)}+b$&emsp;&emsp;$a^{(1)}=\sigma(z^{(1)})$&emsp;&emsp;<br>
$z^{(2)}=w^Tx^{(2)}+b$&emsp;&emsp;$a^{(2)}=\sigma(z^{(2)})$&emsp;&emsp;<br>
...<br>
$z^{(m)}=w^Tx^{(m)}+b$&emsp;&emsp;$a^{(m)}=\sigma(z^{(m)})$&emsp;&emsp;<br>
In ordinary programming procedure, we have 2 for-loop here: a loop for m, and a loop for n(w*x)<br>
But with help of numpy, we can make it more efficient.<br>
At first, we regard all <br>
$x^{(i)}$ as an $n_x\times m$ matrix $X=[x^{(1)}\ x^{(2)}\ \cdots\ x^{(m)}]$, and<br>
$w$ as an $n_x\times 1$ column vector $w=[w_1\ w_2\ \cdots \ w_n]^T$, and <br>
$b$ as an $1\times m$ row vector $b=[b\ b\ \cdots \ b]$<br>

so we make <br>
$Z=[z^{(1)}\ z^{(2)}\ \cdots\ z^{(m)}]=w^TX+b=np.dot(w.T, X)+b$<br>
$b$ here will be broadcasted by python, so it won't trouble if it is a real number actually.

After getting $Z$ calculated by just one line code, we need to calculate $A=[a^{(1)}\ a^{(2)}\ \cdots\ a^{(m)}]=\sigma(Z)$ next.

##week 3
In general, if a neural network has one hidder layer and one output layer, we call it a **two layer** neural network<br>
![avatar](https://github.com/kinglin/NotesOfDL/raw/master/pics/nn_dl_5.png)















