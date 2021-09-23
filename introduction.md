---
marp: true
---

# What is machine learning ?
---
# Does every observable natural phenomenon follow a law?
---
# Math language 
Law as a function 
## How would we *know* that function?
---
# How would we *know* that function?
![Weierstrass ](https://upload.wikimedia.org/wikipedia/commons/f/f1/Karl_Weierstrass.jpg)

---
# Origin of approximation theory 
## Weierstrass Theorem  (1885)

Given $f:[a,b] \rightarrow \mathbb R$ continuous and an arbitrary $\varepsilon > 0$, there exists an algebraic polynomial $p$ such that 
$$ |f(x) - p(x)| \leq \varepsilon, \quad \forall x \in [a,b].$$  

[Stone WeierstrassTheorem ( Weierstrass 1887, simplified proof Stone 1948)](https://en.wikipedia.org/wiki/Stone%E2%80%93Weierstrass_theorem)

---
# Formalizing the approach
- Data come from $\mathcal P (\mathcal D)$. 
- Condition: Independent identically distributed iid
- The information is the characteristic vector $\mathcal X$

# Prediction task
$$f: \mathcal X \rightarrow \mathcal Y$$

# Model 
- Class of functions where we are going to search
- Need a criteria: loss function and algorithm
---
# Lineal regression 
add photo
$$E_{in}(W) = \frac{1}{N} \sum_ {n=1}^N (w^T x_n - y_n)^2$$
---
# Gradient Descendent 

Learning rate  

---

# Perceptron

---
# Neuronal Network  

Is this a polynomial: more or less :) 
