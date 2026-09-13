---
title: Math Typesetting
date: 2026-08-17
description: A brief guide to setup KaTeX
math: true
---

Mathematical notation in a Hugo project via Hugo's embedded instance of the KaTeX.
<!--more-->

In this example we will be using [KaTeX](https://katex.org/)

**Note:** Use the online reference of [Supported TeX Functions](https://katex.org/docs/supported)

And the general: [LaTeX/Mathematics](https://en.wikibooks.org/wiki/LaTeX/Mathematics)


### Examples

Inline math: $\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887...$

Block math:


---

$$
\begin{aligned}
KL(\hat{y} || y) &= \sum_{c=1}^{M}\hat{y}_c \log{\frac{\hat{y}_c}{y_c}} \\
JS(\hat{y} || y) &= \frac{1}{2}(KL(y||\frac{y+\hat{y}}{2}) + KL(\hat{y}||\frac{y+\hat{y}}{2}))
\end{aligned}
$$

---

$$
\sum_{i=1}^{n}{i}=\frac{n(n+1)}{2}
$$

$$
\sum_{\substack{
   0<i<m \\
   0<j<n
  }}
 P(i,j)
$$

$$
\begin{equation}
\prod_{j\geq 0}
\left(\sum_{k\geq 0}a_{jk} z^k\right)
= \sum_{k\geq 0} z^n
\left( \sum_{{k_0,k_1,\ldots\geq 0}
\atop{k_0+k_1+\ldots=n} }
a{_0k_0}a_{1k_1}\ldots \right)
\end{equation}
$$

---

$$
\forall x \in X, \quad \exists y \leq \epsilon
$$

---

$$
\alpha, \Alpha, \beta, \Beta, \gamma, \Gamma, \pi, \Pi, \phi, \varphi, \mu, \Phi
$$

$$
( a ), [ b ], \{ c \}, | d |, \| e \|,
\langle f \rangle, \lfloor g \rfloor,
\lceil h \rceil, \ulcorner i \urcorner,
/ j \backslash
$$

$$
\overrightarrow{\text{cool}} \,{\Huge \Re }\;\overleftarrow{\text{cool}}
$$

$$
\boldsymbol{\beta} = (\beta_1,\beta_2,\dotsc,\beta_n)
$$

---

$$
\cos (2\theta) = \cos^2 \theta - \sin^2 \theta
$$

---

$$
\lim\limits_{x \to \infty} \exp(-x) = 0
$$

---

$$
\begin{aligned}
&a \bmod b \\
&x \equiv a \pmod{b}
\end{aligned}
$$

---

$$
k_{n+1} = n^2 + k_n^2 - k_{n-1}
$$

$$
f(n) = n^5 + 4n^2 + 2 |_{n=17}
$$

---

$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$

$$
\frac{\frac{1}{x}+\frac{1}{y}}{y-z}
$$

$$
\left(\frac{x^2}{y^3}\right)
$$


$$
\begin{equation}
  x = a_0 + \cfrac{1}{a_1
          + \cfrac{1}{a_2
          + \cfrac{1}{a_3 + \cfrac{1}{a_4} } } }
\end{equation}
$$

---

$$
\sqrt{\frac{a}{b}}
$$

$$
\sqrt[n]{1+x+x^2+x^3+\dots+x^n}
$$

---

$$
\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x
$$

$$
\int\limits_a^b
$$

$$
\begin{equation}
\phi(t)=\frac{1}{\sqrt{2\pi}}
\int^t_0 e^{-x^2/2} \mathrm{d}x
\end{equation}
$$

---

$$
\begin{matrix}
a & b & c \\
d & e & f \\
g & h & i
\end{matrix}
$$

$$
\begin{matrix}
-1 & 3 \\
2 & -4
\end{matrix}=
\begin{matrix*}[r]
-1 & 3 \\
2 & -4
\end{matrix*}
$$

$$
A_{m,n} =
\begin{pmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
\vdots  & \vdots  & \ddots & \vdots  \\
a_{m,1} & a_{m,2} & \cdots & a_{m,n}
\end{pmatrix}
$$

$$
\begin{array}{c|c}
1 & 2 \\
\hline
3 & 4
\end{array}
$$

$$
M = \begin{bmatrix}
\frac{5}{6} & \frac{1}{6} & 0           \\[0.3em]
\frac{5}{6} & 0           & \frac{1}{6} \\[0.3em]
0           & \frac{5}{6} & \frac{1}{6}
\end{bmatrix}
$$


A matrix in text must be set smaller:
$\bigl(\begin{smallmatrix} a&b \\ c&d \end{smallmatrix} \bigr)$
to not increase leading in a portion of text.

$$
\begin{equation}
{F}(x,y)=0\quad\mathrm{and}\quad
\left|\begin{array}{ccc}
F_{\!xx}'' & F_{\!xy}'' & F_{\!x}' \\[0.3em]
F_{\!yx}'' & F_{\!yy}'' & F_{\!y}' \\[0.3em]
F_{\!x}' & F_{\!y}' & 0
\end{array} \right| =0
\end{equation}
$$

---



$$
k = {\color{red}x} \mathbin{\color{blue}-} 2
$$

---


$$
( \big( \Big( \bigg( \Bigg(
$$

$$
\begin{aligned}
&\frac{\mathrm d}{\mathrm d x} \left( k g(x) \right) \\
&\frac{\mathrm d}{\mathrm d x} \big( k g(x) \big) \\
&\frac{\mathrm d}{\mathrm d x} \Big( k g(x) \Big)
\end{aligned}
$$

$$
\left\{\frac{x^2}{y^3}\right\}
$$

$$
P\left(A=2\middle|\frac{A^2}{B}>4\right)
$$

$$
\left.\frac{x^3}{3}\right|_0^1
$$


---


$$
f(n) =
\begin{cases}
	n/2       & \quad \text{if } n \text{ is even}\\
	-(n+1)/2  & \quad \text{if } n \text{ is odd}
\end{cases}
$$

$$
\begin{equation}
\{\underbrace{%
\overbrace{\mathstrut a,\ldots,a}^{k\ a's},
\overbrace{\mathstrut b,\ldots,b}^{l\ b's}}
_{k+1\ \mathrm{elements}} \}
\end{equation}
$$

---

$$
50 apples \times 100 apples = lots of apples^2
$$

$$
50 \text{apples} \times 100 \text{apples} = \text{lots of apples}^2
$$

$$
50 \text{ apples} \times 100 \text{ apples} = \text{lots of apples}^2
$$


---

$$
\int y \mathrm{d}x
$$


$$
\int y\, \mathrm{d}x
$$

$$
\int y\: \mathrm{d}x
$$

$$
\int y\; \mathrm{d}x
$$

$$
\int y\! \mathrm{d}x
$$

$$
\left(
\begin{array}{c}
	n \\
	r
\end{array}
\right)
= \frac{n!}{r!(n-r)!}
$$

$$
\left(\!
\begin{array}{c}
	n \\
	r
\end{array}
\!\right)
= \frac{n!}{r!(n-r)!}
$$
