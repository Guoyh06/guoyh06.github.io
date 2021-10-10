# Hall定理

### 定理叙述

给定一个二分图，记$L$，$R$分别是左右两边的点集，对于所有$S\subseteq L$，令$N(S)$表示右边**直接**连到$S$的点

最大匹配数为$|L|$​当且仅当对于每个$S\subseteq L$​有$|N(S)|\geq |S|$​

### 推论
$$
\text{二分图最大匹配数}=|L|-\max\limits_{S\subseteq L}\{|S|-|N(S)|\}
$$

