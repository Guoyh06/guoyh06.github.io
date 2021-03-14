# 2-sat 问题

## 问题形式

有$n$个集合，其中每个集合都有两个元素$\{a,\lnot a\}$

有$m$个形如$a\to b$的命题，表示若取$a$，则必须取$b$

求一种方案，在每个集合中仅取一个元素，使得$m$个命题均为真

## 解决方法

### Tarjan求SCC

对于每个命题$a\to b$，在图上连上$a\to b$和$(\lnot b)\to(\lnot a)$两条边，分别对应原命题和它的逆否命题

对于每个点，从它开始能够遍历到的点，都是它的必要条件;能够遍历到它的点，都是它的充分条件

与这个点在同一个SCC中的点都和它互为充要条件，完全等价

如果存在$a$使得$a\leftrightarrow(\lnot a)$，推出矛盾，无解

下面考虑如何构造一组可行解

对于每个点$u$，设其所属SCC编号为$SCC_u$

$\forall a\to b$

- 若为树边、前向边或横叉边，$b$所在的SCC已经被记录，或和$a$在同一SCC

- 若为返祖边，$a$和$b$在同一SCC

    所以$SCC_a\geq SCC_b$

对于每个集合，取其中SCC编号更小的元素即可

若$SCC_v > SCC_{\lnot v}$

$\forall u\to v,(\lnot v)\to(\lnot u)，SCC_u\geq SCC_v>SCC_{\lnot v}\geq SCC_{\lnot u}$

则$u$不可能被取到，这样取一定满足要求

### 扩展域并查集

若对于每个真命题，其逆命题总是为真，建出的图一定是一个无向图

可以用并查集动态维护每个点所在的强连通分量，判断是否会出现矛盾

每当新添加一个命题$a\leftrightarrow b$，将$a,b$所在集合合并，$\lnot a,\lnot b$所在集合合并

若$\exists u\leftrightarrow\lnot u$，一定有$u\leftrightarrow a$或$u\leftrightarrow\lnot a$，则$\lnot u\leftrightarrow\lnot a$或$\lnot u\leftrightarrow a$，$a\leftrightarrow\lnot a$

只需检查$a\leftrightarrow\lnot a$是否为真即可，若为真，则矛盾
