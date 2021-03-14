# 网络流

## 定义

**网络流图**:带权有向图$G=(V,E)$满足以下条件:

- 有且仅有一个$s\in V$入度为0，称为**源点**
- 有且仅有一个$t\in V$出度为0，称为**汇点**

**弧**:网络流图中的一条带权边$(u,v)\in E$

**容量**:$\forall (u,v)\in E\quad \exists c(u,v)\in R^+$，若$(u,v)\notin E$，$c(u,v)=0$

**流**:通过网络流图中一条弧$(u,v)$的流记作$f(u,v)$，满足$f(u,v)=-f(v,u)$

**网络流**:网络流图中所有流的集合$F=\{f(u,v)|u\in V,v\in V\}$

**可行流**:满足以下条件的网络流:
- **容量限制**:$\forall u\in V,v\in V\quad f(u,v)\leq c(u,v)$
- **流量守恒**:$\forall u\in V,u\neq s,u\neq t\quad \sum_{v\in V} f(u,v)=0$

**剩余容量**:$c_f(u,v)=c(u,v)-f(u,v)$，注意$c_f(u,v)=c(u,v)+f(v,u)$，即使$c(u,v)=0$也有可能出现$c_f(u,v)>0$

**残量网络**:$G_f=(V,E_f)$，由剩余容量构成的网络流图

**最大流**:网络流图中的一个可行流，满足$\sum_{v\in V} f(s,v)$最大

**增广路径**:一条路径$(u_1,u_2,\cdots,u_k)$，$u_1=s$，$u_k=t$，$\forall 1\leq i<k\quad c(u_i,u_{i+1})>0$

## 最大流

### 割

对于一个网络流图$G(V,E)$，割定义为一种**点的划分方式**，将所有点分成两个集合$S$和$T=V\setminus S$，其中源点$s\in S$，汇点$t\in T$

##### 割的容量

定义割$(S,T)$的容量$c(S,T)$为所有从$S$到$T$的边的容量之和，$c(S,T)=\sum_{u\in S,v\in V} c(u,v)$

##### 割的流量

与割的容量类似$f(S,T)=\sum_{u\in S,v\in T}f(u,v)$

### 增广正确性证明

**命题1:对于可行流的任意一个割，割的流量等于总流量**

设总流量为$f_t$，由流量守恒$f_t-f(S,T)=0$，$f(S,T)-f_t=0$

得出$f(S,T)=f_t$

**命题2:可行流的流量一定小于等于任意一个割的容量**

由命题1，$f_t=f(S,T)\leq c(S,T)$

**命题3:以下3个命题等价**

1. 存在一个割$(S,T)$使得$c(S,T)=f(S,T)$

2. $f_t$是网络流图的最大流

3. 残余网路中不存在增广路

$1\to 2$:由命题2，$f_t\leq c(S,T)$，当$f_t=f(S,T)=c(S,T)$时，$f_t$为最大流

$2\to 3$:若存在增广路，可以达到更大的流量，$f_t$不是最大流，矛盾

$3\to 1$:残余网路中没有增广路，取所有可以从$s$通过残余网路中$c_f(u,v)>0$的边遍历到的点为$S$，剩余点为$T$，则不存在$u\in S,v\in T$，使得$f(u,v)\neq c(u,v)$，$c(S,T)=f(S,T)$

$2\to 1$即为**最大流最小割定理**:一个网络流图的最大流等于它的最小割

$3\to 2$即可证明当不存在增广路时当前流量为最大流

### EK算法

不断在残余网络上通过BFS找到一条增广路经，然后对其进行增广，生成新的残余网络

**增广过程**:沿着增广路经$(u_1,u_2,\cdots,u_k)$遍历，记其中容量最小的边容量为$minf$，$\forall 1\leq i<k\quad f(u_i,u_{i+1}):=f(u_i,u_{i+1})+minf$,并且更新残余网络$c_f(u_i,u_{i+1}):=c_f(u_i,u_{i+1})-f(u_i,u_{i+1})$，$c_f(u_{i+1},u_i):=c_f(u_{i+1},u_i)+f(u_i,u_{i+1})$

**实现细节**:用链式前向星存图时可以从2开始存，边$i$的反向边就是$i\operatorname{xor}1$

**复杂度**:$O(VE^2)$

### Dinic 算法

通过BFS将图分层，DFS进行增广，遍历到每个点时，只找比当前点层数大1的点增广

通过这样操作，保证了每次都是沿着最短路进行增广

#### 优化

- **当前弧优化(必须加)**:

    在一轮DFS中，每条边只被增广一次

    记录每个点最后遍历到的边，下一次遍历到这个点时直接从下一条边开始增广

    不加当前弧优化可能常数更小，但是时间复杂度**是错的**！

- 若遍历到的下一个点无法继续增广，则标记这个点，下次再访问到直接跳出

#### 时间复杂度

在一般图上的时间复杂度为$O(V^2E)$，在二分图上的时间复杂度为$O(E\sqrt{V})$，~~在比较规则的图上跑得都很快

#### 代码

```cpp
bool bfs(int S, int T){
	memset(dep, -1, sizeof(dep));
	queue <int> q;
	q.push(S);
	dep[S] = 0;
	while (!q.empty()){
		int nw = q.front();
		q.pop();
		for (int i = fte[nw]; i; i = g[i].nxt){
			int nxtn = g[i].t;
			if (dep[nxtn] != -1) continue;
			if (g[i].c == 0) continue;
			dep[nxtn] = dep[nw] + 1;
			q.push(nxtn);
		}
	}
	return dep[T] != -1;
}
int dfs(int nw, int mxf, int T){
    if (mxf == 0) return 0;
	if (nw == T) return mxf;
	int df = 0;
	for (int &i = curE[nw]; i; i = g[i].nxt){ //当前弧优化
		int nxtn = g[i].t;
		if (g[i].c == 0 || dep[nxtn] != dep[nw] + 1) continue;
		int nwf = dfs(nxtn, min(mxf - df, g[i].c), T);
        if (!nwf) dep[nxtn] = -1; //优化2
		g[i].c -= nwf;
		g[i ^ 1].c += nwf;
		df += nwf;
		if (df == mxf) break;
	}
	return df;
}
int dinic(int S, int T){
	int ans = 0;
	while (bfs(S, T)){
		memcpy(curE, fte, sizeof(curE));
		ans += dfs(S, 2e9, T);
	}
	return ans;
}
```

### capacity scaling

这是算法导论的思考题26-5给出的算法

从大到小枚举$K\in\{x|x=2^i,i\in\mathbb{Z}\}$，每次做一遍Dinic，但只处理所有容量$\geq K$的边

伪代码:

```c++
MAX-FLOW-BY-SCALING(G, s, t)
    C = max_{(u, v) ∈ E} c(u, v)
    initialize flow f to 0
    K = 2^{floor(lg C)}
    while K ≥ 1
        while there exists an augmenting path p of capacity at least K
            augment flow f along p
        K = K / 2
    return f
```

复杂度可以达到$O(E^2\log C)$

**注意**：即使没有加入容量$<K$的边，残余网络上仍然可能存在容量$<K$的边，不能对其进行处理

#### 证明[^1]

**最外层的循环**：只能执行$O(\log C)$次

**中间的循环**：

- 经过上次处理后，每条增广路的容量都$\leq 2K$，整个图的最小割最大为$2KE$
- 因为只处理容量$\geq K$的边，每次增广都会增加$K$的流量，总共最多会增广$\frac{2KE}{K}=2E$次
- 总共只会循环$O(E)$次

**寻找增广路&增广**：用Dinic算法$O(E)$增广一次

#### 代码

```cpp
# include <queue>
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int MAXV = 205;
const int MAXE = 5051;
struct Edge{
	int t;
	ll c;
	int nxt;
	Edge(int t = 0, ll c = 0, int nxt = 0): t(t), c(c), nxt(nxt){}
} g[MAXE * 2];
struct Edge2{
	int u, v;
	ll c;
} e[MAXE];
int n, m, gsz = 1;
int fte[MAXV], curE[MAXV];
int dep[MAXV];
bool cmp(Edge2 x, Edge2 y){
	return x.c > y.c;
}
void addedge(int u, int v, ll c){
	g[++gsz] = Edge(v, c, fte[u]);
	fte[u] = gsz;
	g[++gsz] = Edge(u, 0, fte[v]);
	fte[v] = gsz;
}
bool bfs(int S, int T, ll lim){
	queue <int> q;
	memset(dep, -1, sizeof(dep));
	dep[S] = 0;
	q.push(S);
	while (!q.empty()){
		int nw = q.front();
		q.pop();
		for (int i = fte[nw]; i; i = g[i].nxt){
			int nxtn = g[i].t;
			if (g[i].c < lim || dep[nxtn] != -1) continue;
			dep[nxtn] = dep[nw] + 1;
			q.push(nxtn);
		}
	}
	return dep[T] != -1;
}
ll dfs(int nw, ll mxf, int T, ll lim){
	if (nw == T) return  mxf;
	if (mxf == 0) return 0;
	ll df = 0;
	for (int &i = curE[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (g[i].c < lim || dep[nxtn] != dep[nw] + 1) continue;
		ll nwf = dfs(nxtn, min(mxf - df, g[i].c), T, lim);
		g[i].c -= nwf;
		g[i ^ 1].c += nwf;
		df += nwf;
		if (df == mxf) break;
	}
	return df;
}
ll dinic(int S, int T, ll lim){
	ll ans = 0;
	while (bfs(S, T, lim)){
		memcpy(curE, fte, sizeof(fte));
		ans += dfs(S, 1e18, T, lim);
	}
	return ans;
}
int main(){
	int S, T;
	scanf("%d%d%d%d", &n, &m, &S, &T);
	for (int i = 1; i <= m; i++) scanf("%d%d%lld", &e[i].u, &e[i].v, &e[i].c);
	sort(e + 1, e + m + 1, cmp);
	int pos = 1;
	ll ans = 0;
	for (int i = 30; i >= 0; i--){
		while (pos <= m && e[pos].c >= (1 << i)){
			addedge(e[pos].u, e[pos].v, e[pos].c);
			pos++;
		}
		ans += dinic(S, T, (1 << i));
	}
	printf("%lld\n", ans);
	return 0;
}
```



## 费用流

 给定一个网络$G=(V,E)$，每条边除了有容量限制$c(u,v)$外，还有一个费用$w(u,v)$，并且满足$w(u,v)=-w(v,u)$

当边$(u,v)$流量为$f(u,v)$时，需要花费$f(u,v)\cdot w(u,v)$的费用

**最小费用最大流**:在最大化总流量的情况下最小化总费用，在最大化$\sum_{(s,v)\in E}f(s,v)$时最小化$\sum_{(u,v)\in E}f(u,v)\cdot w(u,v)$

### SSP算法(类Dinic)

每次用SPFA寻找单位流量花费最小的增广路，并且用类Dinic进行增广

只增广在最短路图(DAG)上的边

对每个点维护$vis$数组，如果DFS到环直接推出

不能用来处理包含负圈的图

##### 时间复杂度

时间复杂度为$O(nmf)$，当流量变大时不太优秀，但是在一般情况下常数很好

##### 代码

```cpp
bool spfa(int S, int T){
	memset(inq, false, sizeof(inq));
	memset(dis, 0x3f, sizeof(dis));
	while (!q.empty()) q.pop();
	dis[S] = 0;
	q.push(S);
	while (!q.empty()){
		int nw = q.front();
		q.pop();
		inq[nw] = false;
		for (int i = fte[nw]; i; i = g[i].nxt){
			int nxtn = g[i].t;
			if (dis[nxtn] <= dis[nw] + g[i].w || g[i].c == 0) continue;
			dis[nxtn] = dis[nw] + g[i].w;
			if (!inq[nxtn]) q.push(nxtn);
			inq[nxtn] = true;
		}
	}
	return dis[T] < 1e18;
}
ll dfs(int nw, ll mxf, int T){
	if (nw == T) return mxf;
	if (mxf == 0) return 0;
	vis[nw] = true;
	ll df = 0;
	for (int &i = curE[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (dis[nxtn] != dis[nw] + g[i].w || g[i].c == 0 || vis[nxtn]) continue;
		ll nwf = dfs(nxtn, min(mxf - df, g[i].c), T);
		g[i].c -= nwf;
		g[i ^ 1].c += nwf;
		df += nwf;
		if (df == mxf) break;
	}
	vis[nw] = false;
	return df;
}
ll dinic(int S, int T){
	ll ans = 0;
	while (spfa(S, T)){
        memcpy(curE, fte, sizeof(fte));
		ans += dis[T] * dfs(S, 1e18, T);
	}
	return ans;
}
```



## 有上下界网络流

网路图$G=(V,E)$中的每一条边$(u,v)$都有两个限制$f_{low}(u,v)$，$f_{up}(u,v)$，分别表示这条边流量的下界和上界

### 无源汇有上下界可行流

首先满足所有下界，假设所有边都流过了大小为$f_{low}(u,v)$的流量

**注意**:此部分流量**不可撤回**，直接令$c(u,v)=f_{up}(u,v)-f_{min}(u,v)$即可

记净流入点$u$的流量为$inf_u$，若为净流出则$inf_u<0$

新建虚拟源点$s^\prime$和$t^\prime$

对每个点$u$:

- $inf_u=0$，不作任何操作
- $inf_u>0$，从$s^\prime$连向$u$一条容量为$inf_u$的边
- $inf_u<0$，从$u$连向$t^\prime$一条容量为$-inf_u$的边

用Dinic做从$s^\prime$到$t^\prime$的最大流

若$\exists u\in V\quad f(s,u)\neq c(s,u)$，则没有办法满足，无解

```cpp
# include <queue>
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int MAXV = 205;
const int MAXE = 10251 + MAXV;
struct Edge{
	int t, c, nxt;
	Edge(){}
	Edge(int t, int c, int nxt): t(t), c(c), nxt(nxt){}
} g[MAXE * 2];
int n, m, gsz = 1;
int low[MAXE];
int inf[MAXV];
int dep[MAXV];
int fte[MAXV], curE[MAXV];
void addedge(int u, int v, int c){
	g[++gsz] = Edge(v, c, fte[u]);
	fte[u] = gsz;
	g[++gsz] = Edge(u, 0, fte[v]);
	fte[v] = gsz;
}
void addedge1(int u, int v, int mif, int mxf){
	inf[v] += mif;
	inf[u] -= mif;
	addedge(u, v, mxf - mif);
}
bool bfs(int S, int T){
	queue <int> q;
	memset(dep, -1, sizeof(dep));
	dep[S] = 0;
	q.push(S);
	while (!q.empty()){
		int nw = q.front();
		q.pop();
		for (int i = fte[nw]; i; i = g[i].nxt){
			int nxtn = g[i].t;
			if (g[i].c == 0 || dep[nxtn] != -1) continue;
			dep[nxtn] = dep[nw] + 1;
			q.push(nxtn);
		}
	}
	return dep[T] != -1;
}
int dfs(int nw, int mxf, int T){
	if (mxf == 0) return 0;
	if (nw == T) return mxf;
	int df = 0;
	for (int &i = curE[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (g[i].c == 0 || dep[nxtn] != dep[nw] + 1) continue;
		int nwf = dfs(nxtn, min(mxf - df, g[i].c), T);
		if (nwf == 0) dep[nxtn] = -1;
		g[i].c -= nwf;
		g[i ^ 1].c += nwf;
		df += nwf;
		if (df == mxf) break;
	}
	return df;
}
void dinic(int S, int T){
	while (bfs(S, T)){
		memcpy(curE, fte, sizeof(curE));
		dfs(S, 1e9, T);
	}
}
bool check(int S){
	for (int i = fte[S]; i; i = g[i].nxt) if (g[i].c != 0) return false;
	return true;
}
int main(){
	scanf("%d%d", &n, &m);
	int S = 0, T = n + 1;
	for (int i = 1; i <= m; i++){
		int u, v, mif, mxf;
		scanf("%d%d%d%d", &u, &v, &mif, &mxf);
		addedge1(u, v, mif, mxf);
		low[i] = mif;
	}
	for (int i = 1; i <= n; i++){
		if (inf[i] > 0) addedge(S, i, inf[i]);
		else addedge(i, T, -inf[i]);
	}
	dinic(S, T);
	if (!check(S)) printf("NO\n");
	else {
		printf("YES\n");
		for (int i = 1; i <= m; i++) printf("%d\n", low[i] + g[i * 2 + 1].c);
	}
	return 0;
}
```



### 有源汇有上下界可行流

连一条$s\to t$的边，然后按照无源汇的做就行了

### 有源汇有上下界最大流

先按可行流做，最后做一遍从$s$到$t$的最大流

因为每条边的流量已经减去下界，而容量为上下界之差，所以不管如何增广，每条边的流量总是满足上下界

用可行流加上最大流

### 有源汇有上下界最小流

先按照可行流做，最后做一遍从$t$到$s$的最大流

相当于把多余的流量全部退流

用可行流减去最大流

### 有源汇有上下界费用流

在处理下界时记录费用，只要把所有最大流换成最小费用最大流就可以了

## 一些技巧

- 若要限制单个点的容量，将每个点都拆成两个不同的点

    所有入边连到第一个点，所有出边从第二个点连出

    两个点之间连一条边，这条边容量就是点的容量

[^1]:参考<https://walkccc.me/CLRS/Chap26/Problems/26-5/>

