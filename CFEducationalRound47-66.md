# Educational Codeforces Round 47（1009）

## A

用双指针$i$、$j$分别指向当前两个数组目前的位置。如果$c_i\leq a_j$，就买得起，两个指针分别变化1，否则买不起，只有$i$变化
```cpp
for (int i = 1; i <= n && j <= m; i++){
	if (a[j] >= c[i]) j++;
}
```

最后数一共用了多少支票就行了

## B

题中给出的条件可以等价成串中的每一个1可以自由移动，只要把所有1放到第一个2之前就一定是最小值了

```cpp
for (int i = 1; i <= sz; i++){
	if (i == mk){
		for (int i = 1; i <= n1; i++) printf("1");
	}
	if (s[i] != '1') printf("%c", s[i]);
}
```

## C

每次增加的值为$x_i+d_i\sum |i-j|$

可以预处理出$\sum |i-j|$的最大值和最小值，当$d$为正数时取最大值，当$d$为负数时取最小值

```cpp
ll half = (n + 1) / 2;
ll mxd = n * (n - 1) / 2, mid = (n & 1) ? half * (half - 1) : half * half;
ll ans = 0;
for (int i = 1; i <= m; i++){
	ll x, d;
	scanf("%lld%lld", &x, &d);
	ans += x * n;
	if (d > 0) ans += (d * mxd);
	else ans += (d * mid);
}
printf("%lf\n", (double)ans / n);
```

## D

在数据较大时$m$​远小于$\sum_1^n \phi(i)$​，用程序枚举后发现$n$​取575时$\sum_1^n \phi(i)$​已经大于$10^5$​了

直接枚举两个数再求gcd就行了

从1到每个点都连一条边就能使图联通

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 100051;
int n, m, cnt;
int ans1[MAXN], ans2[MAXN];
int gcd(int u, int v){
	if (v == 0) return u;
	else return gcd(v, u % v);
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++){
		for (int j = i + 1; j <= n; j++){
			if (gcd(i, j) == 1){
				cnt++;
				ans1[cnt] = i;
				ans2[cnt] = j;
				if (cnt >= m) break;
			}
		}
		if (cnt >= m) break;
	}
	if (cnt < m || cnt < n - 1) printf("Impossible\n");
	else {
		printf("Possible\n");
		for (int i = 1; i <= m; i++) printf("%d %d\n", ans1[i], ans2[i]);
	}
	return 0;
}
```



## E

概率型DP，令$f_i$为从$i$开始走（之前在$i$休息过）到终点的代价

转移：枚举下一次休息的地点
$$
f_i=\sum\limits_{j=i+1}^{n}\frac 1{2^{j-i}}(f_j+\sum\limits_{k=1}^{j-i}a_k)+\frac 1{2^{n-i}}\sum\limits_{k=1}^{n-i+1}a_k
$$
预处理$a_i$的前缀和$s_i=\sum\limits_{j=1}^ia_j$，$\frac{f_i}{2^i}$的后缀和$sf_i=\sum\limits_{j=i}^n\frac{f_i}{2^i}$，$\frac{s_j}{2^i}$的前缀和$ss_i=\sum\limits_{j=1}^i\frac{s_j}{2^i}$

转移式变为
$$
f_i=2^isf_{i+1}+ss_{n-i}+\frac1{2^{n-i}}s_{n-i+1}
$$
有点卡常，$2^i$需要递推

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int MAXN = 1000051;
const int P = 998244353;
const int INV2 = 499122177;
int n;
ll a[MAXN], f[MAXN], s[MAXN], ss[MAXN];
ll pwr(ll x, ll y){
	y = (y % (P - 1) + (P - 1)) % (P - 1);
	ll ans = 1;
	while (y){
		if (y & 1) ans = ans * x % P;
		x = x * x % P;
		y >>= 1;
	}
	return ans;
}
int main(){
	scanf("%d", &n);
	ll p0 = 1;
	for (int i = 1; i <= n; i++){
		scanf("%lld", a + i);
		// a[i] = i;
		s[i] = (s[i - 1] + a[i]) % P;
		p0 = p0 * INV2 % P;
		ss[i] = (ss[i - 1] + p0 * s[i]) % P;
	}
	ll sf = 0;
	ll p1 = pwr(2, n), p2 = pwr(2, -n), pn = pwr(2, -n);
	for (int i = n; i >= 1; i--){
		f[i] = p1 * sf % P;
		f[i] = (f[i] + ss[n - i]) % P;
		f[i] = (f[i] + pn * p1 % P * s[n - i + 1]) % P;
		sf = (sf + p2 * f[i]) % P;
		p1 = p1 * INV2 % P;
		p2 = p2 * 2 % P;
	}
	ll ans = f[1] * pwr(2, n - 1) % P;
	printf("%lld\n", (ans + P) % P);
	return 0;
}
```

## F

长链剖分DP，直接求出每个点的d，DP的时候维护一下答案

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int MAXN = 1000051;
struct Edge{
	int t, nxt;
} g[MAXN << 1];
int n, gsz, cnt;
int fte[MAXN];
int dfn[MAXN], sz[MAXN];
int dep[MAXN], hgt[MAXN];
int fa[MAXN], hs[MAXN], top[MAXN];
int f[MAXN], ans[MAXN];
void addedge(int u, int v){
	g[++gsz] = (Edge){v, fte[u]};
	fte[u] = gsz;
}
void dfs0(int nw){
	sz[nw] = 1;
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (nxtn == fa[nw]) continue;
		fa[nxtn] = nw;
		dep[nxtn] = dep[nw] + 1;
		dfs0(nxtn);
		if (hgt[nxtn] + 1 > hgt[nw]){
			hgt[nw] = hgt[nxtn] + 1;
			hs[nw] = nxtn;
		}
		sz[nw] += sz[nxtn];
	}
}
void dfs1(int nw){
	dfn[nw] = ++cnt;
	if (hs[nw]){
		top[hs[nw]] = top[nw];
		dfs1(hs[nw]);
	}
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (nxtn == fa[nw] || nxtn == hs[nw]) continue;
		top[nxtn] = nxtn;
		dfs1(nxtn);
	}
}
void dfs3(int nw){
	if (hs[nw]){
		dfs3(hs[nw]);
		ans[nw] = ans[hs[nw]] + 1;
	}
	f[dfn[nw]] = 1;
	if (f[dfn[nw] + ans[nw]] <= 1) ans[nw] = 0;
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (nxtn == fa[nw] || nxtn == hs[nw]) continue;
		dfs3(nxtn);
		for (int j = 0; j <= hgt[nxtn]; j++){
			f[dfn[nw] + j + 1] += f[dfn[nxtn] + j];
			if (f[dfn[nw] + j + 1] > f[dfn[nw] + ans[nw]]) ans[nw] = j + 1;
			if (f[dfn[nw] + j + 1] == f[dfn[nw] + ans[nw]] && j + 1 < ans[nw]) ans[nw] = j + 1;
		}
	}
}
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n - 1; i++){
		int u, v;
		scanf("%d%d", &u, &v);
		addedge(v, u);
		addedge(u, v);
	}
	dfs0(1);
	dfs1(1);
	dfs3(1);
	for (int i = 1; i <= n; i++) printf("%d\n", ans[i]);
	return 0;
}
```

## G

贪心选择当前字母，重点在如何判断后面是否能继续填下去

本质上是一个匹配问题，判断原来的字符串是否能完美匹配到新的字符串上去

根据Hall定理，任意一个子集匹配点一定要多余子集本身大小

一个字母，全选的情况显然比选一部分劣，因为子集大小增大，匹配集却不变

对于每个字母，只需要枚举全选/不选一共$2^6=64$​​种

总复杂度$O(n\cdot2^{|\Sigma|})$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 100051;
int n, m;
char s[MAXN];
int cnt[6];
int cc[1 << 6];
int lm[MAXN];
int cl[1 << 6];
int lg[1 << 6];
bool check(){
	for (int i = 1; i < (1 << 6); i++){
		if (i == (1 << lg[i])) cc[i] = cnt[lg[i]];
		else cc[i] = cc[i ^ (i & (-i))] + cnt[lg[i & (-i)]];
		// printf("cc %d %d %d\n", i, cl[i], cc[i]);
		if (cl[i] < cc[i]) return false;
	}
	return true;
}
int calc(){
	int s1 = (1 << 6) - 1;
	for (int i = 1; i < (1 << 6); i++){
		if (i == (1 << lg[i])) cc[i] = cnt[lg[i]];
		else cc[i] = cc[i ^ (i & (-i))] + cnt[lg[i & (-i)]];
		if (cl[i] < cc[i]) s1 &= i;
	}
	return s1;
}
int main(){
	lg[1] = 0;
	for (int i = 2; i < (1 << 6); i++) lg[i] = (1 << (lg[i - 1] + 1)) <= i ? lg[i - 1] + 1 : lg[i - 1];
	scanf("%s", s + 1);
	n = strlen(s + 1);
	for (int i = 1; i <= n; i++) cnt[s[i] - 'a']++;
	for (int i = 1; i <= n; i++) lm[i] = (1 << 6) - 1;
	scanf("%d", &m);
	for (int i = 1; i <= m; i++){
		int pos;
		scanf("%d%s", &pos, s + 1);
		int len = strlen(s + 1);
		lm[pos] = 0;
		for (int j = 1; j <= len; j++) lm[pos] |= (1 << (s[j] - 'a'));
	}
	for (int i = 1; i <= n; i++){
		// printf("lm %d %d\n", i, lm[i]);
		for (int s1 = 1; s1 < (1 << 6); s1++){
			if (s1 & lm[i]){
				cl[s1]++;
				// printf("s1 %d\n", s1);
			}
		}
	}
	if (!check()){
		printf("Impossible\n");
		return 0;
	}
	for (int i = 1; i <= n; i++){
		for (int s1 = 1; s1 < (1 << 6); s1++) if (s1 & lm[i]) cl[s1]--;
		int s1 = calc() & lm[i];
		cnt[lg[s1 & (-s1)]]--;
		printf("%c", 'a' + lg[s1 & (-s1)]);
	}
	putchar('\n');
	return 0;
}
```



# Educational Codeforces Round 48（1016）

## A

对于每一天，把这一天要写的名字和这一页已经写的名字求和后除以m就是要翻的页数

```cpp
for (int i = 1; i <= n; i++){
	printf("%d ", (s + a[i]) / m);
	s = (s + a[i]) % m;
}
```

## B

直接用KMP在s中匹配就行了

## C

要把所有格子都走恰好一遍，路径一定是如下图这样，先上下走再来回走

![](https://cdn.luogu.com.cn/upload/image_hosting/t39d2xv2.png)

把所有情况都枚举一下再取最大值

```cpp
for (int i = 1; i <= n; i++){
	if (i & 1) g[i] = g[i - 1] + a[0][i] * (2 * i - 2) + a[1][i] * (2 * i - 1);
	else g[i] = g[i - 1] + a[1][i] * (2 * i - 2) + a[0][i] * (2 * i - 1);
	// printf("%lld ", g[i]);
}
// putchar('\n');
for (int i = 1; i <= n; i++){
	s[0][0][i] = s[0][0][i - 1] + a[0][i];
	s[1][1][i] = s[1][1][i - 1] + a[1][i];
	f[0][0][i] = f[0][0][i - 1] + a[0][i] * (i - 1);
	f[1][1][i] = f[1][1][i - 1] + a[1][i] * (i - 1);
}
s[0][1][n] = s[0][0][n] + a[1][n];
s[1][0][n] = s[1][1][n] + a[0][n];
f[0][1][n] = f[0][0][n] + a[1][n] * n;
f[1][0][n] = f[1][1][n] + a[0][n] * n;
for (int i = n - 1; i >= 1; i--){
	s[0][1][i] = s[0][1][i + 1] + a[1][i];
	s[1][0][i] = s[1][0][i + 1] + a[0][i];
	f[0][1][i] = f[0][1][i + 1] + a[1][i] * (2 * n - i);
	f[1][0][i] = f[1][0][i + 1] + a[0][i] * (2 * n - i);
}
// for (int i = 1; i <= n; i++) printf("%3lld ", f[0][0][i]);
// putchar('\n');
// for (int i = 1; i <= n; i++) printf("%3lld ", f[0][1][i]);
// putchar('\n');
ll lnm = 0;
for (int i = 0; i < n; i++){
	ll nw = g[i] + f[i & 1][!(i & 1)][i + 1] - f[i & 1][i & 1][i] + i * (s[i & 1][!(i & 1)][i + 1] - s[i & 1][i & 1][i]);
	// printf("in %d %lld %lld %lld\n", i, nw, f[i & 1][!(i & 1)][i + 1], f[i & 1][i & 1][i]);
	ans = max(ans, nw);
}
ans = max(ans, g[n]);
```

## D

观察到$\operatorname{Xor}_1^n a_i xor\operatorname{Xor}_1^m b_i = 0$

如果不满足上述条件则不存在对应矩阵

在填矩阵的时候，可以假装某一个限制不存在，这里假装$a_n$不存在

因为如果满足其他限制，根据上式可以推出最后一行的异或和一定就是给出的$a_n$

下面构造了一种比较简单的填法

$$
\begin{bmatrix}
0     &0     &\cdots&0      &a_1                   & \\ 
0     &\ddots&      &0      &a_2                   & \\ 
\vdots&      &      &\vdots &\vdots                & \\ 
      &      &      &0      &a_{n-1}               & \\ 
b_1   &b_2   &\cdots&b_{m-1}&Xor_1^{n-1}a_i xor b_m& 
\end{bmatrix}
$$

易得$\operatorname{Xor}_1^{n-1}a_ixor\operatorname{Xor}_1^mb_i=a_n$

```cpp
for (int i = 1; i < n; i++){
	for (int j = 1; j < m; j++){
		printf("0 ");
	}
	printf("%d\n", a[i]);
}
for (int i = 1; i < m; i++) printf("%d ", b[i]);
printf("%d\n", s1 ^ a[n] ^ b[m]);
```

## E

几何题

如图，对于中间的线段DE，一定是从F到G的时候C都在阴影中

二分求中间线段的范围，边上的线段特判一下

![](https://cdn.luogu.com.cn/upload/image_hosting/sbzfsego.png)

~~建议考虑光的衍射~~

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 200051;
const double INF = 2e9;
struct Node{
	double l, r;
	Node(double l = 0, double r = 0): l(l), r(r){}
} l[MAXN];
int n, q;
int sy, a, b;
ll s[MAXN];
bool cmpl(Node u, Node v){
	return u.l < v.l;
}
bool cmpr(Node u, Node v){
	return u.r < v.r;
}
int main(){
	scanf("%d%d%d", &sy, &a, &b);
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		int u, v;
		scanf("%d%d", &u, &v);
		l[i] = Node(u, v);
		s[i] = s[i - 1] + (v - u);
	}
	l[n + 1] = Node(INF, INF);
	scanf("%d", &q);
	while (q--){
		int u, v;
		scanf("%d%d", &u, &v);
		double lx = (1.0 * v * a - 1.0 * sy * u) / (v - sy);
		double rx = (1.0 * v * b - 1.0 * sy * u) / (v - sy);
		int lp = lower_bound(l + 1, l + n + 1, Node(lx, lx), cmpl) - l - 1;
		int rp = upper_bound(l + 1, l + n + 1, Node(rx, rx), cmpr) - l;
		double ans = (lp == rp) ? (rx - lx) : (max(0.0, l[lp].r - lx) + max(0.0, rx - l[rp].l) + s[rp - 1] - s[lp]);
		ans = ans * (v - sy) / v;
		printf("%.10f\n", ans);
	}
	return 0;
}
```



## F

先预处理$1$到每个点的距离和$n$到每个点的距离

按照到$1$的距离从大到小遍历，用线段树查询已遍历的点（且与当前节点没有连边）中到$n$的最大距离

查询区间个数为边数，均摊复杂度$O(m\log n)$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 300051;
const ll INF = 0x3f3f3f3f3f3f3f3f;
struct Edge{
	int t;
	ll w;
	Edge(int t = 0, ll w = 0): t(t), w(w){}
};
vector <Edge> g[MAXN];
int n, m;
struct SegTree{
	struct Node{
		ll s;
		int ls, rs;
	} t[MAXN * 4];
	int rt, tsz;
	void update(int nw){
		t[nw].s = max(t[t[nw].ls].s, t[t[nw].rs].s);
	}
	void modify(int &nw, int lft, int rgt, int pos, ll nm){
		if (!nw) nw = ++tsz;
		if (lft == rgt){
			t[nw].s = nm;
			return;
		}
		int mid = (lft + rgt) >> 1;
		if (pos <= mid) modify(t[nw].ls, lft, mid, pos, nm);
		else modify(t[nw].rs, mid + 1, rgt, pos, nm);
		update(nw);
	}
	ll getsum(int nw, int lft, int rgt, int l, int r){
		if (l > r) return -INF;
		if (!nw) return -INF;
		if (lft == l && rgt == r) return t[nw].s;
		int mid = (lft + rgt) >> 1;
		ll ans = -INF;
		if (l <= mid) ans = max(ans, getsum(t[nw].ls, lft, mid, l, min(mid, r)));
		if (r >= mid + 1) ans = max(ans, getsum(t[nw].rs, mid + 1, rgt, max(mid + 1, l), r));
		return ans;
	}
} sgt;
int a[MAXN], rk[MAXN];
ll dis[MAXN], dis1[MAXN], disn[MAXN];
bool cmp1(int u, int v){
	return dis1[u] < dis1[v];
}
bool cmp1e(Edge u, Edge v){
	return dis1[u.t] < dis1[v.t];
}
bool cmpn(int u, int v){
	return disn[u] < disn[v];
}
void dfs(int nw, int fa = -1){
	int sz = g[nw].size();
	for (int i = 0; i < sz; i++){
		int nxtn = g[nw][i].t;
		if (nxtn == fa) continue;
		dis[nxtn] = dis[nw] + g[nw][i].w;
		dfs(nxtn, nw);
	}
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n - 1; i++){
		int u, v;
		ll w;
		scanf("%d%d%lld", &u, &v, &w);
		g[u].push_back(Edge(v, w));
		g[v].push_back(Edge(u, w));
	}
	dis[1] = 0;
	dfs(1);
	memcpy(dis1, dis, sizeof(dis));
	dis[n] = 0;
	dfs(n);
	memcpy(disn, dis, sizeof(dis));
	for (int i = 1; i <= n; i++) a[i] = i;
	sort(a + 1, a + n + 1, cmp1);
	for (int i = 1; i <= n; i++) rk[a[i]] = i;
	ll mxw = 0;
	for (int i = n; i >= 1; i--){
		sort(g[a[i]].begin(), g[a[i]].end(), cmp1e);
		int sz = g[a[i]].size();
		int lst = n;
		for (int j = sz - 1; j >= 0; j--){
			mxw = max(mxw, dis1[a[i]] + sgt.getsum(sgt.rt, 1, n, rk[g[a[i]][j].t] + 1, lst));
			if (rk[g[a[i]][j].t] < i) break;
			lst = rk[g[a[i]][j].t] - 1;
		}
		// printf("ai %d %lld\n", a[i], mxw);
		sgt.modify(sgt.rt, 1, n, i, disn[a[i]]);
	}
	for (int i = 1; i <= m; i++){
		int u;
		scanf("%d", &u);
		printf("%lld\n", min(dis1[n], mxw + u));
	}
	return 0;
}
```



# Educational Codeforces Round 49（1027）

## A

如果对应的字母相同，或者如相差正好为2就可以构造出回文串

```cpp
bool check(){
	for (int i = 1; i + i <= n; i++){
		int df = max(s[i], s[n - i + 1]) - min(s[i], s[n - i + 1]);
		if (df != 0 && df != 2) return false;
	}
	return true;
}
```

## B

分类讨论两坐标之和是否为偶数，如果是偶数就从1开始，不是就从$\left\lceil\frac{n^2}{2}\right\rceil+1$开始

```cpp
ll nw = u * n + v;
ll ans;
if ((u + v) & 1) ans = nw / 2 + ((n * n + 1) / 2) + 1;
else ans = nw / 2 + 1;
```

## C

在长方形的形状更接近与正方形时，$\frac{P^2}{S}$更小

对于一种长度的边，只要找比它小且最接近它的可行边就行了

```cpp
sort(a + 1, a + n + 1);
int sz = unique(a + 1, a + n + 1) - a - 1;
int l = -1;
for (int i = 1; i <= sz; i++){
	if (bx[a[i]] >= 4){
		l = a[i];
		ans1 = 16;
		ans2 = 1;
		ans3 = ans4 = a[i];
	} else if (bx[a[i]] >= 2){
		if (l != -1){
			// printf("li %d %d\n", l, a[i]);
			ll p = 2 * (a[i] + l);
			ll s = a[i] * l;
			if (p * p * ans2 < ans1 * s){
				ans1 = p * p;
				ans2 = s;
				ans3 = l;
				ans4 = a[i];
			}
		}
		l = a[i];
	}
	bx[a[i]] = 0;
}
```

## D

老鼠的所有行走路径会组成由若干基环树组成的基环森林

对于每一棵基环树，至少要在环中放一个陷阱，来保证从整棵树中任一点出发的老鼠都能被捉到

找出每个环，再取环中布置陷阱消耗的最小值

```cpp
int dfs(int nw){
	vis[nw] = true;
	st[++ssz] = nw;
	if (rid[a[nw]]) return rid[nw] = rid[a[nw]];
	if (vis[a[nw]]){
		rid[nw] = ++rsz;
		ans[rsz] = 1e9;
		while (ssz > 0 && st[ssz] != a[nw]){
			ans[rsz] = min(ans[rsz], c[st[ssz]]);
			ssz--;
		}
		ans[rsz] = min(ans[rsz], c[a[nw]]);
		return rsz;
	}
	return rid[nw] = dfs(a[nw]);
}
```

```cpp
int fnl = 0;
for (int i = 1; i <= rsz; i++) fnl += ans[i];
```

## E

注意到整个正方形由第一行和第一列决定，且第一行中的最长连续段和第一列最长连续段的乘积为正方形最大单色矩形的面积

转化为序列上问题，求长为$n$的序列，最长连续段为$i$的染色方案数，可以DP

$f_{i,j}$表示前$i$个元素，最长连续段为$j$的方案数，数据不大，暴力转移就行，枚举上一段有多长
$$
f_{i,j}=\sum\limits_{k=0}^{j-1}f_{i-j,k}+\sum\limits_{k=i-j}^{i-1}f_{k,j}
$$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 505;
const int P = 998244353;
int n, m;
ll f[MAXN][MAXN];
int main(){
	scanf("%d%d", &n, &m);
	f[0][0] = 1;
	for (int i = 1; i <= n; i++){
		for (int j = 1; j <= i; j++){
			for (int k = 0; k < j; k++) f[i][j] += f[i - j][k];
			for (int k = i - j; k < i; k++) f[i][j] += f[k][j];
			f[i][j] %= P;
		}
	}
	ll ans = 0;
	for (int i = 1; i <= n; i++){
		for (int j = 1; i * j < m && j <= n; j++){
			ans = (ans + f[n][i] * f[n][j]) % P;
		}
	}
	printf("%lld\n", 2 * ans % P);
	return 0;
}
```

## F

BSU=DSU？

可反悔贪心，也可以看成类似二分图匹配的算法

当在一个位置放元素时，将这个位置的指针指向该元素可以放的另一个位置，如果后面想要在这个位置放时，就可以让当前元素反悔

用并查集进行维护

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 2000051;
int n, csz;
int p[MAXN];
int a[MAXN], b[MAXN];
int c[MAXN];
bool flag[MAXN];
int findroot(int nw){
	if (p[nw] == nw) return nw;
	else return p[nw] = findroot(p[nw]);
}
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%d%d", a + i, b + i);
		c[++csz] = a[i];
		c[++csz] = b[i];
	}
	sort(c + 1, c + csz + 1);
	csz = unique(c + 1, c + csz + 1) - c - 1;
	for (int i = 1; i <= csz; i++) p[i] = i;
	int ans = 0;
	for (int i = 1; i <= n; i++){
		a[i] = lower_bound(c + 1, c + csz + 1, a[i]) - c;
		b[i] = lower_bound(c + 1, c + csz + 1, b[i]) - c;
		if (flag[findroot(a[i])] && flag[findroot(b[i])]){
			printf("-1\n");
			return 0;
		}
		if (findroot(a[i]) > findroot(b[i]) || flag[findroot(a[i])]) swap(a[i], b[i]);
		int ra = findroot(a[i]), rb = findroot(b[i]);
		ans = max(ans, c[ra]);
		if (ra == rb) flag[ra] = true;
		else p[ra] = rb;
	}
	printf("%d\n", ans);
	return 0;
}
```

# Educational Codeforces Round 50（1036）

## A

对每个点，高度升高1则图形面积增加1，各个点的高度和就等于$k$，图形整体高度最小值为$\left\lceil\frac{k}{n}\right\rceil$

```cpp
ans = (k + n - 1) / n;
```

## B

整体策略：先在起点来回走对角线（红），再走到终点左下（绿），最后一直向右上走（蓝）

![](https://cdn.luogu.com.cn/upload/image_hosting/czq641cx.png)

需要分4类讨论：

1. 第二步需要走的距离为偶数，总步数减去第三步所需步数为偶数

最好的情况，按照上图处理即可

2. 第二步需要走的距离为偶数，总步数减去第三步所需步数为奇数

第一个样例的情况，需要走两步不是对角线的路

![](https://cdn.luogu.com.cn/upload/image_hosting/og74k7m0.png)

3. 第二步需要走的距离为奇数，总步数减去第三步所需步数为偶数

![](https://cdn.luogu.com.cn/upload/image_hosting/q4c3p1yu.png)

4. 第二步需要走的距离为奇数，总步数减去第三步所需步数为奇数

![](https://cdn.luogu.com.cn/upload/image_hosting/6kecixas.png)

如果$k$比第二步和第三步所需的步数少则不可能走到

```cpp
if (n > m) swap(n, m);
ll tol = n + (m - n) / 2 * 2;
if ((m - n) & 1){
	if ((k - tol) & 1) k--;
	else {
		k--;
		tol++;
	}
} else {
	if ((k - tol) & 1) {
		tol--;
		k -= 2;
	}
}
if (tol > k) printf("-1\n");
else printf("%lld\n", k);
```

## C

满足条件的数一共有$10^3\operatorname{C}_{18}^3+10^2\operatorname{C}_{18}^2+10\operatorname{C}_{18}^{1}=831480$种

先预处理出所有满足条件的数，每次询问时二分搜索对应下标

```cpp
ten[0] = 1;
for (int i = 1; i <= 18; i++) ten[i] = ten[i - 1] * 10;
for (int i = 0; i < 18 - 2; i++){
	for (int j = i + 1; j < 18 - 1; j++){
		for (int k = j + 1; k < 18; k++){
			for (int nm1 = 0; nm1 < 10; nm1++){
				for (int nm2 = 0; nm2 < 10; nm2++){
					for (int nm3 = 0; nm3 < 10; nm3++){
						a[++asz] = nm1 * ten[i] + nm2 * ten[j] + nm3 * ten[k];
					}
				}
			}
		}
	}
}
for (int i = 0; i < 18 - 1; i++){
	for (int j = i + 1; j < 18; j++){
		for (int nm1 = 0; nm1 < 10; nm1++){
			for (int nm2 = 0; nm2 < 10; nm2++){
				a[++asz] = nm1 * ten[i] + nm2 * ten[j];
			}
		}
	}
}
for (int i = 0; i < 18; i++){
	for (int nm1 = 0; nm1 < 10; nm1++){
		a[++asz] = nm1 * ten[i];
	}
}
a[++asz] = 1e18;
sort(a + 1, a + asz + 1);
asz = unique(a + 1, a + asz + 1) - a - 1;
int t;
scanf("%d", &t);
while (t--){
	ll u, v;
	scanf("%lld%lld", &u, &v);
	int pl = lower_bound(a + 1, a + asz + 1, u) - a - 1;
	int pr = upper_bound(a + 1, a + asz + 1, v) - a - 1;
	printf("%d\n", pr - pl);
}
```

## D

在变化的过程中，数组所有元素的和一直不变，只要$a$和$b$元素和不同就不可能通过变化使$a$和$b$相等

如果$a$和$b$元素和相等，可以将数组里每个数都加在一起得到一个合法解

用两个指针分别指向$a$和$b$数组中第一个未匹配的数

每次找到$a$中的一个右端点，二分查找$b$中的右端点使得$a$、$b$中选中的子串和相等

如果找到，就将两个指向左端点的指针分别移到对应右端点右侧

```cpp
int la = 1, lb = 1;
for (int ra = 1; ra <= n; ra++){
	ll nm = s1[ra] - s1[la - 1] + s2[lb - 1];
	int pos = lower_bound(s2 + lb, s2 + m + 1, nm) - s2;
	if (s2[pos] == nm){
		lb = pos + 1;
		la = ra + 1;
		ans++;
	}
}
```

## E

大模拟，先求每个线段盖住多少整点，再减去多条线同时覆盖的情况

用set保证多线共点时恰好把多余的减去

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 1051;
int n;
ll ans;
set <pair <int, int> > st;
bool flag[MAXN];
int ax[MAXN], ay[MAXN], bx[MAXN], by[MAXN], p[MAXN], q[MAXN];
int gcd(int u, int v){
	if (v == 0) return u;
	else return gcd(v, u % v);
}
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%d%d%d%d", ax + i, ay + i, bx + i, by + i);
		if (ax[i] > bx[i]){
			swap(ax[i], bx[i]);
			swap(ay[i], by[i]);
		}
	}
	for (int i = 1; i <= n; i++){
		int g = gcd(abs(ax[i] - bx[i]), abs(ay[i] - by[i]));
		ans += g + 1;
		p[i] = (bx[i] - ax[i]) / g;
		q[i] = (by[i] - ay[i]) / g;
		// printf("pq %d %d %d\n", i, p[i], q[i]);
		// printf("ans %lld\n", ans);
	}
	for (int i = 1; i <= n; i++){
		st.clear();
		for (int j = i + 1; j <= n; j++){
			// printf("ij %d %d\n", i, j);
			if (1ll * (by[j] - ay[j]) * (bx[i] - ax[i]) == 1ll * (by[i] - ay[i]) * (bx[j] - ax[j])) continue;
			if (1ll * p[i] * q[j] - 1ll * p[j] * q[i] == 0) continue;
			if ((1ll * q[j] * (ax[j] - ax[i]) - 1ll * p[j] * (ay[j] - ay[i])) % (1ll * p[i] * q[j] - 1ll * p[j] * q[i])) continue;
			ll k1 = (1ll * q[j] * (ax[j] - ax[i]) - 1ll * p[j] * (ay[j] - ay[i])) / (1ll * p[i] * q[j] - 1ll * p[j] * q[i]);
			ll k2 = (q[j] == 0) ? ((k1 * p[i] + ax[i] - ax[j]) / p[j]) : ((k1 * q[i] + ay[i] - ay[j]) / q[j]);
			// printf("k1 %lld\n", k1);
			if (k1 < 0) continue;
			if (p[i] == 0 && k1 > (by[i] - ay[i]) / q[i]) continue;
			if (p[i] != 0 && k1 > (bx[i] - ax[i]) / p[i]) continue;
			if (k2 < 0) continue;
			if (p[j] == 0 && k2 > (by[j] - ay[j]) / q[j]) continue;
			if (p[j] != 0 && k2 > (bx[j] - ax[j]) / p[j]) continue;
			if (st.find(make_pair(ax[i] + k1 * p[i], ay[i] + k1 * q[i])) != st.end()) continue;
			st.insert(make_pair(ax[i] + k1 * p[i], ay[i] + k1 * q[i]));
			ans--;
			// printf("ans %lld\n", ans);
		}
	}
	printf("%lld\n", ans);
	return 0;
}
```

## F

一个数$x$​合法当且仅当不存在整数$y$和$k>1$满足$x=y^k$

用莫比乌斯函数进行容斥$ans=n-1+\sum\limits_{d=2}^{60}\mu(d)\sum\limits_{i=1}^n[\exists x\in\mathbb{Z},i=x^d]=n-1+\sum\limits_{d=2}^{60}\mu(d)\left\lfloor n^{1/d}-1\right\rfloor$​

注意pow会被卡精度，需要左右调整一下

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const ll INF = 1000000000000000051;
int t;
ll n;
int mu[61];
ll pwr(ll x, ll y){
	ll ans = 1;
	while (y){
		if (y & 1) ans = INF / ans >= x ? ans * x : INF;
		x = INF / x >= x ? x * x : INF;
		y >>= 1;
	}
	return ans;
}
int main(){
	for (int i = 2; i <= 60; i++){
		int x = i;
		mu[i] = 1;
		for (int j = 2; j * j <= x; j++){
			if (x % j) continue;
			if (x % (j * j) == 0){
				mu[i] = 0;
				break;
			}
			x /= j;
			mu[i] = -mu[i];
		}
		if (x > 1) mu[i] = -mu[i];
		// printf("mu %d %d\n", i, mu[i]);
	}
	scanf("%d", &t);
	while (t--){
		scanf("%lld", &n);
		ll ans = n - 1;
		for (int d = 2; d <= 60; d++){
			ll nm = (ll)(pow(n, 1.0 / d));
			while (pwr(nm + 1, d) <= n) nm++;
			while (pwr(nm, d) > n) nm--;
			ans += mu[d] * (nm - 1);
			// printf("ans %lld %lf\n", ans, pow(n, 1.0 / d));
		}
		printf("%lld\n", ans);
	}
	return 0;
}
```



# Educational Codeforces Round 51（1051）

## A

分别统计小写字母、大写字母、数字的个数，哪个没有就补那个

```cpp
cnt1 = cnt2 = cnt3 = nm = 0;
scanf("%s", s + 1);
int sz = strlen(s + 1);
for (int i = 1; i <= sz; i++){
	if (s[i] >= 'a' && s[i] <= 'z'){
		if (!cnt1++) nm++;
	} else if (s[i] >= 'A' && s[i] <= 'Z'){
		if (!cnt2++) nm++;
	} else if (s[i] >= '0' && s[i] <= '9'){
		if (!cnt3++) nm++;
	}
}
// printf("%d %d %d\n", cnt1, cnt2, cnt3);
if (nm == 3){
	printf("%s\n", s + 1);
} else if (nm == 2){
	char c;
	if (!cnt1) c = 'a';
	else if (!cnt2) c = 'A';
	else c = '0'; 
	for (int i = 1; i <= sz; i++){
		if (s[i] >= 'a' && s[i] <= 'z'){
			if (cnt1 > 1 && nm == 2){
				nm++;
				s[i] = c;
			}
		} else if (s[i] >= 'A' && s[i] <= 'Z'){
			if (cnt2 > 1 && nm == 2){
				nm++;
				s[i] = c;
			}
		} else if (s[i] >= '0' && s[i] <= '9'){
			if (cnt3 > 1 && nm == 2){
				nm++;
				s[i] = c;
			}
		}
	}
	printf("%s\n", s + 1);
} else {
	if (cnt1) printf("A0");
	else if (cnt2) printf("a0");
	else printf("aA");
	printf("%s\n", s + 3);
}
```

## B

不要被题目忽悠了，其实这题非常简单

因为$\gcd(x,x+1)=1$，只要取相邻两个数为一组就行了

```cpp
printf("YES\n");
for (ll i = l; i <= r; i += 2){
	printf("%lld %lld\n", i, i + 1);
}
```

## C

如果有偶数个数只出现了一次，两边各分一半

如果有奇数个数只出现一次，将一个出现三次以上的数拆一个放在左边，其他放右边

```cpp
for (int i = 1; i <= n; i++){
	scanf("%d", a + i);
	bx[a[i]]++;
}
int cnt1 = 0, cnt3 = 0, cnta = 0;
for (int i = 1; i <= 100; i++){
	if (bx[i] == 1) cnt1++;
	if (bx[i] >= 3) cnt3++;
}
if (cnt1 & 1){
	if (cnt3 == 0){
		printf("NO\n");
		return 0;
	}
	bool flag = false;
	for (int i = 1; i <= 100; i++){
		if (bx[i] == 1 && cnta * 2 + 2 <= cnt1){
			b1[i]++;
			cnta++;
		}
		if (bx[i] >= 3 && !flag){
			b1[i]++;
			flag = true;
		}
	}
} else {
	for (int i = 1; i <= 100; i++){
		if (bx[i] == 1 && cnta * 2 + 2 <= cnt1){
			b1[i]++;
			cnta++;
		}
	}
}
printf("YES\n");
for (int i = 1; i <= n; i++){
	if (b1[a[i]]){
		b1[a[i]]--;
		printf("A");
	} else {
		printf("B");
	}
}
```

## D

DP，状压最后一列的状态，每次统计新增了多少个连通块

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int MAXN = 1051;
const int P = 998244353;
int n, k;
ll f[MAXN][2 * MAXN][4];
int main(){
	scanf("%d%d", &n, &k);
	f[1][1][0] = f[1][2][1] = f[1][2][2] = f[1][1][3] = 1;
	for (int i = 1; i <= n; i++){
		for (int j = 1; j <= k; j++){
			//00
			f[i + 1][j + 0][0] = (f[i + 1][j + 0][0] + f[i][j][0]) % P;
			f[i + 1][j + 1][1] = (f[i + 1][j + 1][1] + f[i][j][0]) % P;
			f[i + 1][j + 1][2] = (f[i + 1][j + 1][2] + f[i][j][0]) % P;
			f[i + 1][j + 1][3] = (f[i + 1][j + 1][3] + f[i][j][0]) % P;
			//01
			f[i + 1][j + 0][0] = (f[i + 1][j + 0][0] + f[i][j][1]) % P;
			f[i + 1][j + 0][1] = (f[i + 1][j + 0][1] + f[i][j][1]) % P;
			f[i + 1][j + 2][2] = (f[i + 1][j + 2][2] + f[i][j][1]) % P;
			f[i + 1][j + 0][3] = (f[i + 1][j + 0][3] + f[i][j][1]) % P;
			//10
			f[i + 1][j + 0][0] = (f[i + 1][j + 0][0] + f[i][j][2]) % P;
			f[i + 1][j + 2][1] = (f[i + 1][j + 2][1] + f[i][j][2]) % P;
			f[i + 1][j + 0][2] = (f[i + 1][j + 0][2] + f[i][j][2]) % P;
			f[i + 1][j + 0][3] = (f[i + 1][j + 0][3] + f[i][j][2]) % P;
			//11
			f[i + 1][j + 1][0] = (f[i + 1][j + 1][0] + f[i][j][3]) % P;
			f[i + 1][j + 1][1] = (f[i + 1][j + 1][1] + f[i][j][3]) % P;
			f[i + 1][j + 1][2] = (f[i + 1][j + 1][2] + f[i][j][3]) % P;
			f[i + 1][j + 0][3] = (f[i + 1][j + 0][3] + f[i][j][3]) % P;
		}
	}
	printf("%lld\n", (f[n][k][0] + f[n][k][1] + f[n][k][2] + f[n][k][3]) % P);
	return 0;
}
```

## E

用扩展KMP求$a$​每个后缀和$l,r$​ 的LCP，然后就可以$O(1)$​比较$a$​中一段和$l,r$​​的大小

直接DP，双指针维护转移区间

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 1000051;
const int P = 998244353;
int n, lsz, rsz;
char a[MAXN], l[MAXN * 2], r[MAXN * 2];
int zl[MAXN * 2], zr[MAXN * 2];
int lcp[MAXN], rcp[MAXN];
ll f[MAXN], s[MAXN];
void getlcp(){
	memcpy(l + lsz + 1, a + 1, n * sizeof(char));
	memcpy(r + rsz + 1, a + 1, n * sizeof(char));
	int lp = 0, rp = 0;
	for (int i = 2; i <= lsz + n; i++){
		if (i > rp) zl[i] = 0;
		else zl[i] = min(zl[i - lp + 1], rp - i + 1);
		while (i + zl[i] <= lsz + n && l[1 + zl[i]] == l[i + zl[i]]) zl[i]++;
		if (i + zl[i] - 1 > rp){
			lp = i;
			rp = i + zl[i] - 1;
		}
	}
	lp = rp = 0;
	for (int i = 2; i <= rsz + n; i++){
		if (i > rp) zr[i] = 0;
		else zr[i] = min(zr[i - lp + 1], rp - i + 1);
		while (i + zr[i] <= rsz + n && r[1 + zr[i]] == r[i + zr[i]]) zr[i]++;
		if (i + zr[i] - 1 > rp){
			lp = i;
			rp = i + zr[i] - 1;
		}
	}
	memcpy(lcp + 1, zl + lsz + 1, n * sizeof(int));
	memcpy(rcp + 1, zr + rsz + 1, n * sizeof(int));
}
bool checkl(int pos){
	return lcp[pos] >= lsz || a[pos + lcp[pos]] > l[1 + lcp[pos]];
}
bool checkr(int pos){
	return rcp[pos] < rsz && a[pos + rcp[pos]] > r[1 + rcp[pos]];
}
int main(){
	// freopen("1.in", "r", stdin);
	scanf("%s%s%s", a + 1, l + 1, r + 1);
	n = strlen(a + 1);
	lsz = strlen(l + 1);
	rsz = strlen(r + 1);
	getlcp();
	// for (int i = 1; i <= n; i++) printf("lrcp %d %d %d\n", i, lcp[i], rcp[i]);
	int lp = 0, rp = -1;
	f[0] = s[0] = 1;
	for (int i = 1; i <= n; i++){
		while (i - (rp + 1) > lsz) rp++;
		if (rp + 1 + lsz == i && checkl(rp + 2)) rp++;
		while (i - lp > rsz) lp++;
		if (lp + rsz == i && checkr(lp + 1)) lp++;
		if (lp <= 0) f[i] = s[rp];
		else f[i] = (s[rp] - s[lp - 1]) % P;
		if (l[1] == '0' && a[i] == '0') f[i] = (f[i] + f[i - 1]) % P;
		s[i] = s[i - 1];
		if (a[i + 1] != '0') s[i] = (s[i] + f[i]) % P;
		// printf("f %d %lld %lld\n", i, f[i], s[i]);
		// printf("lr %d %d\n", lp, rp);
	}
	printf("%lld\n", (f[n] + P) % P);
	return 0;
}
```

## F

边只比点多一点

先求出一个生成树，然后用多的边跑一遍floyd

回答询问就枚举走了多余边的那一段

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 100051;
const int MAXA = 45;
const ll INF = 0x3f3f3f3f3f3f3f3f;
struct Edge{
	int t;
	ll w;
	int nxt;
	Edge(int t = 0, ll w = 0, int nxt = 0): t(t), w(w), nxt(nxt){}
} g[MAXN * 2];
int n, m, q, asz, gsz;
int fte[MAXN];
void addedge(int u, int v, int w){
	g[++gsz] = Edge(v, w, fte[u]);
	fte[u] = gsz;
}
int p[MAXN];
int findroot(int nw){
	if (p[nw] == nw) return nw;
	else return p[nw] = findroot(p[nw]);
}
int dep[MAXN];
int fa[MAXN][21];
ll fd[MAXN][21];
void dfs0(int nw){
	// printf("dfs0 %d %d\n", nw, fa[nw][0]);
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (nxtn == fa[nw][0]) continue;
		dep[nxtn] = dep[nw] + 1;
		fa[nxtn][0] = nw;
		fd[nxtn][0] = g[i].w;
		for (int j = 1; j <= 20; j++){
			fa[nxtn][j] = fa[fa[nxtn][j - 1]][j - 1];
			fd[nxtn][j] = fd[nxtn][j - 1] + fd[fa[nxtn][j - 1]][j - 1];
		}
		dfs0(nxtn);
	}
}
ll getdis(int u, int v){
	if (dep[u] < dep[v]) swap(u, v);
	ll ans = 0;
	for (int i = 20; i >= 0; i--){
		if (dep[u] - (1 << i) >= dep[v]){
			ans += fd[u][i];
			u = fa[u][i];
		}
	}
	if (u == v) return ans;
	for (int i = 20; i >= 0; i--){
		if (fa[u][i] != fa[v][i]){
			ans += fd[u][i] + fd[v][i];
			u = fa[u][i];
			v = fa[v][i];
		}
	}
	return ans + fd[u][0] + fd[v][0];
}
int a[MAXA];
int b[MAXN];
ll f[MAXA][MAXA];
ll dis[MAXA][MAXN];
void dfs1(int nw, int rt, int fa = -1){
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (nxtn == fa) continue;
		dis[rt][nxtn] = dis[rt][nw] + g[i].w;
		dfs1(nxtn, rt, nw);
	}
}
int main(){
	memset(f, 0x3f, sizeof(f));
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++) p[i] = i;
	for (int i = 1; i <= m; i++){
		int u, v, w;
		scanf("%d%d%d", &u, &v, &w);
		int ru = findroot(u), rv = findroot(v);
		if (ru == rv){
			if (!b[u]){
				a[++asz] = u;
				b[u] = asz;
			}
			if (!b[v]){
				a[++asz] = v;
				b[v] = asz;
			}
			f[b[u]][b[v]] = f[b[v]][b[u]] = min(f[b[u]][b[v]], 1ll * w);
		} else {
			p[ru] = rv;
			addedge(u, v, w);
			addedge(v, u, w);
		}
	}
	dfs0(1);
	for (int i = 1; i <= asz; i++){
		for (int j = 1; j <= asz; j++){
			f[i][j] = min(f[i][j], getdis(a[i], a[j]));
		}
	}
	for (int k = 1; k <= asz; k++){
		for (int i = 1; i <= asz; i++){
			for (int j = 1; j <= asz; j++){
				f[i][j] = min(f[i][j], f[i][k] + f[k][j]);
			}
		}
	}
	for (int i = 1; i <= asz; i++) dfs1(a[i], i);
	scanf("%d", &q);
	while (q--){
		int u, v;
		scanf("%d%d", &u, &v);
		ll ans = getdis(u, v);
		// printf("ans %lld\n", ans);
		for (int i = 1; i <= asz; i++){
			for (int j = 1; j <= asz; j++){
				ans = min(ans, dis[i][u] + f[i][j] + dis[j][v]);
				// printf("ans %lld\n", ans);
			}
		}
		printf("%lld\n", ans);
	}
	return 0;
}
```

# Educational Codeforces Round 52（1065）

## A

按照题目所述，先计算能买多少个，再计算能送多少个

```cpp
ll ans = s / c;
ans = ans + ans / a * b;
```

## B

使独立点个数最小的策略：将点分成两个两个一组，每组连一条边

```cpp
minAns = max(0ll, n - 2 * m);
```

独立点个数最大策略：将所有边连在$k$个点中，使其他点都成为独立点

二分出最小的$k$，若$k=1$则$maxAns=n$，否则$maxAns=n-k$

```cpp
ll lft = 1, rgt = n, k = n;
while (lft <= rgt){
	ll mid = (lft + rgt) >> 1;
	if (mid * (mid - 1) / 2 >= m){
		k = mid;
		rgt = mid - 1;
	} else lft = mid + 1;
}
if (k == 1) maxAns = n;
else maxAns = n - k;
```

## C

用差分数组记录每个高度的方块比上面多出多少

从上往下扫描，利用差分数组维护当前方块数量，如果当前和大于$k$则答案加一

```cpp
ll nw = 0, cnt = 0;
for (ll i = mx; i >= 0; i--){
	nw += d[i];
	if (nw == n) break;
	if (cnt + nw > k){
		ans++;
		cnt = nw;
	} else cnt += nw;
}
if (cnt) ans++;
```

## D

先用floyd求出任意两个状态之间的距离

状态包括当前坐标以及所用的棋子，0是马，1是象，2是车

距离包括两个参数，分别代表时间和换棋子的次数

```cpp
struct Node{
	int nm1, nm2;
	bool operator < (const Node &o) const{
		if (nm1 != o.nm1) return nm1 < o.nm1;
		else return nm2 < o.nm2;
	}
	Node operator + (const Node &o) const{
		return (Node){nm1 + o.nm1, nm2 + o.nm2};
	}
};
```

```cpp
int m0[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
for (int x1 = 0; x1 < n; x1++){
	for (int y1 = 0; y1 < n; y1++){
		for (int j = 0; j < 3; j++){
			for (int k = 0; k < 8; k++){
				int x2 = x1 + m0[k][0], y2 = y1 + m0[k][1];
				// if (x1 == 0 && y1 == 0 && x2 == 2 && y2 == 1) printf("------------- %d\n", j);
				if (x2 >= n || x2 < 0 || y2 >= n || y2 < 0) continue;
				f[x1][y1][j][x2][y2][0] = (Node){1 + (j != 0), j != 0};
			}
			for (int k = 1; ; k++){
				int x2 = x1 - k, y2 = y1 - k;
				if (x2 < 0 || x2 >= n || y2 < 0 || y2 >= n) break;
				f[x1][y1][j][x2][y2][1] = (Node){1 + (j != 1), j != 1};
			}
			for (int k = 1; ; k++){
				int x2 = x1 + k, y2 = y1 + k;
				if (x2 < 0 || x2 >= n || y2 < 0 || y2 >= n) break;
				f[x1][y1][j][x2][y2][1] = (Node){1 + (j != 1), j != 1};
			}
			for (int k = 1; ; k++){
				int x2 = x1 + k, y2 = y1 - k;
				if (x2 < 0 || x2 >= n || y2 < 0 || y2 >= n) break;
				f[x1][y1][j][x2][y2][1] = (Node){1 + (j != 1), j != 1};
			}
			for (int k = 1; ; k++){
				int x2 = x1 - k, y2 = y1 + k;
				if (x2 < 0 || x2 >= n || y2 < 0 || y2 >= n) break;
				f[x1][y1][j][x2][y2][1] = (Node){1 + (j != 1), j != 1};
			}
			for (int k = 0; k < n; k++){
				if (k != y1) f[x1][y1][j][x1][k][2] = (Node){1 + (j != 2), j != 2};
				if (k != x1) f[x1][y1][j][k][y1][2] = (Node){1 + (j != 2), j != 2};
			}
		}
	}
}
for (int x3 = 0; x3 < n; x3++){
	for (int y3 = 0; y3 < n; y3++){
		for (int p3 = 0; p3 < 3; p3++){
			for (int x1 = 0; x1 < n; x1++){
				for (int y1 = 0; y1 < n; y1++){
					for (int p1 = 0; p1 < 3; p1++){
						for (int x2 = 0; x2 < n; x2++){
							for (int y2 = 0; y2 < n; y2++){
								for (int p2 = 0; p2 < 3; p2++){
									f[x1][y1][p1][x2][y2][p2] = min(f[x1][y1][p1][x2][y2][p2], f[x1][y1][p1][x3][y3][p3] + f[x3][y3][p3][x2][y2][p2]);
									if (x3 == 2 && y3 == 1 && x1 == 0 && y1 == 2 && x2 == 2 && y2 == 0 && p1 == 0 && p3 == 0 && p2 == 2){
										// printf("---------------- %d %d\n", f[x1][y1][p1][x2][y2][p2].nm1, f[x1][y1][p1][x2][y2][p2].nm2);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
```

再进行dp，$dp[i][j]$表示走到数字$i$，最后用的棋子为$j$的最小距离

```cpp
dp[1][0] = dp[1][1] = dp[1][2] = (Node){0, 0};
for (int i = 2; i <= n * n; i++){
	for (int j = 0; j < 3; j++){
		dp[i][j] = (Node){1000000000, 1000000000};
		dp[i][j] = min(dp[i][j], dp[i - 1][0] + f[px[i - 1]][py[i - 1]][0][px[i]][py[i]][j]);
		dp[i][j] = min(dp[i][j], dp[i - 1][1] + f[px[i - 1]][py[i - 1]][1][px[i]][py[i]][j]);
		dp[i][j] = min(dp[i][j], dp[i - 1][2] + f[px[i - 1]][py[i - 1]][2][px[i]][py[i]][j]);
		// printf("dp %d %d %d %d\n", i, j, dp[i][j].nm1, dp[i][j].nm2);
	}
}
Node ans = min(dp[n * n][0], min(dp[n * n][1], dp[n * n][2]));
```

## E

对于每一组对称的两段$[b_{i-1}+1,b_i]\cup[n-b_i+1,n-b_{i-1}]$，一共有$|A|^{b_i-b_{i-1}}(|A|^{b_i-b_{i-1}}-1)/2+|A|^{b_i-b_{i-1}}$种方案（先统计不同的情况，再加上相同的）

把每一段的乘起来就行

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXM = 200051;
const int P = 998244353;
const int INV2 = 499122177;
int n, m, A;
int b[MAXM];
ll pwr(ll x, ll y){
	x %= P;
	y = (y % (P - 1) + (P - 1)) % (P - 1);
	ll ans = 1;
	while (y){
		if (y & 1) ans = ans * x % P;
		x = x * x % P;
		y >>= 1;
	}
	return ans;
}
int main(){
	scanf("%d%d%d", &n, &m, &A);
	for (int i = 1; i <= m; i++) scanf("%d", b + i);
	sort(b + 1, b + m + 1);
	ll ans = 1;
	for (int i = 1; i <= m; i++){
		ll nm = pwr(A, b[i] - b[i - 1]);
		ans = ans * (nm * (nm - 1) % P * INV2 % P + nm) % P;
	}
	ans = ans * pwr(A, n - b[m] * 2) % P;
	printf("%lld\n", ans);
	return 0;
}
```

## F

根据题目描述的规则，每个节点向其子节点连边，每个叶子节点向自己的$k$级祖先连边

然后用tarjan缩点，再在DAG上记搜

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 1000051;
struct Edge{
	int t, nxt;
	Edge(int t = 0, int nxt = 0): t(t), nxt(nxt){}
} g[MAXN * 2];
int n, k, gsz;
int fa[MAXN];
int fte[MAXN * 2];
bool lf[MAXN];
void addedge(int u, int v){
	g[++gsz] = Edge(v, fte[u]);
	fte[u] = gsz;
}
int vt[MAXN];
int dep[MAXN];
void dfs0(int nw){
	vt[dep[nw]] = nw;
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		dep[nxtn] = dep[nw] + 1;
		dfs0(nxtn);
	}
	if (lf[nw]) addedge(nw, vt[max(0, dep[nw] - k)]);
}
int cnt, ssz;
int st[MAXN], tp;
int dfn[MAXN], low[MAXN], scc[MAXN];
bool inst[MAXN];
int w[MAXN];
void tarjan(int nw){
	// printf("tarjan %d\n", nw);
	dfn[nw] = low[nw] = ++cnt;
	st[++tp] = nw;
	inst[nw] = true;
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (!dfn[nxtn]){
			tarjan(nxtn);
			low[nw] = min(low[nw], low[nxtn]);
		} else if (inst[nxtn]) low[nw] = min(low[nw], dfn[nxtn]);
	}
	if (dfn[nw] == low[nw]){
		ssz++;
		while (st[tp] != nw){
			scc[st[tp]] = ssz;
			w[ssz] += lf[st[tp]];
			inst[st[tp]] = false;
			tp--;
		}
		scc[nw] = ssz;
		w[ssz] += lf[nw];
		inst[nw] = false;
		tp--;
	}
}
int f[MAXN];
int dfs1(int nw){
	// printf("dfs1 %d\n", nw);
	if (f[nw] != -1) return f[nw];
	int mx = 0;
	for (int i = fte[nw + n]; i; i = g[i].nxt){
		int nxtn = g[i].t - n;
		mx = max(mx, dfs1(nxtn));
	}
	return f[nw] = w[nw] + mx;
}
int main(){
	memset(lf, true, sizeof(lf));
	scanf("%d%d", &n, &k);
	for (int i = 2; i <= n; i++){
		scanf("%d", fa + i);
		addedge(fa[i], i);
		lf[fa[i]] = false;
	}
	dfs0(1);
	tarjan(1);
	for (int i = 1; i <= n; i++){
		// printf("scc %d %d\n", i, scc[i]);
		for (int j = fte[i]; j; j = g[j].nxt){
			int nxtn = g[j].t;
			if (scc[i] != scc[nxtn]) addedge(scc[i] + n, scc[nxtn] + n);
		}
	}
	memset(f, -1, sizeof(f));
	printf("%d\n", dfs1(scc[1]));
	return 0;
}
```

# Educational Codeforces Round 53（1073）

## A

找到两个相邻的不同字母，每个字母一定不会超过一半

```cpp
bool suc = false;
for (int i = 2; i <= n; i++){
	if (s[i] != s[i - 1]){
		printf("YES\n%c%c\n", s[i - 1], s[i]);
		suc = true;
		break;
	}
}
if (!suc) printf("NO\n");
```

## B

记录目前已经取出多少书，再记录每一个书出现的位置

每次取书时判断书是否已经取过，如果没有，就直接暴力将栈顶的书一个一个取出来

```cpp
int mk = 0;
for (int i = 1; i <= n; i++){
	if (p[b[i]] <= mk) printf("0 ");
	else {
		printf("%d ", p[b[i]] - mk);
		mk = p[b[i]];
	}
}
```

## C

预处理某一步之前所有操作变化坐标，类似前缀和

二分所要更改的长度，枚举每一个更改的区间

利用前缀和算出本区间应该变化的坐标，判断是否可行

```cpp
bool check(int x){
	for (int i = 1; i <= n - x + 1; i++){
		int j = i + x - 1;
		Node nw = t - s[n] + s[j] - s[i - 1];
		if (abs(nw.x) + abs(nw.y) <= x) return true;
	}
	return false;
}
```

```cpp
for (int i = 1; i <= n; i++){
	switch (st[i]){
		case 'U' : s[i] = s[i - 1] + mv[0]; break;
		case 'D' : s[i] = s[i - 1] + mv[1]; break;
		case 'L' : s[i] = s[i - 1] + mv[2]; break;
		case 'R' : s[i] = s[i - 1] + mv[3]; break;
	}
}
if (abs(t.x) + abs(t.y) > n || abs(t.x + t.y - n) & 1){
	printf("-1\n");
	return 0;
}
int lft = 0, rgt = n, ans = -1;
while (lft <= rgt){
	int mid = (lft + rgt) >> 1;
	// printf("lrm %d %d %d\n", lft, rgt, mid);
	if (check(mid)){
		ans = mid;
		rgt = mid - 1;
	} else lft = mid + 1;
}
```

## D

每一次，暴力计算当前的钱在一圈中的购买方案，之后只要钱足够，一定还会用这个方案买，计算能用这种方案买多少次

```cpp
ll tol = 0;
while (true){
	ll s = 0, nm = 0;
	for (int i = 1; i <= n; i++){
		if (t >= a[i]){
			s += a[i];
			nm++;
			t -= a[i];
		}
	}
	if (s == 0) break;
	tol += t / s * nm + nm;
	t %= s;
}
```

## E

数位DP，记录当前位数，已选集合，是否为前导0，是否贴下界，是否贴上界

DP时候判断已选集合大小是否超过$k$

```cpp
# include <bits/stdc++.h>
# define ll long long
# define fst first
# define snd second
using namespace std;
const int MAXN = 21;
const int P = 998244353;
int k;
ll l, r;
int a[MAXN], b[MAXN];
int cnt[1 << 10];
ll p10[MAXN];
bool vis[MAXN][1 << 10][2][2][2];
pair <ll, ll> f[MAXN][1 << 10][2][2][2];
pair <ll, ll> dfs(int nw, int s, bool f0, bool f1, bool f2){
	if (nw == 20) return make_pair(0, 1);
	if (vis[nw][s][f0][f1][f2]) return f[nw][s][f0][f1][f2];
	vis[nw][s][f0][f1][f2] = true;
	int mi = f1 ? a[nw] : 0, mx = f2 ? b[nw] : 9;
	pair <ll, ll> ans = make_pair(0, 0);
	for (int i = mi; i <= mx; i++){
		int ns = (f0 && i == 0) ? s : (s | (1 << i));
		if (cnt[ns] > k) continue;
		pair <ll, ll> na = dfs(nw + 1, ns, f0 && i == 0, f1 && i == mi, f2 && i == mx);
		ans.fst = (ans.fst + na.fst + i * p10[19 - nw] % P * na.snd % P) % P;
		ans.snd = (ans.snd + na.snd) % P;
	}
	// printf("f %d %d %d %d %d %lld %lld\n", nw, s, f0, f1, f2, ans.fst, ans.snd);
	return f[nw][s][f0][f1][f2] = ans;
}
int main(){
	for (int i = 1; i < (1 << 10); i++){
		if (i == (i & (-i))) cnt[i] = 1;
		else cnt[i] = cnt[i ^ (i & (-i))] + 1;
	}
	p10[0] = 1;
	for (int i = 1; i <= 18; i++) p10[i] = p10[i - 1] * 10ll % P;
	scanf("%lld%lld%d", &l, &r, &k);
	for (int i = 19; i >= 1; i--){
		a[i] = l % 10;
		l /= 10;
	}
	for (int i = 19; i >= 1; i--){
		b[i] = r % 10;
		r /= 10;
	}
	// for (int i = 1; i <= 19; i++) printf("ab %d %d %d\n", i, a[i], b[i]);
	printf("%lld\n", dfs(1, 0, true, true, true).fst);
	return 0;
}
```



## F

死活调不出来

# Educational Codeforces Round 54（1076）

## A

删掉第一个比后一个字符大的字符，如果没有就删最后一个字符

```cpp
int work(){
	for (int i = 1; i <= n; i++){
		if (s[i] > s[i + 1]) return i;
	}
	return n;
}
```

## B

只有第一次操作时$n$的最大质因数不是2

$n\equiv d(\mod 2)$，所以$2|n-d$

之后每一次$d$都等于2

```cpp
ll smallestPrimeDivisor(ll x){
	for (int i = 2; (ll)i * i <= x; i++){
		if (x % i == 0) return i;
	}
	return x;
}
```

```cpp
ans = (n - smallestPrimeDivisor(n)) / 2 + 1
```

## C

联立两个方程解得$a=\frac{d+\sqrt{d^2-4d}}{2}$，$b=\frac{d-\sqrt{d^2-4d}}{2}$

输出时需要保留多一点位数，不然会判错

```cpp
if (d * d - 4 * d < 0){
	printf("N\n");
	continue;
}
a = (d + sqrt(double(d * d - 4 * d))) / 2.0;
b = d - a;
printf("Y %.10lf %.10lf\n", a, b);
```

## D

先用dijkstra求最短路，记录路径，求出由最短路经构成的图

```cpp
void dijkstra(){
	memset(dis, 0x3f, sizeof(dis));
	dis[1] = 0;
	q.push((Node){1, 0});
	while (!q.empty()){
		Node nw = q.top();
		q.pop();
		if (nw.di > dis[nw.id]) continue;
		for (int i = fte[nw.id]; i; i = g[i].nxt){
			int nxtn = g[i].t;
			if (nw.di + g[i].c < dis[nxtn]){
				pre[nxtn] = i;
				dis[nxtn] = nw.di + g[i].c;
				q.push((Node){nxtn, dis[nxtn]});
			}
		}
	}
}
```
```cpp
for (int i = 2; i <= n; i++) flag[pre[i]] = true;
```

从1开始在最短路图上dfs或bfs，每找到一条边就加进答案里

```cpp
s.push(1);
int cnt = 0;
while (!s.empty()){
	int nw = s.top();
	s.pop();
	for (int i = fte[nw]; i; i = g[i].nxt){
		if (!flag[i]) continue;
		printf("%d ", (i >> 1));
		cnt++;
		if (cnt >= k || cnt >= n - 1) break;
		int nxtn = g[i].t;
		s.push(nxtn);
	}
	if (cnt >= k || cnt >= n - 1) break;
}
```

## E

当遍历到一个点时，将所有挂在这个点的修改打在深度上，跳出时再删除

一个点的最终权值就是遍历到它时它所在深度的改动总和

```cpp
# include <vector>
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int MAXN = 300051;
struct Edge{
	int t, nxt;
} g[MAXN * 2];
struct BIT{
	ll c[MAXN];
	inline int lowbit(int x){
		return x & (-x);
	}
	void add(int pos, ll nm){
		for (int i = pos; i < MAXN; i += lowbit(i)) c[i] += nm;
	}
	ll getsum(int pos){
		ll ans = 0;
		for (int i = pos; i; i -= lowbit(i)) ans += c[i];
		return ans;
	}
} t;
int n, m, gsz;
int fte[MAXN];
int dep[MAXN];
ll ans[MAXN];
vector <pair <int, int> > adj[MAXN];
void addedge(int x, int y){
	g[++gsz] = (Edge){y, fte[x]};
	fte[x] = gsz;
}
void dfs(int nw, int fa){
	int sz = adj[nw].size();
	for (int i = 0; i < sz; i++){
		t.add(dep[nw], adj[nw][i].second);
		t.add(dep[nw] + adj[nw][i].first + 1, -adj[nw][i].second);
	}
	ans[nw] = t.getsum(dep[nw]);
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (nxtn == fa) continue;
		dep[nxtn] = dep[nw] + 1;
		dfs(nxtn, nw);
	}
	for (int i = 0; i < sz; i++){
		t.add(dep[nw], -adj[nw][i].second);
		t.add(dep[nw] + adj[nw][i].first + 1, adj[nw][i].second);
	}
}
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n - 1; i++){
		int u, v;
		scanf("%d%d", &u, &v);
		addedge(u, v);
		addedge(v, u);
	}
	scanf("%d", &m);
	for (int i = 1; i <= m; i++){
		int v, d;
		ll x;
		scanf("%d%d%lld", &v, &d, &x);
		adj[v].push_back(make_pair(d, x));
	}
	dep[1] = 1;
	dfs(1, -1);
	for (int i = 1; i <= n; i++) printf("%lld ", ans[i]);
	putchar('\n');
	return 0;
}
```

## F

$fx_i$表示前$i$个串拼完后末尾最少有多少T，$fy_i$表示前$i$个串拼完后末尾最少有多少F
$$
\begin{cases}fx_i=fx_{i-1}+x_i-k\cdot y_i\\fy_i=fy_{i-1}+y_i-k\cdot x_i\end{cases}
$$
如果要让最后的T最少，一定是：k个T，1个F，k个T，1个F，……，F同理

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 300051;
int n, k;
int x[MAXN], y[MAXN];
int fx[MAXN], fy[MAXN];
int main(){
	scanf("%d%d", &n, &k);
	for (int i = 1; i <= n; i++) scanf("%d", x + i);
	for (int i = 1; i <= n; i++) scanf("%d", y + i);
	for (int i = 1; i <= n; i++){
		fx[i] = max(0ll, fx[i - 1] + x[i] - 1ll * k * y[i]);
		fy[i] = max(0ll, fy[i - 1] + y[i] - 1ll * k * x[i]);
		// printf("fxy %d %d %d\n", i, fx[i], fy[i]);
		if (fx[i] > k || fy[i] > k){
			printf("NO\n");
			return 0;
		}
	}
	printf("YES\n");
	return 0;
}
```

# Educational Codeforces Round 55（1082）

## A

一共有三种方案：直接翻到$y$，先翻到开头再翻到$y$，先翻到结尾再翻到$y$

将三种方案取最小值

```cpp
if (abs(y - x) % d == 0) ans = min(ans, abs(y - x) / d);
if ((y - 1) % d == 0) ans = min(ans, (x - 1 + d - 1) / d + (y - 1) / d);
if ((n - y) % d == 0) ans = min(ans, (n - x + d - 1) / d + (n - y) / d);
```

## B

预处理每个点左边和右边分别有多少连续的金牌

交换时一定是将一个银牌与金牌交换

计算每个银牌和金牌交换后两边连起来一共能有多少个连续金牌

```cpp
for (int i = 1; i <= n; i++){
	s1[i] = (a[i] == 'G') ? s1[i - 1] + 1 : 0;
	ans = max(ans, s1[i]);
}
for (int i = n; i >= 1; i--) s2[i] = (a[i] == 'G') ? s2[i + 1] + 1 : 0;
for (int i = 1; i <= n; i++) cnt += (a[i] == 'G');
for (int i = 1; i <= n; i++){
	if (a[i] == 'G') continue;
	if (s1[i - 1] + s2[i + 1] == cnt){
		ans = max(ans, s1[i - 1] + s2[i + 1]);
	} else ans = max(ans, s1[i - 1] + s2[i + 1] + 1);
}
```

## C

将不同学科按照参加人数从大到小排序

在每个学科内按照水平从大到小排序

设第$i$门学科一共有$a[i]$人

设$s[i][j]$为第$i$门学科，前$j$个人水平之和，$\sum_{i=1}^ma[i]=n$，可以$O(n)$预处理

就算水平是负数也不能不管，可以拿来凑数用

设$f[i][j]$为考虑前$i$门学科，每个学科参加$j$人时的最大水平之和，同样$O(n)$转移

$f[i][j]=\max\{f[i - 1][j], f[i - 1][j] + s[i][j]\}$

最后答案就是所有$f$的最大值

$ans=\max_{1\leq i\leq m}\{\max_{1\leq j\leq a[i]}\{f[i][j]\}\}$

```cpp
# include <vector>
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const ll NR = 100051;
struct Node{
	ll nm, id;
} b[NR];
ll n, m, ans;
ll c[NR];
vector <ll> a[NR];
ll f[NR];
bool cmp(Node x, Node y){
	return x.nm > y.nm;
}
bool cmp2(ll x, ll y){
	return x > y;
}
int main(){
	scanf("%lld%lld", &n, &m);
	for (ll i = 1; i <= n; i++){
		ll s, r;
		scanf("%lld%lld", &s, &r);
		a[s].push_back(r);
		b[s].nm++;
	}
	for (ll i = 1; i <= m; i++) b[i].id = i;
	sort(b + 1, b + m + 1, cmp);
	for (ll i = 1; i <= m; i++) c[b[i].id] = i;
	for (ll i = 1; i <= m; i++){
		ll sz = a[b[i].id].size();
		sort(a[b[i].id].begin(), a[b[i].id].end(), cmp2);
		ll s = 0;
		for (ll j = 0; j < sz; j++){
			s += a[b[i].id][j];
			ll nw = f[j] + s;
			ans = max(ans, nw);
			f[j] = max(f[j], nw);
		}
	}
	printf("%lld\n", ans);
	return 0;
}
```

## D

构造一种图，图中有一条链作为直径，链上有2个度为1的点和其他所有度数大于1的点

每个度数大于2的点下面可以挂上若干个度数为1的点

![](https://cdn.luogu.com.cn/upload/image_hosting/tbnam8sr.png)

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 505;
int n, nm1, nm2, bsz;
int b[NR];
int a[NR];
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%d", a + i);
		if (a[i] == 1){
			b[++bsz] = i;
			nm1++;
		} else nm2 += a[i] - 2;
	}
	if (nm2 < nm1 - 2){
		printf("NO\n");
		return 0;
	}
	printf("YES %d\n", n - nm1 + min(2, nm1) - 1);
	printf("%d\n", n - 1);
	int l = (bsz >= 1) ? b[bsz--] : -1;
	for (int i = 1; i <= n; i++){
		if (a[i] == 1) continue;
		if (l != -1) printf("%d %d\n", l, i);
		for (int j = 1; j <= a[i] - 2 && bsz > 1; j++){
			printf("%d %d\n", i, b[bsz--]);
		}
		l = i;
	}
	if (bsz >= 1) printf("%d %d\n", l, b[bsz--]);
	return 0;
}
```

## E

$f_i$表示$k=c-i$时区间和区间左边最大的$c$个数

遍历到一个点有两种选择，一种是把左边的区间右端点扩展到当前点，另一种是新建一个只包含当前点的区间
$$
f_{a[i]}=\max\{f_{a[i]}+1,\sum\limits_{i=1}^{i-1}[a[i]==c]+1\}
$$
统计答案时需要加上右边原来就是$c$的点

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 500051;
int n, c;
int a[MAXN], s[MAXN], f[MAXN];
int main(){
	scanf("%d%d", &n, &c);
	for (int i = 1; i <= n; i++){
		scanf("%d", a + i);
		s[i] = s[i - 1] + (a[i] == c);
	}
	int ans = 0;
	for (int i = 1; i <= n; i++){
		f[a[i]]++;
		f[a[i]] = max(f[a[i]], s[i - 1] + 1);
		ans = max(ans, f[a[i]] + s[n] - s[i]);
	}
	printf("%d\n", ans);
	return 0;
}
```

## F

在Trie树上DP，$f_{u,j,l}$​表示目前正在计算$u$​子树，$u$​最近选中的祖先深度为$j$​，可以选$l$​​个

$f'_{u,i,l}=\max\limits_{0\leq j\leq l}\{\max\{f_{u,i,j}+f_{v,i,l-j},f_{u,i,j}+f_{v,dep[v],l-j-1}\}\}$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 505;
const int MAXK = 11;
int n, k;
char s[MAXN];
struct Trie{
	int nxt[10];
} t[MAXN];
int cnt[MAXN], dep[MAXN];
int rt = 1, tsz = 1;
int f[MAXN][MAXN][MAXK], f1[MAXN][MAXN][MAXK];
void insert(int nm){
	int sz = strlen(s + 1);
	int p = rt;
	for (int i = 1; i <= sz; i++){
		if (!t[p].nxt[s[i] - '0']){
			t[p].nxt[s[i] - '0'] = ++tsz;
			dep[t[p].nxt[s[i] - '0']] = dep[p] + 1;
		}
		p = t[p].nxt[s[i] - '0'];
		cnt[p] += nm;
	}
}
void dfs(int nw){
	// printf("dfs %d %d\n", nw, cnt[nw]);
	for (int i = 0; i <= dep[nw]; i++) f[nw][i][0] = 0;
	for (int i = 0; i < 10; i++){
		int nxtn = t[nw].nxt[i];
		if (!nxtn) continue;
		dfs(nxtn);
		memset(f1[nw], 0, sizeof(f1[nw]));
		for (int j = 0; j <= dep[nw]; j++){
			for (int l = 0; l <= k; l++){
				for (int u = 0; u <= l; u++){
					f1[nw][j][l] = max(f1[nw][j][l], f[nw][j][l - u] + f[nxtn][j][u]);
					if (u >= 1) f1[nw][j][l] = max(f1[nw][j][l], f[nw][j][l - u] + f[nxtn][dep[nxtn]][u - 1] + cnt[nxtn] * (dep[nxtn] - j));
				}
			}
		}
		memcpy(f[nw], f1[nw], sizeof(f1[nw]));
		// printf("f %d:\n", nw);
		// for (int j = 0; j <= dep[nw]; j++){
		// 	for (int l = 0; l <= k; l++){
		// 		printf("%d ", f[nw][j][l]);
		// 	}
		// 	putchar('\n');
		// }
		// putchar('\n');
	}
}
int main(){
	scanf("%d%d", &n, &k);
	int ans = 0;
	for (int i = 1; i <= n; i++){
		int u;
		scanf("%s%d", s + 1, &u);
		ans += u * strlen(s + 1);
		insert(u);
	}
	dfs(rt);
	ans -= f[rt][0][k];
	printf("%d\n", ans);
	return 0;
}
```

# Educational Codeforces Round 56

## A

直接让骰子每次都骰到2，如果最终的和为奇数就骰一次3

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int t;
int x;
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d", &x);
		printf("%d\n", x / 2);
	}
	return 0;
}
```

## B

如果全是一个字母则不可能不是回文串

将整个字符串排序，如果不全是一个字母则不是回文串

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 1051;
int t;
char s[NR];
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%s", s + 1);
		int sz = strlen(s + 1);
		sort(s + 1, s + sz + 1);
		if (s[sz] == s[1]) printf("-1\n");
		else printf("%s\n", s + 1);
	}
	return 0;
}
```

## C

对于每一个$b[2*i]$，在满足所有条件的情况下使$a[i]$尽可能小，使$a[n-i-1]$尽可能大

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 200051;
int n;
ll b[NR], a[NR];
int main(){
	scanf("%d", &n);
	a[n + 1] = 1e18;
	for (int i = 1; i + i <= n; i++){
		scanf("%lld", b + i);
		a[i] = max(a[i - 1], b[i] - a[n - i + 2]);
		a[n - i + 1] = b[i] - a[i];
	}
	for (int i = 1; i <= n; i++) printf("%lld ", a[i]);
	putchar('\n');
	return 0;
}
```

## D

对图的每个联通分量进行01染色

如果在0颜色里填奇数，有$2^{cnt0}$种情况

如果在1颜色里填奇数，有$2^{cnt1}$种情况

最终把各个连通分量的情况乘起来就是答案$ans=\Pi 2^{cnt0}+2^{cnt1}$

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 300051;
const int PR = 998244353;
struct Edge{
	int t, nxt;
} g[NR * 2];
int t;
int n, m, gsz;
ll cnt1, cnt0, ans;
int fte[NR];
bool vis[NR], col[NR];
ll pwr(ll x, ll y){
	ll ans = 1;
	while (y){
		if (y & 1) ans = ans * x % PR;
		x = x * x % PR;
		y >>= 1;
	}
	return ans;
}
void addedge(int x, int y){
	g[++gsz] = (Edge){y, fte[x]};
	fte[x] = gsz;
}
bool dfs(int nw){
	cnt0 += !col[nw];
	cnt1 += col[nw];
	vis[nw] = true;
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (vis[nxtn]){
			if (col[nxtn] == col[nw]) return false;
			else continue;
		}
		col[nxtn] = col[nw] ^ 1;
		if (!dfs(nxtn)) return false;
	}
	return true;
}
int main(){
	scanf("%d", &t);
	while (t--){
		gsz = 0;
		ans = 1;
		scanf("%d%d", &n, &m);
		for (int i = 1; i <= n; i++) fte[i] = 0;
		for (int i = 1; i <= n; i++) vis[i] = false;
		for (int i = 1; i <= m; i++){
			int u, v;
			scanf("%d%d", &u, &v);
			addedge(u, v);
			addedge(v, u);
		}
		bool suc = true;
		for (int i = 1; i <= n; i++){
			if (vis[i]) continue;
			cnt1 = cnt0 = 0;
			col[i] = 1;
			if (!dfs(i)){
				suc = false;
				break;
			}
			ans = ans * (pwr(2, cnt1) + pwr(2, cnt0) % PR) % PR;
		}
		if (!suc) printf("0\n");
		else printf("%lld\n", ans);
	}
	return 0;
}
```

# Educational Codeforces Round 57

## A

直接输出$l$和$2l$，这是区间内最小的一对倍数

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int t;
int l, r;
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d%d", &l, &r);
		printf("%d %d\n", l, l * 2);
	}
	return 0;
}
```

## B

如果最左边和最右边的字符不一样，要么删左边要么删右边

如果一样，可以删除中间的字符

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 200051;
const int PR = 998244353;
ll n;
char lc, rc;
char s[NR];
int main(){
	scanf("%lld", &n);
	scanf("%s", s + 1);
	lc = s[1];
	rc = s[n];
	ll lnm = 1, rnm = 1;
	while (s[lnm] == lc) lnm++;
	while (s[n - rnm + 1] == rc) rnm++;
	ll ans;
	if (lnm == n + 1) ans = n * (n + 1) / 2;
	else if (lc == rc) ans = lnm * rnm;
	else ans = lnm + rnm - 1;
	printf("%lld\n", ans % PR);
	return 0;
}
```

## C

设$n$边形中AC两点距离为$m$条边，可以证明$\angle ABC=\frac{m}{n}180^o$

需要注意的是，当$m=n-1$时会找不到B点，需要将$n$和$m$都乘二

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int t;
int n;
int gcd(int x, int y){
	if (y == 0) return x;
	else return gcd(y, x % y);
}
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d", &n);
		int g = gcd(180, n);
		int ans = 180 / g;
		if (n / g == 180 / g - 1) ans *= 2;
		printf("%d\n", ans);
	}
	return 0;
}
```

# Educational Codeforces Round 58

## A

如果$d < l$则最小满足条件的数就是$d$本身

否则就是大于$r$的第一个$d$的倍数

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int t;
int main(){
	scanf("%d", &t);
	while (t--){
		int l, r, d;
		scanf("%d%d%d", &l, &r, &d);
		if (d >= l) printf("%d\n", (r / d + 1) * d);
		else printf("%d\n", d);
	}
	return 0;
}
```

## B

找到最左边的一组`[:`，找到最右边的一组`;]`，这样中间的`|`会最多

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 500051;
int n, ans = 4;
char s[NR];
int main(){
	scanf("%s", s + 1);
	n = strlen(s + 1);
	int p1 = 1, p2 = n;
	while (s[p1] != '[' && p1 < n) p1++;
	p1++;
	while (s[p1] != ':' && p1 < n) p1++;
	while (s[p2] != ']' && p2 > p1) p2--;
	p2--;
	while (s[p2] != ':' && p2 > p1) p2--;
	if (p1 >= p2){
		printf("-1\n");
		return 0;
	}
	for (int i = p1 + 1; i <= p2 - 1; i++){
		if (s[i] == '|') ans++;
	}
	printf("%d\n", ans);
	return 0;
}
```

## C

按照$l$从小到大排序

如果当遍历到某个区间时，之前所有区间都无法够到下一个区间，则之前的区间分一组，之后的分到另一组

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 100051;
struct Seg{
	int lft, rgt, id;
} a[NR];
int t;
int n;
int ans[NR];
bool cmp(Seg x, Seg y){
	return x.lft < y.lft;
}
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d", &n);
		for (int i = 1; i <= n; i++){
			scanf("%d%d", &a[i].lft, &a[i].rgt);
			a[i].id = i;
		}
		sort(a + 1, a + n + 1, cmp);
		int mxr = 0, p = -1;
		for (int i = 1; i < n; i++){
			mxr = max(mxr, a[i].rgt);
			if (mxr < a[i + 1].lft){
				p = i;
				break;
			}
		}
		if (p == -1){
			printf("-1\n");
			continue;
		}
		for (int i = 1; i <= p; i++) ans[a[i].id] = 1;
		for (int i = p + 1; i <= n; i++) ans[a[i].id] = 2;
		for (int i = 1; i <= n; i++) printf("%d ", ans[i]);
		putchar('\n');
	}
	return 0;
}
```

## E

把钱包横着拿，再把所有支票都横着放进去，即如果$x>y$则交换$x$和$y$

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int n;
int main(){
	scanf("%d", &n);
	int mix = 1, miy = 0;
	while (n--){
		char op;
		int x, y;
		scanf(" %c", &op);
		scanf("%d%d", &x, &y);
		if (x > y) swap(x, y);
		if (op == '+'){
			mix = max(mix, x);
			miy = max(miy, y);
		} else {
			if (x < mix || y < miy){
				printf("NO\n");
				continue;
			} else printf("YES\n");
		}
	}
	return 0;
}
```

# Educational Codeforces Round 59

## A

一共就分两组，把第一个数分到第一组，其他数分到第二组

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 305;
int t;
int n;
char s[NR];
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d", &n);
		scanf("%s", s + 1);
		if (n == 2 && s[1] >= s[2]){
			printf("NO\n");
			continue;
		}
		printf("YES\n2\n%c %s\n", s[1], s + 2);
	}
	return 0;
}
```

## B

观察到$n\equiv S(n)(\mod 9)$ 且 $S(n)\leq 9$

$S(n)=(n-1)\mod 9+1$

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int t;
int main(){
	scanf("%d", &t);
	while (t--){
		ll k;
		int x;
		scanf("%lld%d", &k, &x);
		ll ans = k * 9 + x - 9;
		printf("%lld\n", ans);
	}
	return 0;
}
```

## C

在一串连续的相同字符中选权值最大的$k$个，用堆动态维护

```cpp
# include <queue>
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 200051;
priority_queue <ll> q;
int n, k;
ll a[NR];
char s[NR];
int main(){
	scanf("%d%d", &n, &k);
	for (int i = 1; i <= n; i++) scanf("%lld", a + i);
	ll ans = 0;
	scanf("%s", s + 1);
	for (int i = 1; i <= n; i++){
		if (s[i] != s[i - 1]){
			while (!q.empty()){
				ans += -q.top();
				q.pop();
			}
			q.push(-a[i]);
		} else {
			if (q.size() < k) q.push(-a[i]);
			else if (-q.top() < a[i]){
				q.pop();
				q.push(-a[i]);
			}
		}
	}
	while (!q.empty()){
		ans += -q.top();
		q.pop();
	}
	printf("%lld\n", ans);
	return 0;
}
```

# Educational Codeforces Round 60

## A

平均值最大的子串一定是由几个连续的最大值组成的

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 100051;
int n, nw, ans, mx;
int a[NR];
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%d", a + i);
		mx = max(mx, a[i]);
	}
	for (int i = 1; i <= n; i++){
		if (a[i] != mx) nw = 0;
		else nw++;
		ans = max(ans, nw);
	}
	printf("%d\n", ans);
	return 0;
}
```

## B

先发$k$个值最大的表情，再发一个次大的表情，一直重复

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 200051;
int n;
ll m, k;
int mn1, mn2;
ll a[NR];
int main(){
	scanf("%d%lld%lld", &n, &m, &k);
	for (int i = 1; i <= n; i++){
		scanf("%lld", a + i);
		if (a[i] > a[mn1]){
			mn2 = mn1;
			mn1 = i;
		} else if (a[i] > a[mn2]){
			mn2 = i;
		}
	}
	int cnt = m / (k + 1);
	ll ans = cnt * a[mn2] + (m - cnt) * a[mn1];
	printf("%lld\n", ans);
	return 0;
}
```

## C

设$\overrightarrow{f_i}$为在前$i$天受风的影响移动的向量

$\overrightarrow{f_{i*n+j}} = \overrightarrow{f_j} + i * \overrightarrow{f_n}$

注意到如果$i$天能到终点，那么第$i+1$天也能到达终点，只要让船在最后一天逆着风走就行

对天数进行二分，如果$|\overrightarrow{targe} - (\overrightarrow{source}+\overrightarrow{f_{mid}})|\leq mid$ 则可以走到

# Educational Codeforces Round 61

## A

要使括号能够全部匹配需要满足：

1. 如果有$cnt3$,前面必须有至少一个$cnt1$

2. $cnt1$和$cnt4$的数量必须相同

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int cnt1, cnt2, cnt3, cnt4;
bool check(){
	if (cnt3 > 0 && cnt1 == 0) return false;
	return cnt4 == cnt1;
}
int main(){
	scanf("%d%d%d%d", &cnt1, &cnt2, &cnt3, &cnt4);
	if (check()) printf("1\n");
	else printf("0\n");
	return 0;
}
```

## B

每次折扣都买最贵的$q$个，第$q$贵的东西不用花钱

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 300051;
int n, m;
ll s;
ll a[NR];
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%lld", a + i);
		s += a[i];
	}
	sort(a + 1, a + n + 1);
	scanf("%d", &m);
	for (int i = 1; i <= m; i++){
		int q;
		scanf("%d", &q);
		ll ans = s - a[n - q + 1];
		printf("%lld\n", ans);
	}
	return 0;
}
```

## C

对于每个栅栏，用差分数组统计一共有多少人刷到它

用前缀和的方式记录第$i$个栅栏之前有多少个栅栏只刷过一次，有多少只刷过两次

每次暴力枚举两个人，计算重叠部分有多少只刷过两次的，单独部分有多少只刷过一次的

这些栅栏在没有这两个人之后就没人刷了

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 5051;
int n, q, ans = 1e9;
int l[NR], r[NR];
int d[NR], a[NR];
int s1[NR], s2[NR];
int main(){
	scanf("%d%d", &n, &q);
	for (int i = 1; i <= q; i++){
		scanf("%d%d", l + i, r + i);
		d[l[i]]++;
		d[r[i] + 1]--;
	}
	for (int i = 1; i <= n; i++){
		a[i] = a[i - 1] + d[i];
		s1[i] = s1[i - 1] + (a[i] == 1);
		s2[i] = s2[i - 1] + (a[i] == 2);
	}
	for (int i = 1; i <= q; i++){
		for (int j = i + 1; j <= q; j++){
			int px = i, py = j;
			if (l[px] > l[py]) swap(px, py);
			int nw;
			if (r[px] >= r[py]){
				nw = s1[r[px]] - s1[r[py]] + s2[r[py]] - s2[l[py] - 1] + s1[l[py] - 1] - s1[l[px] - 1];
			} else {
				nw = s1[r[py]] - s1[r[px]] + s2[r[px]] - s2[l[py] - 1] + s1[l[py] - 1] - s1[l[px] - 1];
			}
			// printf("ijn %d %d %d\n", i, j, nw);
			ans = min(ans, nw);
		}
	}
	printf("%d\n", n - ans);
	return 0;
}
```

# Educational Codeforces Round 62

## A

是一页一页往后翻的，**不能跳着看**

从前往后扫描，记录已经扫过的$a_i$的最大值

如果都已经看过就等到下一天再看（答案加一）

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 10051;
int n, ans;
int a[NR];
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) scanf("%d", a + i);
	int mxn = 0;
	for (int i = 1; i <= n; i++){
		mxn = max(mxn, a[i]);
		if (i >= mxn) ans++;
	}
	printf("%d\n", ans);
	return 0;
}
```

## B

一定要有一个字符能一下消掉其他所有字符

把最左边的`>`左边全删掉，或把最右边的`<`右边全删掉

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 105;
int t, n;
char s[NR];
int main(){
	scanf("%d", &t);
	while (t--){
		int mx = 0, mi = 1e9;
		scanf("%d", &n);
		scanf("%s", s + 1);
		for (int i = 1; i <= n; i++){
			if (s[i] == '>') mi = min(mi, i);
			else mx = max(mx, i);
		}
		printf("%d\n", min(mi - 1, n - mx));
	}
	return 0;
}
```

## C

首先把所有歌按照$b$降序排列

从前往后扫描，选取当前$t$前$k$大的歌，用堆维护前$k$大

```cpp
# include <queue>
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 300051;
struct Node{
	ll t, b;
} a[NR];
int n, k;
ll ans;
priority_queue <int> q;
bool cmp(Node x, Node y){
	return x.b > y.b;
}
int main(){
	scanf("%d%d", &n, &k);
	for (int i = 1; i <= n; i++) scanf("%d%d", &a[i].t, &a[i].b);
	sort(a + 1, a + n + 1, cmp);
	ll s = 0;
	for (int i = 1; i <= n; i++){
		if (q.size() < k){
			q.push(-a[i].t);
			s += a[i].t;
		} else if (a[i].t > -q.top()){
			s += a[i].t - (-q.top());
			q.pop();
			q.push(-a[i].t);
		}
		ans = max(ans, s * a[i].b);
	}
	printf("%lld\n", ans);
	return 0;
}
```

# Educational Codeforces Round 63

## A

找到两个相邻的字符，且后面字符比前面小，将这两个字符交换

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 300051;
int n;
char s[NR];
int main(){
	scanf("%d", &n);
	scanf("%s", s + 1);
	for (int i = 2; i <= n; i++){
		if (s[i - 1] > s[i]){
			printf("YES\n%d %d\n", i - 1, i);
			return 0;
		}
	}
	printf("NO\n");
	return 0;
}
```

## B

先手的策略：从前往后，只要不是8就删

后手的策略：从前往后，只要是8就删

判断前面的8能不能被删完

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 100051;
int n;
char s[NR];
int main(){
	scanf("%d", &n);
	scanf("%s", s + 1);
	int cnt = 0;
	int mv = (n - 11) / 2;
	for (int i = 1; i <= n; i++){
		if (s[i] == '8') cnt++;
		if (i - cnt > mv) break;
	}
	if (cnt > mv) printf("YES\n");
	else printf("NO\n");
	return 0;
}
```

## C

计算所有相邻时间的差的最大公约数，只要$p_i$是最大公约数的约数，就可以在每次都响

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 300051;
int n, m;
ll a[NR], p[NR];
ll gcd(ll x, ll y){
	if (y == 0) return x;
	else return gcd(y, x % y);
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++) scanf("%lld", a + i);
	for (int i = 1; i <= m; i++) scanf("%lld", p + i);
	ll g = a[2] - a[1];
	for (int i = 2; i <= n; i++) g = gcd(g, a[i] - a[i - 1]);
	for (int i = 1; i <= m; i++){
		if (g % p[i] == 0){
			printf("YES\n%lld %d\n", a[1], i);
			return 0;
		}
	}
	printf("NO\n");
	return 0;
}
```

# Educational Codeforces Round 64

## A

圆套圆，三角和正方形互相套都会出现无数个

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 105;
int n;
int a[NR];
int main(){
	scanf("%d", &n);
	int ans = 0;
	for (int i = 1; i <= n; i++) scanf("%d", a + i);
	for (int i = 2; i <= n; i++){
		if ((a[i] != 1 && a[i - 1] != 1) || (a[i] == 1 && a[i - 1] == 1)){
			printf("Infinite\n");
			return 0;
		}
		else if (max(a[i], a[i - 1]) == 2){
			ans += 3;
		} else ans += 4;
	}
	printf("Finite\n%d\n", ans);
	return 0;
}
```

## B

把所有奇数的字符放在前面，偶数的放在后面

或者把偶数放在前面，奇数放在后面

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 105;
int t, n;
char s[NR], s1[NR], s2[NR];
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%s", s + 1);
		n = strlen(s + 1);
		int sz1 = 0, sz2 = 0;
		for (int i = 1; i <= n; i++){
			if ((s[i] - 'a') & 1) s1[++sz1] = s[i];
			else s2[++sz2] = s[i];
		}
		sort(s1 + 1, s1 + sz1 + 1);
		sort(s2 + 1, s2 + sz2 + 1);
		s1[sz1 + 1] = s2[sz2 + 1] = 0;
		if (!sz1) printf("%s\n", s2 + 1);
		else if (!sz2) printf("%s\n", s1 + 1);
		else if (abs(s2[1] - s1[sz1]) != 1) printf("%s%s\n", s1 + 1, s2 + 1);
		else if (abs(s1[1] - s2[sz2]) != 1) printf("%s%s\n", s2 + 1, s1 + 1);
		else printf("No answer\n");
	}
	return 0;
}
```

# Educational Codeforces Round 65

## A

统计在前$n-k+1$个数字中有没有8

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 105;
int t, n;
char s[NR];
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d", &n);
		scanf("%s", s + 1);
		bool flag = false;
		for (int i = 1; i <= n - 10; i++){
			if (s[i] == '8'){
				flag = true;
				break;
			}
		}
		if (flag) printf("YES\n");
		else printf("NO\n");
	}
	return 0;
}
```

## B

询问(1,2) (2,3) (3,4) (4,5)，在把所有数乘起来除以前面的数就能得到5和6的乘积

在这些乘积中找23的倍数，如果两个乘积都是23的倍数，则中间的数就是23

在通过这个数把其他数推出来

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int n = 6;
int s[11], a[11];
int main(){
	for (int i = 2; i < 6; i++){
		printf("%d %d\n", i - 1, i);
		fflush(stdout);
		scanf("%d", s + i);
	}
	s[6] = 4 * 8 * 15 * 16 * 23 * 42 / s[4] / s[2];
	// for (int i = 1; i <= 6; i++) printf("si %d\n", s[i]);
	int mk = 1;
	for (int i = 2; i <= 6; i++){
		if (s[i] % 23 == 0){
			if (i == 2 && s[i + 1] % 23 != 0) continue;
			mk = i;
			break;
		}
	}
	a[mk] = 23;
	for (int i = mk - 1; i >= 1; i--) a[i] = s[i + 1] / a[i + 1];
	for (int i = mk + 1; i <= 6; i++) a[i] = s[i] / a[i - 1];
	printf("! ");
	for (int i = 1; i <= 6; i++) printf("%d ", a[i]);
	putchar('\n');
	return 0;
}
```

## C

用并查集，同时维护每个集合的大小，最后直接查询

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 500051;
int n, m;
int p[NR], sz[NR];
int findroot(int x){
	if (!p[x]) return x;
	else return p[x] = findroot(p[x]);
}
void merge(int x, int y){
	int rx = findroot(x), ry = findroot(y);
	p[ry] = rx;
	sz[rx] += sz[ry];
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++) sz[i] = 1;
	for (int i = 1; i <= m; i++){
		int k, a, b;
		scanf("%d", &k);
		if (k == 0) continue;
		k--;
		scanf("%d", &a);
		for (int j = 1; j <= k; j++){
			scanf("%d", &b);
			merge(a, b);
		}
	}
	for (int i = 1; i <= n; i++) printf("%d ", sz[findroot(i)]);
	putchar('\n');
	return 0;
}
```

## D

将每一对括号按照看到的顺序01分组

```cpp
# include <stack>
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 200051;
int n, cnt;
int c[NR];
char s[NR];
stack <int> st;
int main(){
	scanf("%d", &n);
	scanf("%s", s + 1);
	for (int i = 1; i <= n; i++){
		if (s[i] == '(') st.push(i);
		else {
			c[st.top()] = c[i] = ((++cnt) & 1);
			st.pop();
		}
	}
	for (int i = 1; i <= n; i++) printf("%d", c[i]);
	putchar('\n');
	return 0;
}
```

# Educational Codeforces Round 66

## A

每次按照题目所述操作，将$n$减成$k$的倍数，需要操作$n\mod k$次，再用$n$除以$k$，操作1次

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
int t;
ll n, k;
int main(){
	scanf("%d", &t);
	while (t--){
		ll ans = 0;
		scanf("%lld%lld", &n, &k);
		ans += n % k;
		while (n >= k){
			n /= k;
			ans += n % k + 1;
		}
		printf("%lld\n", ans);
	}
	return 0;
}
```

## B

用栈记录前$i$重循环次数的乘积，注意要对$2^{32}$取min，不然最大次数是$100^{100000}$，longlong也救不回来

加的时候直接加当前的次数，判断一下越界

```cpp
# include <stack>
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 100051;
int t;
stack <ll> st;
int main(){
	scanf("%d", &t);
	ll ans = 0;
	st.push(1);
	while (t--){
		char op[5];
		ll n;
		scanf("%s", op);
		if (op[0] == 'f'){
			scanf("%lld", &n);
			ll fn = min((1ll << 32), st.top() * n);
			st.push(fn);
		} else if (op[0] == 'e'){
			st.pop();
		} else {
			if (ans >= (1ll << 32) - st.top()){
				printf("OVERFLOW!!!\n");
				return 0;
			}
			ans += st.top();
		}
	}
	printf("%lld\n", ans);
	return 0;
}
```

## C

要使$f_k(x)$最小，取$f_k(\left\lfloor\frac{a_i+a_{i-k}}{2}\right\rfloor)=\left\lceil\frac{a_i-a_{i-k}}{2}\right\rceil$

此时$a_{i-k}-a_i$是离$x$最近的$k+1$个点，$x$取在$a_{i-k}$和$a_i$中间最小

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 200051;
int t;
int n, k;
int a[NR];
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d%d", &n, &k);
		int mi = 1e9, mn;
		for (int i = 1; i <= n; i++) scanf("%d", a + i);
		for (int i = k + 1; i <= n; i++){
			if ((a[i] - a[i - k] + 1) / 2 < mi){
				mi = (a[i] - a[i - k] + 1) / 2;
				mn = (a[i] + a[i - k]) / 2;
			}
		}
		printf("%d\n", mn);
	}
	return 0;
}
```

# Educational Codeforces Round 67（1187）

## A

前几个可能全是棍/糖
$$
ans=\max\{n-s+1,n-t+1\}
$$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
int T, n, s, t;
int main(){
	scanf("%d", &T);
	while (T--){
		scanf("%d%d%d", &n, &s, &t);
		printf("%d\n", max(n - s + 1, n - t + 1));
	}
	return 0;
}
```

## B

$p_{c,i}$表示第$i$个$c$字母出现的位置，一个人的名字中若含$i$个字母$c$，则它至少要取到第$i$个字母$c$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 200051;
int n, m;
char s0[MAXN], s1[MAXN];
int p[26][MAXN];
int cnt0[26], cnt1[26];
int main(){
	scanf("%d", &n);
	scanf("%s", s0 + 1);
	for (int i = 1; i <= n; i++){
		p[s0[i] - 'a'][++cnt0[s0[i] - 'a']] = i;
		// printf("%d %d %d\n", s0[i] - 'a', cnt0[s0[i] - 'a'], i);
	}
	scanf("%d", &m);
	while (m--){
		scanf("%s", s1 + 1);
		int sz = strlen(s1 + 1);
		int ans = 0;
		for (int i = 1; i <= sz; i++) cnt1[s1[i] - 'a']++;
		for (int i = 0; i < 26; i++) ans = max(ans, p[i][cnt1[i]]);
		printf("%d\n", ans);
		for (int i = 1; i <= sz; i++) cnt1[s1[i] - 'a']--;
	}
	return 0;
}
```

## C

用并查集将给定单调不减的区间合并，检查不是单调不减的区间是否全在一个集合里

给所有集合按照出现位置倒序赋值，一定满足要求

（这题可以$O(n\operatorname{\alpha}(n))$的）

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 1051;
int n, m, qsz;
int a[MAXN];
int ql[MAXN], qr[MAXN];
int p[MAXN];
int findroot(int nw){
	if (p[nw] == nw) return nw;
	else return p[nw] = findroot(p[nw]);
}
bool check(){
	for (int i = 1; i <= qsz; i++) if (findroot(ql[i]) == findroot(qr[i])) return false;
	return true;
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++) p[i] = i;
	for (int i = 1; i <= m; i++){
		int op, l, r;
		scanf("%d%d%d", &op, &l, &r);
		if (op == 1){
			for (int j = findroot(l); j < r; j = findroot(j + 1)) p[j] = findroot(j + 1);
		} else {
			qsz++;
			ql[qsz] = l;
			qr[qsz] = r;
		}
	}
	if (!check()) printf("NO\n");
	else {
		printf("YES\n");
		for (int i = 1; i <= n; i++){
			if (!a[findroot(i)]) a[findroot(i)] = n - i + 1;
			printf("%d ", a[findroot(i)]);
		}
		putchar('\n');
	}
	return 0;
}
```

## D

对任意区间排序等价于若干次对长度为2的区间排序

对于一个数，原来在它前面且比它小的数绝对不可能跑到它的后面去

先求出每个数最后去哪里，把这个数的值作为下标打在线段树上，线段树维护最大值

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 300051;
int T, n;
int a[MAXN], b[MAXN];
int c[MAXN], d[MAXN];
int p[MAXN];
int nm1[MAXN], nm2[MAXN];
struct SegTree{
	struct Node{
		int s;
		int ls, rs;
	} t[MAXN * 4];
	int rt, tsz;
	int newnode(){
		tsz++;
		t[tsz].s = 0;
		t[tsz].ls = t[tsz].rs = 0;
		return tsz;
	}
	void update(int nw){
		t[nw].s = max(t[t[nw].ls].s, t[t[nw].rs].s);
	}
	void modify(int &nw, int lft, int rgt, int pos, int nm){
		if (!nw) nw = newnode();
		if (lft == rgt){
			t[nw].s = max(t[nw].s, nm);
			return;
		}
		int mid = (lft + rgt) >> 1;
		if (pos <= mid) modify(t[nw].ls, lft, mid, pos, nm);
		else modify(t[nw].rs, mid + 1, rgt, pos, nm);
		update(nw);
	}
	int getsum(int nw, int lft, int rgt, int l, int r){
		if (l > r) return 0;
		if (!nw) return 0;
		if (lft == l && rgt == r) return t[nw].s;
		int mid = (lft + rgt) >> 1;
		int ans = 0;
		if (l <= mid) ans = max(ans, getsum(t[nw].ls, lft, mid, l, min(mid, r)));
		if (r >= mid + 1) ans = max(ans, getsum(t[nw].rs, mid + 1, rgt, max(mid + 1, l), r));
		return ans;
	}
} sgt;
bool cmpa(int u, int v){
	if (a[u] != a[v]) return a[u] < a[v];
	else return u < v;
}
bool cmpb(int u, int v){
	if (b[u] != b[v]) return b[u] < b[v];
	else return u < v;
}
bool solve(){
	sgt.rt = sgt.tsz = 0;
	for (int i = 1; i <= n; i++) c[i] = i;
	for (int i = 1; i <= n; i++) d[i] = i;
	sort(c + 1, c + n + 1, cmpa);
	sort(d + 1, d + n + 1, cmpb);
	for (int i = 1; i <= n; i++){
		// printf("i %d %d %d %d %d\n", i, c[i], d[i], a[c[i]], b[d[i]]);
		if (a[c[i]] != b[d[i]]) return false;
		p[c[i]] = d[i];
	}
	for (int i = 1; i <= n; i++){
		if (sgt.getsum(sgt.rt, 1, n, 1, a[i] - 1) > p[i]) return false;
		sgt.modify(sgt.rt, 1, n, a[i], p[i]);
	}
	return true;
}
int main(){
	scanf("%d", &T);
	while (T--){
		scanf("%d", &n);
		for (int i = 1; i <= n; i++) scanf("%d", a + i);
		for (int i = 1; i <= n; i++) scanf("%d", b + i);
		if (solve()) printf("YES\n");
		else printf("NO\n");
	}
	return 0;
}
```

## E

如果从$u$​开始走，则最后得到权值为$\sum\limits_v(\operatorname{dis}(u,v)+1)$，dfs所有点的时候就可以顺便维护​​

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 200051;
struct Edge{
	int t, nxt;
	Edge(int t = 0, int nxt = 0): t(t), nxt(nxt){}
} g[MAXN * 2];
int n, gsz;
int fte[MAXN];
int fa[MAXN], sz[MAXN];
ll tol, ans = 0;
void addedge(int u, int v){
	g[++gsz] = Edge(v, fte[u]);
	fte[u] = gsz;
}
void dfs0(int nw){
	sz[nw] = 1;
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (nxtn == fa[nw]) continue;
		fa[nxtn] = nw;
		dfs0(nxtn);
		sz[nw] += sz[nxtn];
	}
	tol += sz[nw];
}
void dfs(int nw){
	ans = max(ans, tol);
	for (int i = fte[nw]; i; i = g[i].nxt){
		int nxtn = g[i].t;
		if (nxtn == fa[nw]) continue;
		tol -= sz[nxtn];
		tol += n - sz[nxtn];
		dfs(nxtn);
		tol += sz[nxtn];
		tol -= n - sz[nxtn];
	}
}
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n - 1; i++){
		int u, v;
		scanf("%d%d", &u, &v);
		addedge(u, v);
		addedge(v, u);
	}
	dfs0(1);
	dfs(1);
	printf("%lld\n", ans);
	return 0;
}
```

# Educational Codeforces Round 68（1194）

## A

容易发现，最后隔一个删一个，$ans=2x$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
int t, n, x;
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d%d", &n, &x);
		printf("%lld\n", 2ll * x);
	}
	return 0;
}
```

## B

暴力检查每行每列还需要涂多少，特判一下行列交叉只用涂一次

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 400051;
int q, n, m;
char s[MAXN];
int fr[MAXN], fc[MAXN];
int main(){
	scanf("%d", &q);
	while (q--){
		scanf("%d%d", &n, &m);
		for (int i = 1; i <= n; i++) scanf("%s", s + (i - 1) * m + 1);
		memset(fr, 0, sizeof(int) * (n + 1));
		memset(fc, 0, sizeof(int) * (m + 1));
		for (int i = 1; i <= n; i++){
			for (int j = 1; j <= m; j++){
				if (s[(i - 1) * m + j] == '.'){
					fr[i]++;
					fc[j]++;
				}
			}
		}
		int ans = 1e9;
		for (int i = 1; i <= n; i++){
			for (int j = 1; j <= m; j++){
				ans = min(ans, fr[i] + fc[j] - (s[(i - 1) * m + j] == '.'));
			}
		}
		printf("%d\n", ans);
	}
	return 0;
}
```

## C

检查$s$是否是$t$的子序列，检查每种字母在$s$和$p$中数量的和是否大于在$t$中的数量

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 105;
int T;
int cnt[26];
char s[MAXN], t[MAXN], p[MAXN];
bool check(){
	memset(cnt, 0, sizeof(cnt));
	int ssz = strlen(s + 1), tsz = strlen(t + 1), psz = strlen(p + 1);
	int np = 1;
	for (int i = 1; i <= ssz; i++){
		while (np <= tsz && t[np] != s[i]) cnt[t[np++] - 'a']++;
		if (np > tsz) return false;
		np++;
	}
	// printf("np %d\n", np);
	while (np <= tsz) cnt[t[np++] - 'a']++;
	for (int i = 1; i <= psz; i++){
		cnt[p[i] - 'a']--;
		// printf("cnt %d %d\n", i, cnt[p[i] - 'a']);
	}
	for (int i = 0; i < 26; i++) if (cnt[i] > 0) return false;
	return true;
}
int main(){
	scanf("%d", &T);
	while (T--){
		scanf("%s%s%s", s + 1, t + 1, p + 1);
		if (check()) printf("Yes\n");
		else printf("No\n");
	}
	return 0;
}
```

## D

观察SG函数发现如果$3\nmid k$就没有影响

如果$3\mid k$，则SG函数就类似这样：0XX0XX...0XXX 0XX0XX...0XXX......，其中X代表非0数

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
int T, n, k;
bool solve(){
	if (k % 3 != 0) return n % 3 != 0;
	else return n % (k + 1) == k || n % (k + 1) % 3 != 0;
}
int main(){
	scanf("%d", &T);
	while (T--){
		scanf("%d%d", &n, &k);
		if (solve()) printf("Alice\n");
		else printf("Bob\n");
	}
	return 0;
}
```

# Educational Codeforces Round 69（1197）

## A

所有木棍中最长的两个用作竖着的棍

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 100051;
int t, n;
int a[MAXN];
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d", &n);
		for (int i = 1; i <= n; i++) scanf("%d", a + i);
		sort(a + 1, a + n + 1);
		printf("%d\n", min(a[n - 1] - 1, n - 2));
	}
	return 0;
}
```

## B

题目条件等价于序列单峰，先升后降

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 200051;
int n;
int a[MAXN];
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) scanf("%d", a + i);
	for (int i = 2; i <= n - 1; i++){
		if (a[i] < a[i - 1] && a[i] < a[i + 1]){
			printf("NO\n");
			return 0;
		}
	}
	printf("YES\n");
	return 0;
}
```

## C

如果在$i$和$i-1$中间截断，答案会减少$a_i-a_{i-1}$​，取最大的$k-1$个断点

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 300051;
int n, k;
int a[MAXN], b[MAXN];
int main(){
	scanf("%d%d", &n, &k);
	for (int i = 1; i <= n; i++){
		scanf("%d", a + i);
		b[i] = a[i] - a[i - 1];
	}
	int ans = a[n] - a[1];
	sort(b + 2, b + n + 1);
	for (int i = 1; i <= k - 1; i++) ans -= b[n - i + 1];
	printf("%d\n", ans);
	return 0;
}
```

## D

令$f_i=\max\limits_{1\leq j<i}\{-s_j-k\left\lceil\frac{i-j}m\right\rceil\}$，$f_i$的值可以从$f_{i-m}$​地推
$$
f_i=\max\{f_{i-m},\max\limits_{i-m\leq j< i}\{-s_j-k\left\lceil\frac{i-j}m\right\rceil\}\}
$$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 300051;
int n, m, k;
int a[MAXN];
ll s[MAXN], f[MAXN];
int main(){
	scanf("%d%d%d", &n, &m, &k);
	for (int i = 1; i <= n; i++){
		scanf("%d", a + i);
		s[i] = s[i - 1] + a[i];
	}
	ll ans = 0;
	for (int i = 1; i <= n; i++){
		f[i] = -1e18;
		if (i > m) f[i] = f[i - m] - k;
		for (int j = 1; j <= m; j++) f[i] = max(f[i], -s[i - j] - 1ll * k * ((j + m - 1) / m));
		// printf("f %d %lld\n", i, f[i]);
		ans = max(ans, s[i] + f[i]);
	}
	printf("%lld\n", ans);
	return 0;
}
```

## E

令$f_i$为以第$i$个为最外部套娃的情况下最少浪费的空间，$g_i$为在浪费空间最少的情况下的方案数

按照内部空间从小到大枚举点，此时转移集合一定是所有套娃按照外部空间从小到大排序后的一个前缀，且转移集合大小单调不减

可以均摊$O(n)$​维护转移集合中的最小值，方案数

如果一个方案可以再在内部放套娃，则一定不优，只用判最外层是否能再套一层就能满足要求

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 200051;
const int P = 1000000007;
int n;
int au[MAXN], av[MAXN];
int a[MAXN], b[MAXN];
int f[MAXN], f2[MAXN];
bool cmpu(int u, int v){
	return au[u] < au[v];
}
bool cmpv(int u, int v){
	return av[u] < av[v];
}
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) scanf("%d%d", au + i, av + i);
	for (int i = 1; i <= n; i++) a[i] = i;
	sort(a + 1, a + n + 1, cmpv);
	for (int i = 1; i <= n; i++) b[i] = i;
	sort(b + 1, b + n + 1, cmpu);
	int p = 1, mi = 0;
	f2[0] = 1;
	for (int i = 1; i <= n; i++){
		f2[a[i]] = f2[a[i - 1]];
		while (p <= n && au[b[p]] <= av[a[i]]){
			if (f[b[p]] + av[a[i]] - au[b[p]] == f[mi] + av[a[i]] - au[mi]) f2[a[i]] = (f2[a[i]] + f2[b[p]]) % P;
			if (f[b[p]] + av[a[i]] - au[b[p]] < f[mi] + av[a[i]] - au[mi]) f2[a[i]] = f2[b[p]];
			if (f[b[p]] + av[a[i]] - au[b[p]] <= f[mi] + av[a[i]] - au[mi]) mi = b[p];
			p++;
		}
		f[a[i]] = f[mi] + av[a[i]] - au[mi];
		// printf("f %d %d\n", a[i], mi);
	}
	mi = 1e9;
	for (int i = 1; i <= n; i++) mi = min(mi, f[i]);
	int ans = 0;
	for (int i = 1; i <= n; i++){
		if (f[i] == mi && au[i] > av[a[n]]) ans = (ans + f2[i]) % P;
		// printf("f %d %d %d\n", i, f[i], f2[i]);
	}
	printf("%d\n", ans);
	return 0;
}
```

# Educational Codeforces Round 70（1202）

## A

让$s_x$的最后一位一恰好进位一，字典序最小

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 100051;
int t;
char s1[MAXN], s2[MAXN];
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%s%s", s1 + 1, s2 + 1);
		int sz1 = strlen(s1 + 1), sz2 = strlen(s2 + 1);
		int pos = 1;
		for (int i = 1; i <= sz2; i++){
			if (s2[sz2 - i + 1] == '1'){
				pos = i;
				break;
			}
		}
		int ans = 0;
		for (int i = 1; pos + i - 1 <= sz1; i++){
			if (s1[sz1 - pos - i + 2] == '1'){
				ans = i - 1;
				break;
			}
		}
		printf("%d\n", ans);
	}
	return 0;
}
```

## B

记录原序列中相邻两个数的10*10种情况各出现了多少次，枚举$x$和$y$，用floyd求出任两个数字之间需要添加几个

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 2000051;
const int INF = 0x3f3f3f3f;
int n;
char s[MAXN];
int cnt[10][10];
int f[10][10];
int solve(int u, int v){
	memset(f, 0x3f, sizeof(f));
	for (int i = 0; i < 10; i++) f[i][(i + u) % 10] = f[i][(i + v) % 10] = 1;
	for (int k = 0; k < 10; k++){
		for (int i = 0; i < 10; i++){
			for (int j = 0; j < 10; j++){
				f[i][j] = min(f[i][j], f[i][k] + f[k][j]);
			}
		}
	}
	int ans = 0;
	for (int i = 0; i < 10; i++){
		for (int j = 0; j < 10; j++){
			if (f[i][j] == INF && cnt[i][j] > 0) return -1;
			ans += (f[i][j] - 1) * cnt[i][j];
		}
	}
	return ans;
}
int main(){
	scanf("%s", s + 1);
	n = strlen(s + 1);
	for (int i = 2; i <= n; i++) cnt[s[i - 1] - '0'][s[i] - '0']++;
	for (int i = 0; i < 10; i++){
		for (int j = 0; j < 10; j++) printf("%d ", solve(i, j));
		putchar('\n');
	}
	return 0;
}
```

## C

把序列分成两半，分别记录前缀和后缀的信息，枚举分割点，$O(1)$算出当前答案

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 200051;
const int mv[5][2] = {{0, 1}, {-1, 0}, {0, -1}, {1, 0}, {0, 0}};
const int ch[4] = {'W', 'A', 'S', 'D'};
int t, n;
char s[MAXN];
int mxu1[MAXN], miu1[MAXN], mxv1[MAXN], miv1[MAXN];
int mxu2[MAXN], miu2[MAXN], mxv2[MAXN], miv2[MAXN];
ll calc(int u){
	ll ans = 1e18;
	for (int i = 0; i <= 4; i++){
		ans = min(ans, 1ll * (max(mxu1[u], mxu2[u + 1] + mv[i][0]) - min(miu1[u], miu2[u + 1] + mv[i][0]) + 1)
						   * (max(mxv1[u], mxv2[u + 1] + mv[i][1]) - min(miv1[u], miv2[u + 1] + mv[i][1]) + 1));
	}
	// printf("calc %d %lld\n", u, ans);
	return ans;
}
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%s", s + 1);
		n = strlen(s + 1);
		for (int i = 1; i <= n; i++){
			for (int j = 0; j < 4; j++){
				if (s[i] == ch[j]){
					mxu1[i] = mxu1[i - 1] - mv[j][0];
					miu1[i] = miu1[i - 1] - mv[j][0];
					mxv1[i] = mxv1[i - 1] - mv[j][1];
					miv1[i] = miv1[i - 1] - mv[j][1];
				}
			}
			mxu1[i] = max(mxu1[i], 0);
			miu1[i] = min(miu1[i], 0);
			mxv1[i] = max(mxv1[i], 0);
			miv1[i] = min(miv1[i], 0);
		}
		mxu2[n + 1] = miu2[n + 1] = mxv2[n + 1] = miv2[n + 1] = 0;
		for (int i = n; i >= 1; i--){
			for (int j = 0; j < 4; j++){
				if (s[i] == ch[j]){
					mxu2[i] = mxu2[i + 1] + mv[j][0];
					miu2[i] = miu2[i + 1] + mv[j][0];
					mxv2[i] = mxv2[i + 1] + mv[j][1];
					miv2[i] = miv2[i + 1] + mv[j][1];
				}
			}
			mxu2[i] = max(mxu2[i], 0);
			miu2[i] = min(miu2[i], 0);
			mxv2[i] = max(mxv2[i], 0);
			miv2[i] = min(miv2[i], 0);
		}
		// for (int i = 1; i <= n; i++){
		// 	printf("f %d %d %d %d %d\n", i, mxu1[i], miu1[i], mxv1[i], miv1[i]);
		// }
		// for (int i = 1; i <= n; i++){
		// 	printf("g %d %d %d %d %d\n", i, mxu2[i], miu2[i], mxv2[i], miv2[i]);
		// }
		ll ans = 1e18;
		for (int i = 0; i <= n; i++) ans = min(ans, calc(i));
		printf("%lld\n", ans);
	}
	return 0;
}
```

## D

将$n$分成若干个$\begin{pmatrix}k\\2\end{pmatrix}$相加的形式$n=\sum\limits_{j}\begin{pmatrix}a_j\\2\end{pmatrix}$，如果每次贪心选最大的$k$，$n-\begin{pmatrix}k\\2\end{pmatrix}\leq k$，只需$O(\log n)$次就能把$n$分完

构造序列13...373...37......，其中第$i$个7前面有$a_i$个3

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 100051;
int t, n, cnt;
int a[MAXN];
char s[MAXN];
int work(int nm){
	ll lft = 2, rgt = 100000, ans = 1;
	while (lft <= rgt){
		ll mid = (lft + rgt) >> 1;
		if (mid * (mid - 1) / 2 <= nm){
			ans = mid;
			lft = mid + 1;
		} else rgt = mid - 1;
	}
	return ans;
}
int main(){
	scanf("%d", &t);
	while (t--){
		scanf("%d", &n);
		cnt = 0;
		while (n > 0){
			a[++cnt] = work(n);
			n -= a[cnt] * (a[cnt] - 1) / 2;
		}
		memset(s, '3', sizeof(s));
		s[1] = '1';
		for (int i = 1; i <= cnt; i++){
			s[a[cnt - i + 1] + i + 1] = '7';
		}
		s[a[1] + cnt + 2] = 0;
		printf("%s\n", s + 1);
	}
	return 0;
}
```

## E

预处理$t$的每个前缀和后缀能匹配多少$s$​，枚举分割点

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 200051;
struct ACAM{
	struct Node{
		int nm;
		int fail;
		int nxt[26];
	} t[MAXN];
	int rt0, rt1, tsz;
	ACAM(){
		tsz = 0;
		rt0 = ++tsz;
		rt1 = ++tsz;
		for (int i = 0; i < 26; i++) t[rt0].nxt[i] = rt1;
		t[rt0].fail = t[rt1].fail = rt0;
	}
	void insert(char s[], int sz){
		int p = rt1;
		for (int i = 1; i <= sz; i++){
			if (!t[p].nxt[s[i] - 'a']) t[p].nxt[s[i] - 'a'] = ++tsz;
			p = t[p].nxt[s[i] - 'a'];
		}
		t[p].nm++;
	}
	void getfail(){
		queue <int> q;
		q.push(rt1);
		while (!q.empty()){
			int nw = q.front();
			q.pop();
			for (int i = 0; i < 26; i++){
				if (!t[nw].nxt[i]) t[nw].nxt[i] = t[t[nw].fail].nxt[i];
				else {
					t[t[nw].nxt[i]].fail = t[t[nw].fail].nxt[i];
					t[t[nw].nxt[i]].nm += t[t[t[nw].nxt[i]].fail].nm;
					q.push(t[nw].nxt[i]);
				}
			}
		}
	}
} t1, t2;
int n;
char s0[MAXN];
char s[MAXN];
int f1[MAXN], f2[MAXN];
int main(){
	scanf("%s", s0 + 1);
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%s", s + 1);
		int sz = strlen(s + 1);
		t1.insert(s, sz);
		for (int j = 1; j < sz - j + 1; j++) swap(s[j], s[sz - j + 1]);
		t2.insert(s, sz);
	}
	t1.getfail();
	t2.getfail();
	int sz = strlen(s0 + 1);
	int p = t1.rt1;
	for (int i = 1; i <= sz; i++){
		p = t1.t[p].nxt[s0[i] - 'a'];
		f1[i] = t1.t[p].nm;
	}
	p = t2.rt1;
	for (int i = sz; i >= 1; i--){
		p = t2.t[p].nxt[s0[i] - 'a'];
		f2[i] = t2.t[p].nm;
	}
	ll ans = 0;
	for (int i = 1; i <= sz; i++) ans += 1ll * f1[i] * f2[i + 1];
	printf("%lld\n", ans);
	return 0;
}
```

