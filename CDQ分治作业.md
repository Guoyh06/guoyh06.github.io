# CDQ分治 作业

## P1908 逆序对

给定长度为$n$的序列$a$，求序列中逆序对个数

一个逆序对$(i,j)$满足
$$
\begin{cases}
i<j\\
a_i>a_j
\end{cases}
$$
用CDQ分治解决第一维，剩下只需要做一维偏序就行了

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int MAXN = 500051;
struct Node{
	int t, val, id;
} q[MAXN], q1[MAXN];
int n;
int ans[MAXN];
bool cmpVal(Node x, Node y){
	return x.val > y.val;
}
bool cmpT(Node x, Node y){
	return x.t < y.t;
}
void solve(int lft, int rgt){
	if (lft == rgt) return;
	int mid = (lft + rgt) >> 1;
	solve(lft, mid);
	solve(mid + 1, rgt);
	int p = lft, cnt = 0;
	for (int i = mid + 1; i <= rgt; i++){
		while (p <= mid && q[p].val > q[i].val){
			cnt++;
			p++;
		}
		ans[q[i].id] += cnt;
	}
	merge(q + lft, q + mid + 1, q + mid + 1, q + rgt + 1, q1 + lft, cmpVal);
	memcpy(q + lft, q1 + lft, (rgt - lft + 1) * sizeof(Node));
}
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%d", &q[i].val);
		q[i].t = q[i].id = i;
	}
	sort(q + 1, q + n + 1, cmpT);
	solve(1, n);
	ll tol = 0;
	for (int i = 1; i <= n; i++) tol += ans[i];
	printf("%lld\n", tol);
	return 0;
}
```

## P4390 [BOI2007]Mokia 摩基亚

维护一个$w\times w$的矩阵，支持单点加，查询任意矩形的和

用CDQ分治处理掉时间轴，剩下的就是二维数点问题

对于一个矩形$((x_1,y_1),(x_2,y_2))$，可以分成两部分处理，用$((0,y_1),(x_2,y_2))$减去$((0,y1),(x_2,y_2))$，这样就可以用树状数组维护

![](https://cdn.luogu.com.cn/upload/image_hosting/r1lrpqv4.png)

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
# define lowbit(x) ((x) & (-(x)))
using namespace std;
const int MAXM = 200051;
const int MAXN = 2000051;
struct Node{
	int op, t, u, v1, v2, val, id;
} q[MAXM * 2], q1[MAXM * 2];
struct BIT{
	int sz;
	int c[MAXN];
	void init(int n){
		sz = n;
	}
	void modify(int pos, int nm){
		for (int i = pos; i <= sz; i += lowbit(i)) c[i] += nm;
	}
	int getsum(int pos){
		int ans = 0;
		for (int i = pos; i; i -= lowbit(i)) ans += c[i];
		return ans;
	}
} t;
int n;
int ans[MAXM];
bool cmpU(Node x, Node y){
	return x.u < y.u;
}
bool cmpT(Node x, Node y){
	return x.t < y.t;
}
void solve(int lft, int rgt){
	if (lft == rgt) return;
	int mid = (lft + rgt) >> 1;
	solve(lft, mid);
	solve(mid + 1, rgt);
	int p = lft;
	for (int i = mid + 1; i <= rgt; i++){
		if (q[i].op == 0) continue;
		while (p <= mid && q[p].u <= q[i].u){
			t.modify(q[p].v1, q[p].val);
			p++;
		}
		ans[q[i].id] += q[i].op * (t.getsum(q[i].v2) - t.getsum(q[i].v1 - 1));
	}
	for (int i = lft; i < p; i++) t.modify(q[i].v1, -q[i].val);
	merge(q + lft, q + mid + 1, q + mid + 1, q + rgt + 1, q1 + lft, cmpU);
	memcpy(q + lft, q1 + lft, (rgt - lft + 1) * sizeof(Node));
}
int main(){
	scanf("%*d%d", &n);
	int op, m = 0, cnt = 0;
	while (scanf("%d", &op) == 1){
		if (op == 3) break;
		if (op == 1){
			m++;
			scanf("%d%d%d", &q[m].u, &q[m].v1, &q[m].val);
		} else {
			m++;
			scanf("%d%d%d%d", &q[m].u, &q[m].v1, &q[m + 1].u, &q[m].v2);
			q[m].u--;
			q[m + 1].v1 = q[m].v1;
			q[m + 1].v2 = q[m].v2;
			q[m].op = -1;
			q[m + 1].op = 1;
			q[m].id = q[m + 1].id = ++cnt;
			m++;
		}
	}
	t.init(n);
	solve(1, m);
	for (int i = 1; i <= cnt; i++) printf("%d\n", ans[i]);
	return 0;
}
```

## P3810 【模板】三维偏序（陌上花开）

三维偏序模板题，给定三个长度为$n$的序列$a$、$b$、$c$，求满足$\begin{cases}a_i<a_j\\b_i<b_j\\c_i<c_j\\i\neq j\end{cases}$ 的$(i,j)$个数

用CDQ分治处理掉一维，剩下二维用扫描线二维数点

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
using namespace std;
const int NR = 100051;
struct Node{
	int op, a, b, c, id;
} arr[NR << 1];
int n, k;
int lowbit(int x){
	return x & (-x);
}
struct Tree{
	int c[NR << 1];
	void add(int x, int y){
		for (int i = x; i <= k; i += lowbit(i)) c[i] += y;
	}
	int getsum(int x){
		int fnl = 0;
		for (int i = x; i; i -= lowbit(i)) fnl += c[i];
		return fnl;
	}
} tree;
int ans[NR], nm[NR];
inline int read(){
	int nm = 0;
	char x = getchar();
	while (x < '0' || x > '9') x = getchar();
	while (x >= '0' && x <= '9'){
		nm = nm * 10 + x - '0';
		x = getchar();
	}
	return nm;
}
inline void write(int x){
	int l = 0;
	char c[31];
	while (x){
		c[++l] = x % 10 + '0';
		x /= 10;
	}
	if (l == 0) putchar('0');
	else for (int i = l; i >= 1; i--) putchar(c[i]);
	putchar('\n');
}
bool cmpa(Node x, Node y){
	if (x.a != y.a) return x.a < y.a;
	else return x.op < y.op;
}
bool cmpb(Node x, Node y){
	return x.b < y.b;
}
void solve(int l, int r){
	if (l == r) return;
	int mid = (l + r) >> 1;
	solve(l, mid);
	solve(mid + 1, r);
	sort(arr + l, arr + mid + 1, cmpb);
	sort(arr + mid + 1, arr + r + 1, cmpb);
//	printf("lr %d %d\n", l, r);
//	for (int i = l; i <= r; i++) printf("%d %d %d\n", arr[i].a, arr[i].b, arr[i].c);
	int j = l;
	for (int i = mid + 1; i <= r; i++){
		if (arr[i].op != 2) continue;
		for (; arr[j].b <= arr[i].b && j <= mid; j++)
			if (arr[j].op == 1) tree.add(arr[j].c, 1);
		ans[arr[i].id] += tree.getsum(arr[i].c);
//		printf("igs %d %d %d\n", i, arr[i].id, tree.getsum(arr[i].c));
	}
	for (int i = l; i < j; i++)
		if (arr[i].op == 1) tree.add(arr[i].c, -1);
//	putchar('\n');
}
int main(){
	n = read();
	k = read();
	for (int i = 1; i <= n; i++){
		arr[i].a = read();
		arr[i].b = read();
		arr[i].c = read();
		arr[i].id = i;
		arr[i + n] = arr[i];
		arr[i].op = 1;
		arr[i + n].op = 2;
	}
	sort(arr + 1, arr + n * 2 + 1, cmpa);
	solve(1, n * 2);
	for (int i = 1; i <= n; i++) nm[ans[i]]++;
	for (int i = 1; i <= n; i++) write(nm[i]);
	return 0;
}
```

## P4093 [HEOI2016/TJOI2016]序列

给定长为$n$的序列$a$，且有$m$种变化$(x_i,y_i)$表示$a_{x_i}$可能变成$y_i$，求一个最长的子序列，使得在至多发生一种变化时，子序列单调不降

统计序列中每个元素可能取到的最大值，最小值，$a_i$的最大值记为$b_i$，最小值记为$c_i$

考虑dp，$f_i$表示以$a_i$为结尾的最长子序列长度，$f_i=\max_{\begin{cases}j<i\\b_j\leq a_i\\a_j\leq c_i\end{cases}}\{f_j\}+1$

通过CDQ分治三维数点可以快速求出所有满足条件的 $j$以及$\max{f_j}$

```cpp
# include <cstdio>
# include <cstring>
# include <algorithm>
# define ll long long
# define lowbit(x) ((x) & (-(x)))
using namespace std;
const int MAXN = 100051;
struct Node{
	int a, b, c, id;
} q[MAXN * 2], q1[MAXN * 2];
struct BIT{
	int c[MAXN];
	void modify(int pos, int nm){
		for (int i = pos; i < MAXN; i += lowbit(i)) c[i] = max(c[i], nm);
	}
	void clear(int pos){
		for (int i = pos; i < MAXN; i += lowbit(i)) c[i] = 0;
	}
	int getsum(int pos){
		int ans = 0;
		for (int i = pos; i; i -= lowbit(i)) ans = max(ans, c[i]);
		return ans;
	}
} t;
int n, m;
int f[MAXN];
bool cmpA(Node x, Node y){
	return x.a < y.a;
}
bool cmpB(Node x, Node y){
	return x.b < y.b;
}
void solve(int lft, int rgt){
	if (lft == rgt) return;
	int mid = (lft + rgt) >> 1;
	solve(lft, mid);
	memcpy(q1 + mid + 1, q + mid + 1, sizeof(Node) * (rgt - mid));
	sort(q + lft, q + mid + 1, cmpB);
	sort(q + mid + 1, q + rgt + 1, cmpA);
	int p = lft;
	for (int i = mid + 1; i <= rgt; i++){
		while (p <= mid && q[p].b <= q[i].a){
			t.modify(q[p].a, f[q[p].id]);
			p++;
		}
		f[q[i].id] = max(f[q[i].id], t.getsum(q[i].c) + 1);
		// printf("f %d %d\n", q[i].id, f[q[i].id]);
	}
	for (int i = lft; i < p; i++) t.clear(q[i].a);
	memcpy(q + mid + 1, q1 + mid + 1, sizeof(Node) * (rgt - mid));
	solve(mid + 1, rgt);
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++){
		scanf("%d", &q[i].a);
		q[i].b = q[i].c = q[i].a;
		q[i].id = i;
	}
	for (int i = 1; i <= m; i++){
		int u, v;
		scanf("%d%d", &u, &v);
		q[u].b = max(q[u].b, v);
		q[u].c = min(q[u].c, v);
	}
	for (int i = 1; i <= n; i++) f[i] = 1;
	solve(1, n);
	int ans = 0;
	for (int i = 1; i <= n; i++) ans = max(ans, f[i]);
	printf("%d\n", ans);
	return 0;
}
```

