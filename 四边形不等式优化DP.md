#  四边形不等式优化DP 作业

### P3515 [POI2011]Lightning Conductor

$$
\forall j\in [1,i-1],-a_i\geq -a_j-\sqrt{i-j}+p
$$

$$
\forall j\in[i+1,n],-a_i\geq -a_j-\sqrt{j-i}+p
$$

$$
\begin{aligned}
&w(l,r)+w(l+1,r+1)\leq w(l+1,r)+w(l,r+1)\\
\Leftarrow&-2\sqrt{r-l}\leq -\sqrt{r-l-1}-\sqrt{r-l+1}\\
\Leftarrow&2\sqrt{r-l}\geq\sqrt{r-l-1}+\sqrt{r-l+1}
\end{aligned}
$$

$f(x)=\sqrt{x}$为上凸函数，得证

$w(i,j)=-\sqrt{i-j}$满足四边形不等式，有决策单调性

注意：

- $w(i,j)=-\sqrt{i-j}$满足四边形不等式，$w(i,j)=\lfloor-\sqrt{i-j}\rfloor$并不满足，应该先用小数计算，最后再转成整型
- 两个决策点的权值相同时，应采用新的决策点，因为新的决策点仍可能在前面的位置严格优于当前的决策点

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 500051;
struct Node{
	int pos, id;
	Node(int pos = 0, int id = 0): pos(pos), id(id){}
};
int n;
int a[MAXN], f[MAXN];
Node q[MAXN];
int ql = 1, qr;
double getval(int pos, int id){
	return a[id] - a[pos] + sqrt(abs(pos - id));
}
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%d", a + i);
//		a[i] = 14;
	}
	for (int i = 1; i <= n; i++){
		while (qr > ql && q[ql + 1].pos <= i) ql++;
		if (qr >= ql) f[i] = max(f[i], int(ceil(getval(i, q[ql].id)) + 1e-8));
		// printf("f %d %d %d\n", i, f[i], q[ql].id);
		while (qr > ql && getval(q[qr].pos, q[qr].id) <= getval(q[qr].pos, i)) qr--;
		if (qr < ql) q[++qr] = Node(i, i);
		else {
			int lft = max(i + 1, q[qr].pos), rgt = n, ans = n + 1;
			while (lft <= rgt){
				int mid = (lft + rgt) >> 1;
				if (getval(mid, i) >= getval(mid, q[qr].id)){
					ans = mid;
					rgt = mid - 1;
				} else lft = mid + 1;
			}
			if (ans <= n) q[++qr] = Node(ans, i);
		}
		// for (int i = ql; i <= qr; i++) printf("%d %d\n", q[i].pos, q[i].id);
	}
	ql = 1;
	qr = 0;
	for (int i = n; i >= 1; i--){
		while (qr > ql && q[ql + 1].pos >= i) ql++;
		if (qr >= ql) f[i] = max(f[i], int(ceil(getval(i, q[ql].id)) + 1e-8));
		// printf("f %d %d %d\n", i, f[i], q[ql].id);
		while (qr > ql && getval(q[qr].pos, q[qr].id) <= getval(q[qr].pos, i)) qr--;
		if (qr < ql){
			q[++qr] = Node(i, i);
			continue;
		}
		int lft = 1, rgt = min(q[qr].pos, i - 1), ans = 0;
		while (lft <= rgt){
			int mid = (lft + rgt) >> 1;
			if (getval(mid, i) >= getval(mid, q[qr].id)){
				// printf("getval %d %d %d %f %f\n", mid, i, q[qr].id, getval(mid, i), getval(mid, q[qr].id));
				ans = mid;
				lft = mid + 1;
			} else rgt = mid - 1;
		}
		if (ans >= 1) q[++qr] = Node(ans, i);
		// for (int i = ql; i <= qr; i++) printf("q %d %d\n", q[i].pos, q[i].id);
	}
	for (int i = 1; i <= n; i++) printf("%d\n", f[i]);
	return 0;
}
```

### P1912 [NOI2009]诗人小G

令$f_i$表示只考虑前$i$句时最小的不协调度
$$
f_i=\max\limits_{0\leq j<i}\{f_j+|s_i-s_j+(i-j-1)-L|^P\}
$$
考虑函数$f(x)=|x|^P$，$P\geq 1$函数为下凸函数
$$
\begin{aligned}
&w(l,r+1)+w(l+1,r)\geq w(l,r)+w(l+1,r+1)\\
\Leftarrow&f(s_{r+1}-s_l+r-l-L)+f(s_r-s_{l+1}+r-l-2-L)\geq f(s_r-s_l+r-l-1-L)+f(s_{r+1}-s_{l+1}+r-l-1-L)\\
\Leftarrow&f(A+a_{r+1}+1)+f(A-a_{l+1}-1)\geq f(A)+f(A+a_{r+1}-a_{l+1}),A=s_r-s_l+r-l-1-L\\
\end{aligned}
$$
$w(l,r)$满足四边形不等式

```cpp
# include <bits/stdc++.h>
# define ll long long
# define ld __float128
using namespace std;
const int MAXN = 100051;
const int MAXL = 3000051;
struct Node{
	int pos, id;
	Node(int pos = 0, int id = 0): pos(pos), id(id){}
};
int t, N, L, P;
int l[MAXN], sl[MAXN];
char s[MAXN][35];
ld f[MAXN];
int pre[MAXN];
Node q[MAXN];
int ql = 1, qr;
ld pwr(ld x, int y){
	ld ans = 1;
	while (y){
		if (y & 1) ans = ans * x;
		x = x * x;
		y >>= 1;
	}
	return ans;
}
ll pwr1(ll x, int y){
	ll ans = 1;
	while (y){
		if (y & 1) ans = ans * x;
		x = x * x;
		y >>= 1;
	}
	return ans;
}
ld calc(int nw, int pr){
	return pwr(abs(sl[nw] - sl[pr] + nw - pr - 1 - L), P) + f[pr];
}
void prt(int nw, ll ans){
	if (nw == 0){
		printf("%lld\n", ans);
		return;
	}
	prt(pre[nw], ans + pwr1(abs(sl[nw] - sl[pre[nw]] + nw - pre[nw] - 1 - L), P));
	for (int i = pre[nw] + 1; i < nw; i++) printf("%s ", s[i]);
	printf("%s\n", s[nw]);
}
int main(){
	scanf("%d", &t);
	while (t--){
		ql = 1;
		qr = 0;
		scanf("%d%d%d", &N, &L, &P);
		for (int i = 1; i <= N; i++){
			scanf("%s", s[i]);
			l[i] = strlen(s[i]);
			sl[i] = sl[i - 1] + l[i];
		}
		f[0] = 0;
		q[++qr] = Node(1, 0);
		for (int i = 1; i <= N; i++){
			while (qr > ql && q[ql + 1].pos <= i) ql++;
			f[i] = calc(i, q[ql].id);
			pre[i] = q[ql].id;
			// printf("f %d %Lf\n", i, f[i]);
			while (qr > ql && calc(q[qr].pos, q[qr].id) > calc(q[qr].pos, i)) qr--;
			int lft = max(i + 1, q[qr].pos), rgt = N, ans = N + 1;
			while (lft <= rgt){
				int mid = (lft + rgt) >> 1;
				if (calc(mid, q[qr].id) > calc(mid, i)){
					ans = mid;
					rgt = mid - 1;
				} else lft = mid + 1;
			}
			if (ans <= N) q[++qr] = Node(ans, i);
			// for (int i = ql; i <= qr; i++){
			// 	printf("%d %d\n", q[i].pos, q[i].id);
			// }
		}
		if (f[N] > 1e18) puts("Too hard to arrange");
		else prt(N, 0);
		puts("--------------------");
	}
	return 0;
}
```

### AcWing 282. 石子合并

$$
f_{l,r}=\max\limits_{l\leq k<r}\{f_{l,k}+f_{k+1,r}+sum(l,r)\}
$$

$w(l,r)=sum(l,r)$满足$\forall a\leq b\leq c\leq d,w(a,d)\geq w(b,c), w(a,b)+w(b,c)\geq w(a,c)+w(b,d)$

对于每个$f_{l,r}$，其决策点$d_{l,r}$满足$d_{l,r-1}\leq d_{l,r}\leq d_{l+1,r}$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 305;
int n;
int a[MAXN], s[MAXN];
int f[MAXN][MAXN], d[MAXN][MAXN];
int main(){
	scanf("%d", &n);
	for (int i = 1; i <= n; i++){
		scanf("%d", a + i);
		s[i] = s[i - 1] + a[i];
	}
	for (int l = n; l >= 1; l--){
		f[l][l] = 0;
		d[l][l] = l;
		f[l][l + 1] = s[l + 1] - s[l - 1];
		d[l][l + 1] = l;
		for (int r = l + 2; r <= n; r++){
			f[l][r] = 1e9;
			for (int i = d[l][r - 1]; i <= d[l + 1][r]; i++){
				if (f[l][r] >= f[l][i] + f[i + 1][r]){
					f[l][r] = min(f[l][r], f[l][i] + f[i + 1][r]);
					d[l][r] = i;
				}
			}
			f[l][r] += s[r] - s[l - 1];
			// printf("f %d %d %d %d\n", l, r, f[l][r], d[l][r]);
		}
	}
	printf("%d\n", f[1][n]);
	return 0;
}
```

### AcWing 336. 邮局

$f_{i,j}$表示前$i$个村庄建了$j$个邮局的距离和

枚举最后一个邮局控制的一段$f_{i,j}=\min\limits_{0\leq k<i}\{f_{k,j-1}+w(k,i)\}$
$$
w(l,r)=\begin{cases}s_r-2s_{(l+r+1)/2}+a_{(l+r-1)/2}+s_l&r-l\bmod 2=1\\s_r-2s_{(l+r-1)/2}+s_l&r-l\bmod 2=0\end{cases}
$$
满足四边形不等式

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
const int MAXN = 305;
struct Node{
	int id, pos;
	Node(int id = 0, int pos = 0): id(id), pos(pos){}
};
int n, m;
int a[MAXN], s[MAXN];
int f[MAXN][MAXN];
Node q[MAXN];
int ql, qr;
int calc(int i, int l, int r){
	int mid = (l + r + 1) >> 1;
	// printf("calc %d %d %d %d\n", i, l, r, f[i - 1][l] - a[mid] * ((r - l) & 1) + (s[r] - s[mid]) - (s[mid - 1] - s[l]));
	return f[i - 1][l] - a[mid] * ((r - l + 1) & 1) + (s[r] - s[mid]) - (s[mid - 1] - s[l]);
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++){
		scanf("%d", a + i);
		s[i] = s[i - 1] + a[i];
	}
	sort(a + 1, a + n + 1);
	memset(f, 0x3f, sizeof(f));
	f[0][0] = 0;
	for (int i = 1; i <= m; i++){
		ql = 1, qr = 0;
		q[++qr] = Node(0, 1);
		for (int j = 1; j <= n; j++){
			while (ql < qr && q[ql + 1].pos <= j) ql++;
			if (ql <= qr) f[i][j] = calc(i, q[ql].id, j);
			// printf("f %d %d %d\n", i, j, f[i][j]);
			while (ql < qr && calc(i, j, q[qr].pos) <= calc(i, q[qr].id, q[qr].pos)) qr--;
			int lft = max(q[qr].pos, j + 1), rgt = n, ans = n + 1;
			while (lft <= rgt){
				int mid = (lft + rgt) >> 1;
				if (calc(i, j, mid) <= calc(i, q[qr].id, mid)){
					ans = mid;
					rgt = mid - 1;
				} else lft = mid + 1;
			}
			if (ans <= n) q[++qr] = Node(j, ans);
		}
	}
	printf("%d\n", f[m][n]);
	return 0;
}
```

