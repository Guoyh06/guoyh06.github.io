# 数位DP 作业

### P2657 [SCOI2009]windy数

记录当前位置，上一位数值，目前是否卡到上界，目前是否还是前导0

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
int a[11];
int f[13][11][2][2];
int dfs(int u, int v, bool f1, bool f2){ //当前位置 上一位 是否卡死 是否为前导
	if (u == -1) return 1;
	if (f[u][v][f1][f2] != -1) return f[u][v][f1][f2];
	int ans = 0;
	int mxn = f1 ? a[u] : 9;
	for (int i = 0; i <= mxn; i++){
		if (!f2 && abs(i - v) <= 1) continue;
		ans += dfs(u - 1, i, f1 && i == mxn, f2 && i == 0);
	}
	return f[u][v][f1][f2] = ans;
}
int work(int x){
	memset(f, -1, sizeof(f));
	for (int i = 0; i <= 10; i++){
		a[i] = x % 10;
		x /= 10;
	}
	return dfs(10, 0, true, true);
}
int main(){
	int a, b;
	scanf("%d%d", &a, &b);
	printf("%d\n", work(b) - work(a - 1));
	return 0;
}
```

### AcWing 310. 启示录

记录当前位置，当前6的长度，是否卡住上界

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
int a[15];
ll f[15][4][2];
ll dfs(int nw, int l, bool f1){
	// printf("dfs %d %d %d\n", nw, l, f1);
	if (nw == -1) return l == 3;
	if (f[nw][l][f1] != -1) return f[nw][l][f1];
	int ans = 0;
	int mxn = f1 ? a[nw] : 9;
	for (int i = 0; i <= mxn; i++){
		ans += dfs(nw - 1, ((i == 6 || l == 3) ? min(3, l + 1) : 0), f1 && i == mxn);
	}
	return f[nw][l][f1] = ans;
}
int work(ll x){
	for (int i = 0; i <= 11; i++){
		a[i] = x % 10;
		x /= 10;
	}
	memset(f, -1, sizeof(f));
	return dfs(11, 0, true);
}
int main(){
	int t;
	ll n;
	scanf("%d", &t);
	while (t--){
		scanf("%lld", &n);
		ll lft = 666, rgt = n * 1001, ans = -1;
		while (lft <= rgt){
			// printf("lr %lld %lld\n", lft, rgt);
			ll mid = (lft + rgt) >> 1;
			if (work(mid) >= n){
				ans = mid;
				rgt = mid - 1;
			} else lft = mid + 1;
		}
		printf("%lld\n", ans);
	}
	return 0;
}
```

### AcWing 311. 月之谜

枚举各位数字之和，记录当前位置，当前数的余数，当前数字和，是否卡住上界

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
int tol;
int a[15];
int f[11][101][101][2];
ll dfs(int nw, int u, int v, bool flag){
	// printf("dfs %d %d %d %d\n", nw, u, v, flag);
	if (nw == -1) return u == 0 && v == tol;
	if (f[nw][u][v][flag] != -1) return f[nw][u][v][flag];
	int ans = 0;
	int mxn = flag ? a[nw] : 9;
	for (int i = 0; i <= mxn; i++){
		ans += dfs(nw - 1, (u * 10 + i) % tol, v + i, flag && i == mxn);
	}
	return f[nw][u][v][flag] = ans;
}
int work(int x){
	for (int i = 0; i <= 10; i++){
		a[i] = x % 10;
		x /= 10;
	}
	int ans = 0;
	for (int i = 1; i <= 90; i++){
		memset(f, -1, sizeof(f));
		tol = i;
		ans += dfs(10, 0, 0, true);
	}
	return ans;
}
int main(){
	int l, r;
	scanf("%d%d", &l, &r);
	printf("%d\n", work(r) - work(l - 1));
	return 0;
}
```

### AcWing 338. 计数问题

记录当前位置，是否是前导0，是否卡住上界

若当前数字是要统计的数字，若没有卡住上界则贡献为$10^{pos}$，否则是$lim\bmod 10^{pos}$

```cpp
# include <bits/stdc++.h>
# define ll long long
using namespace std;
ll p10[11];
int a[15], b[15];
ll f[11][11][2][2];
ll dfs(int nw, int nm, bool f1, bool f2){
	if (nw == -1) return 0;
	if (f[nw][nm][f1][f2] != -1) return f[nw][nm][f1][f2];
	ll ans = 0;
	int mxn = f1 ? a[nw] : 9;
	for (int i = 0; i <= mxn; i++){
		ans += dfs(nw - 1, nm, f1 && i == mxn, f2 && i == 0);
		if (f2 && i == 0) continue;
		ans += f1 && i == mxn ? (i == nm) * b[nw] : (i == nm) * p10[nw];
	}
	return f[nw][nm][f1][f2] = ans;
}
ll work(int x, int nm){
	for (int i = 0; i <= 10; i++){
		a[i] = (x / p10[i]) % 10;
		b[i] = x % p10[i] + 1;
	}
	ll ans = 0;
	memset(f, -1, sizeof(f));
	ans += dfs(10, nm, true, true);
	return ans;
}
int main(){
	p10[0] = 1;
	for (int i = 1; i <= 10; i++) p10[i] = p10[i - 1] * 10;
	int l, r;
	while (true){
		scanf("%d%d", &l, &r);
		if (l == 0 && r == 0) break;
		if (l > r) swap(l, r);
		for (int i = 0; i < 10; i++){
			printf("%lld ", work(r, i) - work(l - 1, i));
		}
		putchar('\n');
	}
	return 0;
}
```

