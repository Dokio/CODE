# 算法CheatSheet



# 二分

## 使用条件

1. 排序数组 (30-40%是二分)
2. 当面试官要求你找一个比 O(n) 更小的时间复杂度算法的时候(99%)
3. 找到数组中的一个分割位置，使得左半部分满足某个条件，右半部分不满足(100%)
4. 找到一个最大/最小的值使得某个条件被满足(90%)


## 复杂度

1. 时间复杂度：O(logn)
2. 空间复杂度：O(1)

## 模版

Java

```
public class Solution {
    /**
     * @param A an integer array sorted in ascending order
     * @param target an integer
     * @return an integer
     */
    public int findPosition(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }

        int start = 0, end = nums.length - 1;
        // 要点1: start + 1 < end
        while (start + 1 < end) {
     	// 要点2：start + (end - start) / 2
            int mid = start + (end - start) / 2;
            // 要点3：=, <, > 分开讨论，mid 不 +1 也不 -1
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                start = mid;
            } else {
                end = mid;
            }
        }

        // 要点4: 循环结束后，单独处理start和end
        if (nums[start] == target) {
            return start;
        }
        if (nums[end] == target) {
            return end;
        }
        return -1;
    }
}
```

Python

```
class Solution:
    # @param nums: The integer array
    # @param target: Target number to find
    # @return the first position of target in nums, position start from 0
    def binarySearch(self, nums, target):
        if not nums:
            return -1

        start, end = 0, len(nums) - 1
        # 用 start + 1 < end 而不是 start < end 的目的是为了避免死循环
        # 在 first position of target 的情况下不会出现死循环
        # 但是在 last position of target 的情况下会出现死循环
        # 样例：nums=[1，1] target = 1
        # 为了统一模板，我们就都采用 start + 1 < end，就保证不会出现死循环
        while start + 1 < end:
            # python 没有 overflow 的问题，直接 // 2 就可以了
            # java和C++ 最好写成 mid = start + (end - start) / 2
            # 防止在 start = 2^31 - 1, end = 2^31 - 1 的情况下出现加法 overflow
            mid = (start + end) // 2

            # > , =, < 的逻辑先分开写，然后在看看 = 的情况是否能合并到其他分支里
            if nums[mid] < target:
                start = mid
            elif nums[mid] == target:
                end = mid
            else:
                end = mid

        # 因为上面的循环退出条件是 start + 1 < end
        # 因此这里循环结束的时候，start 和 end 的关系是相邻关系（1和2，3和4这种）
        # 因此需要再单独判断 start 和 end 这两个数谁是我们要的答案
        # 如果是找 first position of target 就先看 start，否则就先看 end
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end

        return -1
```



## 例题

- LintCode14 . 二分查找(在排序的数据集上进行二分) :https://www.lintcode.com/problem/first-position-of-target/description
- LintCode460. 在排序数组中找最接近的K个数 (在未排序的数据集上进行二分): https://www.lintcode.com/problem/find-k-closest-elements/description
- LintCode437.书籍复印(在答案集上进行二分 )[:https://www.lintcode.com/problem/copy-books/description](https://www.lintcode.com/problem/copy-books/description)







# 双指针



## 使用条件

1. 滑动窗口 (90%)
2. 时间复杂度要求 O(n) (80%是双指针)
3. 要求原地操作，只可以使用交换，不能使用额外空间 (80%)
4. 有子数组 subarray /子字符串 substring 的关键词 (50%)
5. 有回文  Palindrome 关键词(50%)

## 复杂度

- 时间复杂度：O(n)
  -  时间复杂度与最内层循环主体的执行次数有关
  - 与有多少重循环无关
- 空间复杂度：O(1)
  - 只需要分配两个指针的额外内存

## 模版

Java

```java
// 相向双指针(patition in quicksort)
public void patition(int[] A, int start, int end) {
    if (start >= end) {
            return;
        }

        int left = start, right = end;
        // key point 1: pivot is the value, not the index
        int pivot = A[(start + end) / 2];

        // key point 2: every time you compare left & right, it should be
        // left <= right not left < right
        while (left <= right) {
            while (left <= right && A[left] < pivot) {
                left++;
            }
            while (left <= right && A[right] > pivot) {
                right--;
            }
            if (left <= right) {
                int temp = A[left];
                A[left] = A[right];
                A[right] = temp;

                left++;
                right--;
            }
        }
}

// 背向双指针(is_palindrome)
private int findLongestPalindromeFrom(String s, int left, int right) {
    int len = 0;
    while (left >= 0 && right < s.length()) {
        if (s.charAt(left) != s.charAt(right)) {
            break;
        }
        len += left == right ? 1 : 2;
        left--;
        right++;
    }

    return len;
}
//同向双指针
int j = 1;
for (int i = 0; i < n; i++) {
	//不满足则循环到满足搭配为止
	while (j < n && !this.check(i, j)) {
		j += 1;
	}
	if (j >= n) {
		break;
	}
	//处理i，j这次搭配
}

```

Python

```python
# 相向双指针(patition in quicksort)

def patition(self, A, start, end):
        if start >= end:
            return

        left, right = start, end
        # key point 1: pivot is the value, not the index
        pivot = A[(start + end) / 2];

        # key point 2: every time you compare left & right, it should be
        # left <= right not left < right
        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1

            while left <= right and A[right] > pivot:
                right -= 1

            if left <= right:
                A[left], A[right] = A[right], A[left]

            left += 1
            right -= 1



# 背向双指针(is_palindrome)
def findLongestPalindromeFrom(self, s, left, right):
    length = 0
    while left >= 0 and right < len(s):
        if s[left] != s[right]:
            break
        length += 1 if left == right else right
        left -= 1
        right += 1
    return length


# 同向双指针
j = 1
for i in range(n):
    # 不满足则循环到满足搭配为止
    while j < n and not check(i, j):
        j += 1
    if j >= n:
        break
    # 处理i，j这次搭配

```



## 例题

- LintCode 1879. 两数之和VII(同向双指针): https://www.lintcode.com/problem/two-sum-vii/description
- LintCode1712.和相同的二元子数组(相向双指针):https://www.lintcode.com/problem/binary-subarrays-with-sum/description
- LintCode627. 最长回文串 (背向双指针): https://www.lintcode.com/problem/longest-palindrome/description











# BFS

## 使用条件

1. 拓扑排序(100%)
2. 出现连通块的关键词(100%)
3. 分层遍历(100%)
4. 简单图最短路径(100%)
5. 给定一个变换规则，从初始状态变到终止状态最少几步(100%)

## 复杂度

- 时间复杂度：O(n)
- 空间复杂度：O(n)

## 模版

Java

```java
// clone graph
public class Solution {
    /**
     * @param node: A undirected graph node
     * @return: A undirected graph node
     */
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        if (node == null) {
            return node;
        }

        // use bfs algorithm to traverse the graph and get all nodes.
        ArrayList<UndirectedGraphNode> nodes = getNodes(node);

        // copy nodes, store the old->new mapping information in a hash map
        HashMap<UndirectedGraphNode, UndirectedGraphNode> mapping = new HashMap<>();
        for (UndirectedGraphNode n : nodes) {
            mapping.put(n, new UndirectedGraphNode(n.label));
        }

        // copy neighbors(edges)
        for (UndirectedGraphNode n : nodes) {
            UndirectedGraphNode newNode = mapping.get(n);
            for (UndirectedGraphNode neighbor : n.neighbors) {
                UndirectedGraphNode newNeighbor = mapping.get(neighbor);
                newNode.neighbors.add(newNeighbor);
            }
        }

        return mapping.get(node);
    }

    private ArrayList<UndirectedGraphNode> getNodes(UndirectedGraphNode node) {
        Queue<UndirectedGraphNode> queue = new LinkedList<UndirectedGraphNode>();
        HashSet<UndirectedGraphNode> set = new HashSet<>();

        queue.offer(node);
        set.add(node);
        while (!queue.isEmpty()) {
            UndirectedGraphNode head = queue.poll();
            for (UndirectedGraphNode neighbor : head.neighbors) {
                if (!set.contains(neighbor)) {
                    set.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }

        return new ArrayList<UndirectedGraphNode>(set);
    }
}
```

Python

```python
# clone graph
class Solution:
    """
    @param: node: A undirected graph node
    @return: A undirected graph node
    """
    def cloneGraph(self, node):
        root = node
        if node is None:
            return node

        # use bfs algorithm to traverse the graph and get all nodes.
        nodes = self.getNodes(node)

        # copy nodes, store the old->new mapping information in a hash map
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)

        # copy neighbors(edges)
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)

        return mapping[root]

    def getNodes(self, node):
        q = collections.deque([node])
        result = set([node])
        while q:
            head = q.popleft()
            for neighbor in head.neighbors:
                if neighbor not in result:
                    result.add(neighbor)
                    q.append(neighbor)
        return result
```



## 例题

- LintCode 974.01 矩阵(分层遍历): https://www.lintcode.com/problem/01-matrix/description
- LintCode 431. 找无向图的连通块: https://www.lintcode.com/problem/connected-component-in-undirected-graph/description
- LintCode 127.拓扑排序:[ https://www.lintcode.com/problem/topological-sorting/description](https://www.lintcode.com/problem/topological-sorting/description)







# 二叉树与分治
## 分治算法的使用条件

- 二叉树相关的问题 (99%)
- 可以一分为二去分别处理之后再合并结果 (100%)
- 数组相关的问题 (10%)

## 模板

Java

```java
public ResultType treeMerge(TreeNode node) {
    // 递归终点
    if (node == null) {
        return ...;
    }

    // 处理左子树
    ResultType leftResult = helper(node.left);
    // 处理右子树
    ResultType rightResult = helper(node.right);

    //合并答案
    result = merge(leftResult, rightResult);

    return result;
}
```



Python

```python
def treeMerge(root):
    # 递归终点
    if root is None:
        return ...

    # 处理左子树
    leftResult = treeMerge(node.left)
    # 处理右子树
    tightResult = treeMerge(node.right)

    # 合并答案
    result = merge(leftResult, rightResult)

    return result

```







# DFS

## 使用条件

- 找满足某个条件的所有方案 (99%)
- 二叉树 Binary Tree 的问题 (90%)
- 组合问题(95%)
  -  问题模型：求出所有满足条件的“组合”
  -  判断条件：组合中的元素是顺序无关的
- 排列问题 (95%)
  -  问题模型：求出所有满足条件的“排列”
  -  判断条件：组合中的元素是顺序“相关”的。

# 一定不要用 DFS 的场景
1. 连通块问题（一定要用 BFS，否则 StackOverflow）
2. 拓扑排序（一定要用 BFS，否则 StackOverflow）


## 复杂度

- 时间复杂度：O(方案个数 * 构造每个方案的时间)
  -  树的遍历：O(n)
  -  排列问题 ： O(n! * n)
  -  组合问题 ： O(2^n * n)

## 模版

Java

```
public ReturnType dfs(参数列表) {
	if (递归出口) {
		记录答案;
		return;
	}

	for (所有的拆解可能性) {
		修改所有的参数
		dfs(参数列表);
		还原所有被修改过的参数
	}
    return ...
}
```

Python

```
def dfs(参数列表) {
	if 递归出口:
		记录答案
		return

	for 所有的拆解可能性:
		修改所有的参数
		dfs(参数列表)
		还原所有被修改过的参数

	return ...
}
```



## 例题

- LintCode 67.二叉树的中序遍历(遍历树):https://www.lintcode.com/problem/binary-tree-inorder-traversal/description
- LintCode 652.因式分解(枚举所有情况): https://www.lintcode.com/problem/factorization/description







# 动态规划

## 使用条件

- 使用场景：
  - 求方案总数(90%)
  - 求最值(80%)
  - 求可行性(80%)
- 不适用的场景：
  - 找所有具体的方案（准确率99%）
  - 输入数据无序(除了背包问题外，准确率60%~70%)
  - 暴力算法已经是多项式时间复杂度（准确率80%）
- 动态规划四要素(对比递归的四要素)：
  - 状态 (State) -- 递归的定义
  - 方程 (Function) --  递归的拆解
  - 初始化 (Initialization) -- 递归的出口
  - 答案 (Answer) --  递归的调用
- 几种常见的动态规划：

- 背包型
  - 给出 n 个物品及其大小,问是否能挑选出一些物品装满大小为m的背包
  - 题目中通常有“和”与“差”的概念，数值会被放到状态中
  - 通常是二维的状态数组，前 i 个组成和为 j 状态数组的大小需要开 (n + 1) * (m + 1)
  - 几种背包类型：
    - 01背包
      - 状态 state
        `    dp[i][j] 表示前 i 个数里挑若干个数是否能组成和为 j`
        方程 function
        `    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - A[i - 1]] 如果 j >= A[i - 1]`
        `    dp[i][j] = dp[i - 1][j] 如果 j < A[i - 1]`
        `    第 i 个数的下标是 i - 1，所以用的是 A[i - 1] 而不是 A[i]`
        初始化 initialization
        `    dp[0][0] = true`
        `    dp[0][1...m] = false`
        答案 answer
        `使得 dp[n][v], 0 s<= v <= m 为 true 的最大 v`
    - 多重背包
      - 状态 state
        `	dp[i][j] 表示前i个物品挑出一些放到 j 的背包里的最大价值和`
        方程 function
        `	dp[i][j] = max(dp[i - 1][j - count * A[i - 1]] + count * V[i - 1])`
        `	其中 0 <= count <= j / A[i - 1]`
        初始化 initialization
        `	dp[0][0..m] = 0`
        答案 answer
        `	dp[n][m]`
- 区间型
-  题目中有 subarray / substring 的信息
  -  大区间依赖小区间
  -  用 `dp[i][j]` 表示数组/字符串中 `i, j` 这一段区间的最优值/可行性/方案总数
  -  状态 state
     `    dp[i][j] 表示数组/字符串中 i,j 这一段区间的最优值/可行性/方案总数`
     方程 function
     `    dp[i][j] = max/min/sum/or(dp[i,j 之内更小的若干区间])`
- 匹配型

  - 通常给出两个字符串
  - 两个字符串的匹配值依赖于两个字符串前缀的匹配值
  - 字符串长度为 n,m 则需要开 (n + 1) x (m + 1) 的状态数组
  - 要初始化 `dp[i][0]` 与 `dp[0][i]`
  - 通常都可以用滚动数组进行空间优化
  - 状态 state
    `    dp[i][j] 表示第一个字符串的前 i 个字符与第二个字符串的前 j 个字符怎么样怎么样(max/min/sum/or)`
- 划分型

  -  是前缀型动态规划的一种, 有前缀的思想
  -  如果指定了要划分为几个部分：
    - dp[i][j] 表示前i个数/字符划分为j个 部分的最优值/方案数/可行性
  - 如果没有指定划分为几个部分:
    - dp[i] 表示前i个数/字符划分为若干个 部分的最优值/方案数/可行性
  -  状态 state
     	指定了要划分为几个部分:`dp[i][j]` 表示前i个数/字符划分为j个部分的最优值/方案数/可行性
        	没有指定划分为几个部分: `dp[i]` 表示前i个数/字符划分为若干个部分的最优值/方案数/可行性
- 接龙型

  -  通常会给一个接龙规则，问你最长的龙有多长
  -  状态表示通常为:  dp[i] 表示以坐标为 i 的元素结尾的最长龙的长度
  -  方程通常是: dp[i] = max{dp[j] + 1}, j 的后面可以接上 i
  -  LIS 的二分做法选择性的掌握，但并不是所有的接龙型DP都可以用二分来优化
  -  状态 state
     `	状态表示通常为: dp[i] 表示以坐标为 i 的元素结尾的最长龙的长度`
     方程 function
     `	dp[i] = max{dp[j] + 1}, j 的后面可以接上 i`

## 复杂度

- 时间复杂度:
  - O(状态总数 * 每个状态的处理耗费)
  - 等于O(状态总数 * 决策数)
- 空间复杂度：
  - O(状态总数) (不使用滚动数组优化)
  - O(状态总数 / n)(使用滚动数组优化, n是被滚动掉的那一个维度)



## 例题

- LintCode563.背包问题V(背包型): https://www.lintcode.com/problem/backpack-v/description
- LintCode76.最长上升子序列(接龙型): https://www.lintcode.com/problem/longest-increasing-subsequence/description
- LintCode 476.石子归并V(区间型):https://www.lintcode.com/problem/stone-game/description
- LintCode 192. 通配符匹配  (匹配型):  https://www.lintcode.com/problem/wildcard-matching/description
- LintCode107.单词拆分(划分型): https://www.lintcode.com/problem/word-break/description




## 哈希表

能解决哪些问题：
1. 查询一个元素是否存在于一个集合 (100%)
2. 统计出现次数 (100%)
3. 要求 O(1) 的时间复杂度进行一些操作 (80%)

不能解决哪些问题：
1. 查询比某个数大的最小值/最接近的值（平衡排序二叉树 Balanced BST 才可以解决）
2. 查询最小值最大值 ( 平衡排序二叉树 Balanced BST 才可以解决)
3. 查询第 k 大的数（堆 heap 可以解决）


# 堆

能解决哪些问题：
1. 找最大值或者最小值(60%)
2. 找第 k 大(pop k 次 复杂度O(nlogk))(50%)
3. 要求 logn 时间对数据进行操作(40%)


不能解决哪些问题：
1. 查询比某个数大的最小值/最接近的值（平衡排序二叉树 Balanced BST 才可以解决）
2. 找某段区间的最大值最小值（线段树 SegmentTree 可以解决）
3. O(n)找第k大 (使用快排中的partition操作)


