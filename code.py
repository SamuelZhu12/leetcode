#  排序
## 快排
class Solution(object):

    def quickSort(self,nums,l,r):
        if(l >= r): return
        i = l
        j = r
        tmp = nums[l]
        while i < j:
            while i<j and nums[j] >= tmp: j -= 1
            nums[i] = nums[j]
            while i<j and nums[i] <= tmp: i += 1
            nums[j] = nums[i]
            nums[i] = tmp
        self.quickSort(nums,l,i-1)
        self.quickSort(nums,i+1,r)
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        self.quickSort(nums,0,len(nums) - 1)
        return nums
        



# 链表
## 相交链表
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        pa = headA
        pb = headB
        while pa != pb:
            if not pa: pa = headB
            else : pa = pa.next
            if not pb: pb = headA
            else : pb = pb.next
        return pa
		
## 反转链表
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None
        while head:
            tmp = head.next
            head.next = pre
            pre = head
            head = tmp
        return pre
		
## 反转链表2
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        # 设置 dummyNode 是这一类问题的一般做法
        dummy_node = ListNode(-1)
        dummy_node.next = head
        pre = dummy_node
        for _ in range(left - 1):
            pre = pre.next

        cur = pre.next
        for _ in range(right - left):
            next = cur.next
            cur.next = next.next
            next.next = pre.next
            pre.next = next
        return dummy_node.next
		
## 删除链表指定值的节点
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = ListNode(0,head)
        cur = dummy
        while cur.next:		# 防止cur遍历到尾节点 
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next

## 删除倒数第n个节点 记住要加头部的懒节点
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        def getLength(head):
            length = 0
            while head:
                length += 1
                head = head.next
            return length
        n = getLength(head) - n + 1
        dummy = ListNode(0,head)
        cur = dummy
        for i in range(1,n):
            cur = cur.next
        cur.next = cur.next.next
        return dummy.next



# 第k大的数
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        
        def partition(arr: List[int], low: int, high: int) -> int:
            pivot = arr[low]                                        # 选取最左边为pivot
  
            left, right = low, high     # 双指针
            while left < right:
                
                while left<right and arr[right] >= pivot:          # 找到右边第一个<pivot的元素
                    right -= 1
                arr[left] = arr[right]                             # 并将其移动到left处
                
                while left<right and arr[left] <= pivot:           # 找到左边第一个>pivot的元素
                    left += 1
                arr[right] = arr[left]                             # 并将其移动到right处
            
            arr[left] = pivot           # pivot放置到中间left=right处
            return left
        
        def randomPartition(arr: List[int], low: int, high: int) -> int:
            pivot_idx = random.randint(low, high)                   # 随机选择pivot
            arr[low], arr[pivot_idx] = arr[pivot_idx], arr[low]     # pivot放置到最左边
            return partition(arr, low, high)                        # 调用partition函数

        def topKSplit(arr: List[int], low: int, high: int, k: int) -> int:
            # mid = partition(arr, low, high)                   # 以mid为分割点【非随机选择pivot】
            mid = randomPartition(arr, low, high)               # 以mid为分割点【随机选择pivot】
            if mid == k-1:                                      # 第k小元素的下标为k-1
                return arr[mid]                                 #【找到即返回】
            elif mid < k-1:
                return topKSplit(arr, mid+1, high, k)           # 递归对mid右侧元素进行排序
            else:
                return topKSplit(arr, low, mid-1, k)            # 递归对mid左侧元素进行排序
        
        n = len(nums)
        return topKSplit(nums, 0, n-1, n-k+1)                   # 第k大元素即为第n-k+1小元素


# 三数之和
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        res = []
        if(n < 3):
            return res
        nums = sorted(nums)
        for i in range(n):
            l = i + 1
            r = n - 1
            if(nums[i] > 0):
                break
            elif i > 0 and nums[i] == nums[i-1]:
                continue
            while i < l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum < 0 : l += 1
                elif sum > 0: r -= 1
                else:
                    res.append([nums[i],nums[l],nums[r]])
                    while l < r and nums[l] == nums[l+1]: l += 1
                    while l < r and nums[r] == nums[r-1]: r -= 1
                    l += 1
                    r -= 1
            
        return res
		
		
# 四数之和
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if len(nums) < 4: return []
        nums = sorted(nums)
        res = []
        i = 0
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]: continue
            for j in range(i+1,len(nums)):
                l = j + 1
                r = len(nums) - 1
                if j > i+1 and nums[j] == nums[j-1] : continue
                while i < j < l < r : 
                    if nums[i] + nums[j] + nums[l] + nums[r] < target:
                        l += 1
                    elif nums[i] + nums[j] + nums[l] + nums[r] > target:
                        r -= 1
                    else:
                        res.append([nums[i],nums[j],nums[l],nums[r]])
                        while l < r and nums[l] == nums[l+1] : l += 1
                        while l < r and nums[r] == nums[r-1] : r -= 1
                        l += 1
                        r -= 1
        return res


		
# 最大子序和
class Solution:

    def maxSubArray(self, nums: List[int]) -> int:ll
        size = len(nums)
        if size == 0:
            return 0
        dp = [0 for _ in range(size)]

        dp[0] = nums[0]
        for i in range(1, size):
            if dp[i - 1] >= 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)


# 买入股票的时机
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) <= 1: return 0
        dp = [0 for _ in prices]
        dp[1] = prices[1] - prices[0]
        
        for i in range(2,len(prices)):
            if dp[i-1] >= 0:
                dp[i] = dp[i - 1] + prices[i]-prices[i-1]
            else:
                dp[i] = prices[i] -prices[i-1]
        return max(max(dp),0)



# 合并有序链表

class Solution(object):

    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        pre_head = ListNode(-1)
        res = pre_head
        while(list1 != None and list2 != None):
            if(list1.val <= list2.val):
                res.next = list1
                list1 = list1.next
            else:
                res.next = list2
                list2 = list2.next
            res = res.next
        if(list1 == None):
            res.next = list2
        else:
            res.next = list1
        return pre_head.next
		
		
		
# 合并非递减数组
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        sorted = []
        p1, p2 = 0, 0
        while p1 < m or p2 < n:
            if p1 == m:
                sorted.append(nums2[p2])
                p2 += 1
            elif p2 == n:
                sorted.append(nums1[p1])
                p1 += 1
            elif nums1[p1] < nums2[p2]:
                sorted.append(nums1[p1])
                p1 += 1
            else:
                sorted.append(nums2[p2])
                p2 += 1
        nums1[:] = sorted


		
# 两数之和
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(nums)
        for i in range(n):
            tmp = target - nums[i]
            if tmp in nums:
                j = nums.index(tmp)
                if i == j :continue
                else : return [i,j]
				
				
# 二叉树
## 二叉树中序遍历
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        def traversal(root):
            if not root : return
            traversal(root.left)
            res.append(root.val)
            traversal(root.right)
        traversal(root)
        return res

## 二叉树层序遍历
### 递归法
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def func(root,level):
            if not root:
                return []
            if len(res) == level: # 遍历到第i层第一个节点，就为该层扩充一个[]，如果访问到第i层第2个节点，则不扩充
                res.append([])
            res[level].append(root.val) 
            if root.left: func(root.left,level+1)
            if root.right: func(root.right,level+1)
        func(root,0)
        return res
### 迭代法
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        # 存储节点的双向队列
        nodeQue = deque([root])
        results = []
        while nodeQue: # 当节点队列不为空时
            res = []
            for i in range(len(nodeQue)): # 遍历存储该层节点的队列
                cur = nodeQue.popleft()
                res.append(cur.val)
                if cur.left: nodeQue.append(cur.left)
                if cur.right: nodeQue.append(cur.right)
            results.append(res)
        return results

## 二叉树锯齿形层序遍历
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        nodeQue = deque([root])
        results = []
        flag= 0
        while nodeQue:
            res = []
            size = len(nodeQue)
            for i in range(size):
                cur = nodeQue.popleft()
                if flag % 2 == 0:  # 仅有此处跟二叉树层序遍历不同
                    res.append(cur.val) # 列表前后插入
                else:res.insert(0,cur.val)
                if cur.left:nodeQue.append(cur.left)
                if cur.right:nodeQue.append(cur.right)
            flag += 1
            results.append(res)
        return results

## 二叉树右视图
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        nodeQue = deque([root])
        res = []
        while nodeQue:
            size = len(nodeQue)
            for i in range(size):
                if i == 0: res.append(nodeQue[-1].val)
                cur = nodeQue.popleft()
                if cur.left: nodeQue.append(cur.left)
                if cur.right: nodeQue.append(cur.right)
        return res

## 二叉树最大深度：
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def traverse(root):
            if not root: return 0
            else : return 1 + max(traverse(root.left),traverse(root.right))
        return traverse(root)

## 二叉树最小深度
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        if not root.left and not root.right:
            return 1

        min_depth = 10 ** 9
        if root.left:
            min_depth = min(self.minDepth(root.left), min_depth)
        if root.right:
            min_depth = min(self.minDepth(root.right), min_depth)

        return min_depth + 1

## 迭代法的最小深度
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0

        # 根节点的深度为1
        queue_ = [(root, 1)]
        while queue_:
            cur, depth = queue_.pop(0)

            if cur.left == None and cur.right == None:
                return depth
            # 先左子节点，由于左子节点没有孩子，则就是这一层了
            if cur.left:
                queue_.append((cur.left, depth + 1))
            if cur.right:
                queue_.append((cur.right, depth + 1))

        return 0

# 岛屿数量
class Solution:
    def dfs(self, grid, r, c):
        grid[r][c] = 0
        nr, nc = len(grid), len(grid[0])
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                    num_islands += 1
                    self.dfs(grid, r, c)
        
        return num_islands


# 最长回文子串
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s
        
        max_len = 1
        begin = 0
        # dp[i][j] 表示 s[i..j] 是否是回文串
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        
        # 递推开始
        # 先枚举子串长度
        for L in range(2, n + 1):
            # 枚举左边界，左边界的上限设置可以宽松一些
            for i in range(n):
                # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                j = L + i - 1
                # 如果右边界越界，就可以退出当前循环
                if j >= n:
                    break
                    
                if s[i] != s[j]:
                    dp[i][j] = False 
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                
                # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin:begin + max_len]


# 回溯
# 全排列
class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(first = 0):
            # 所有数都填完了
            if first == n:  
                res.append(nums[:])
            for i in range(first, n):
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first] # 保留 first-1 的状态
        
        n = len(nums)
        res = []
        backtrack()
        return res

# 带有target的组合 如[2,2,2,3]之和等于7的所有组合
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def backtracking(target, n, first, candidates):
            if target == 0:
                res.append(path[:])
                return
            if target < 0:
                return
            for i in range(first, n):
                path.append(candidates[i])
                backtracking(target - candidates[i], n, i, candidates)
                path.pop()
        res = []
        path = []
        n = len(candidates)
        backtracking(target, n, 0, candidates)
        return res


# 动态规划
## 带障碍的路径规划
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
		# 初始化为0
        dp = [[0 for _ in range(n)] for _ in range(m)]
        if obstacleGrid[0][0] == 1:
            return 0
        else: dp[0][0] = 1
		# 第一行第一列的初始化，如果有障碍，则后面的dp数组置为0
        for i in range(1,n):
            if obstacleGrid[0][i] != 1:
                dp[0][i] = dp[0][i-1]
        for i in range(1,m):
            if obstacleGrid[i][0] != 1:
                dp[i][0] = dp[i-1][0]
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] != 1:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[m-1][n-1]
		
## 爬楼梯
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0 for i in range(n+1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
		
## 拆分两数
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in range(3, n + 1):
            # 假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i），则有以下两种方案：
            # 1) 将 i 拆分成 j 和 i−j 的和，且 i−j 不再拆分成多个正整数，此时的乘积是 j * (i-j)
            # 2) 将 i 拆分成 j 和 i−j 的和，且 i−j 继续拆分成多个正整数，此时的乘积是 j * dp[i-j]
            for j in range(1, i - 1):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[n]

## 搜索二叉树数量
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        ## 找到递推公式是关键，3的顶点包括1、2的形式，4的顶点包括1、2、3的形式
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        for i in range(1,n+1):
            for j in range(1,i+1):
                dp[i] += dp[j-1]*dp[i-j]

        return dp[n]
		
		
# dfs岛屿问题
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        def dfs(grid,i,j):
            if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] == '0': return
            grid[i][j] = '0'
            dfs(grid,i+1,j)
            dfs(grid,i,j+1)
            dfs(grid,i-1,j)
            dfs(grid,i,j-1)
        m = len(grid)
        n = len(grid[0])
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    dfs(grid,i,j)
                    count += 1
        return count

# 岛屿周长
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = len(grid[0])
        count = 0
        edge = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    count += 1
                    if j+1 < n and grid[i][j+1] == 1:
                        edge += 1
                    if i+1 < m and grid[i+1][j] == 1:
                        edge += 1
        return count*4 - 2*edge # 每相邻一个岛屿，就会减少两条边
                
