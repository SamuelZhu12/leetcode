# 快排
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

# 相交链表
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
		
# 反转链表
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
		
# 反转链表2
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
# 相交链表

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
		
# 删除倒数第n个节点 记住要加头部的懒节点
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
				
				
# 二叉树中序遍历
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

# 二叉树层序遍历
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        self.level(root, 0, res)
        return res

    def level(self, root, level, res):
        if not root: return
        if len(res) == level: res.append([])
        res[level].append(root.val)
        if root.left: self.level(root.left, level + 1, res)
        if root.right: self.level(root.right, level + 1, res)

# 二叉树最大深度：
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
                nums[first], nums[i] = nums[i], nums[first]
        
        n = len(nums)
        res = []
        backtrack()
        return res

