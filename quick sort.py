# ����
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
		
		
# ��ת����
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
		
# ��ת����2
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        # ���� dummyNode ����һ�������һ������
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

# ɾ��������n���ڵ� ��סҪ��ͷ�������ڵ�
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



# ��k�����
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        
        def partition(arr: List[int], low: int, high: int) -> int:
            pivot = arr[low]                                        # ѡȡ�����Ϊpivot
  
            left, right = low, high     # ˫ָ��
            while left < right:
                
                while left<right and arr[right] >= pivot:          # �ҵ��ұߵ�һ��<pivot��Ԫ��
                    right -= 1
                arr[left] = arr[right]                             # �������ƶ���left��
                
                while left<right and arr[left] <= pivot:           # �ҵ���ߵ�һ��>pivot��Ԫ��
                    left += 1
                arr[right] = arr[left]                             # �������ƶ���right��
            
            arr[left] = pivot           # pivot���õ��м�left=right��
            return left
        
        def randomPartition(arr: List[int], low: int, high: int) -> int:
            pivot_idx = random.randint(low, high)                   # ���ѡ��pivot
            arr[low], arr[pivot_idx] = arr[pivot_idx], arr[low]     # pivot���õ������
            return partition(arr, low, high)                        # ����partition����

        def topKSplit(arr: List[int], low: int, high: int, k: int) -> int:
            # mid = partition(arr, low, high)                   # ��midΪ�ָ�㡾�����ѡ��pivot��
            mid = randomPartition(arr, low, high)               # ��midΪ�ָ�㡾���ѡ��pivot��
            if mid == k-1:                                      # ��kСԪ�ص��±�Ϊk-1
                return arr[mid]                                 #���ҵ������ء�
            elif mid < k-1:
                return topKSplit(arr, mid+1, high, k)           # �ݹ��mid�Ҳ�Ԫ�ؽ�������
            else:
                return topKSplit(arr, low, mid-1, k)            # �ݹ��mid���Ԫ�ؽ�������
        
        n = len(nums)
        return topKSplit(nums, 0, n-1, n-k+1)                   # ��k��Ԫ�ؼ�Ϊ��n-k+1СԪ��


# ����֮��
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
		
		
# ����֮��
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


		
# ��������
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


# �����Ʊ��ʱ��
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



# �ϲ���������

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
		
		
		
# �ϲ��ǵݼ�����
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


		
# ����֮��
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
				
				
# �������������
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

# �������������
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

# �����������ȣ�
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

# ��������
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


# ������Ӵ�
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s
        
        max_len = 1
        begin = 0
        # dp[i][j] ��ʾ s[i..j] �Ƿ��ǻ��Ĵ�
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        
        # ���ƿ�ʼ
        # ��ö���Ӵ�����
        for L in range(2, n + 1):
            # ö����߽磬��߽���������ÿ��Կ���һЩ
            for i in range(n):
                # �� L �� i ����ȷ���ұ߽磬�� j - i + 1 = L ��
                j = L + i - 1
                # ����ұ߽�Խ�磬�Ϳ����˳���ǰѭ��
                if j >= n:
                    break
                    
                if s[i] != s[j]:
                    dp[i][j] = False 
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                
                # ֻҪ dp[i][L] == true �������ͱ�ʾ�Ӵ� s[i..L] �ǻ��ģ���ʱ��¼���ĳ��Ⱥ���ʼλ��
                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin:begin + max_len]
