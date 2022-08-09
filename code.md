[TOC]

# LRU

用hashmap(dict类)和链表构造一个双向数据结构，即Python中的OrderedDict类。但需要自己去构造该类

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208090950122.png" alt="image-20220805141459339" style="zoom:67%;" />

```python
class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        # 哈希表+链表 最近访问的节点在尾节点，最久未访问的节点在头节点
        self.capacity = capacity
        self.hashmap = {}
        # 新建两个节点 head 和 tail
        self.head = ListNode()
        self.tail = ListNode()
        # 初始化链表为 head <-> tail
        self.head.next = self.tail
        self.tail.prev = self.head

    # 封装方法
    # 将节点由头移到尾（最近访问）
    def move_node_to_tail(self,key):
        # 从哈希表中拿节点
        node = self.hashmap[key]
        # 删除该节点
        node.prev.next = node.next
        node.next.prev = node.prev
        # 插入该节点到尾节点之前
        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node


    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        # 如果key存在于hashmap
        if key in self.hashmap:
            self.move_node_to_tail(key)
        # hashmap中存在就返回key，不存在就返回-1
        res = self.hashmap.get(key,-1)
        if res == -1:
            return res
        else:
            return res.value
    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        # 如果key在hashmap中,改变value，并放到链表尾部（最近访问）
        if key in self.hashmap:
            self.hashmap[key].value = value
            self.move_node_to_tail(key)
        else:
            #在插入之前判断容量
            if len(self.hashmap) == self.capacity:
                #删除头节点对应的字典键值
                self.hashmap.pop(self.head.next.key)
                # 删除最久未使用的元素：链表头部元素
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
            # 此时保证capacity不超过，且key不存在表中，则插入：
            # 字典先插入
            newNode = ListNode(key,value)
            self.hashmap[key] = newNode
            # 链表从尾部插入
            newNode.prev = self.tail.prev
            newNode.next = self.tail
            self.tail.prev.next = newNode
            self.tail.prev = newNode
```



#  排序

## 快排
```python
## 时间复杂度：基于随机选取主元的快速排序时间复杂度为期望 O(nlogn)，其中 n为数组的长度。详细证明过程可以见《算法导论》第七章，这里不再大篇幅赘述。

## 空间复杂度：O(h)O(h)，其中 hh 为快速排序递归调用的层数。我们需要额外的 O(h)O(h) 的递归调用的栈空间，由于划分的结果不同导致了快速排序递归调用的层数也会不同，最坏情况下需 O(n)O(n) 的空间，最优情况下每次都平衡，此时整个递归树高度为 \log nlogn，空间复杂度为 O(\log n)O(logn)。

class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def quicksort(nums,l,r):
            if l >= r:
                return 

            # 随机选择一个pivot
             index = random.randint(l, r)
             pivot = nums[index]
             #将pivot换到最左端
             nums[l],nums[index] = nums[index],nums[l]
             i,j = l,r
             while i < j:
                 while i < j and nums[j] >= pivot:
                     j -= 1 
                 while i < j and nums[i] <= pivot:
                     i += 1
                 if i != j:
                    # 未重合时交换元素
                     nums[i],nums[j] = nums[j],nums[i]
             #将最左端的pivot放置i与j重合的地方
             nums[l],nums[i] = nums[i],nums[l]

            quicksort(nums,l,i-1)
            quicksort(nums,i+1,r)
        quicksort(nums,0,len(nums) - 1)
        return nums
```
## 归并排序
```python
时间复杂度：O(nlogn)。由于归并排序每次都将当前待排序的序列折半成两个子序列递归调用，然后再合并两个有序的子序列，而每次合并两个有序的子序列需要 O(n)的时间复杂度，所以我们可以列出归并排序运行时间 T(n)T(n) 的递归表达式：
空间复杂度：O(n)。我们需要额外 O(n)空间的tmp数组，且归并排序递归调用的层数最深为nlogn，所以我们还需要额外的 O(logn) 的栈空间，所需的空间复杂度即为 O(n+logn) = O(n)O(n+logn)=O(n)。

class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def merge_sort(nums, l, r):
            if l == r:
                return
            mid = (l + r) // 2
         # 最上面一层递归的结果是数组nums[l:mid+1]（左端点为l,右端点为mid）和nums[mid+1:r+1]（左端点为mid+1,右端点为r）两个有序数组的合并
            merge_sort(nums, l, mid)
            merge_sort(nums, mid + 1, r)
            tmp = []
            i, j = l, mid + 1 # 设置左端点
            while i <= mid and j <= r:
                if nums[j] < nums[i]:
                    tmp.append(nums[j])
                    j += 1
                else:
                    tmp.append(nums[i])
                    i += 1
            if i > mid: #连接未访问的有序数组元素
                tmp = tmp + nums[j:r+1]
            elif j > r:
                tmp = tmp + nums[i:mid+1]
            nums[l:r+1] = tmp # 合并两个数组
        merge_sort(nums, 0, len(nums) - 1)
        return nums
```
## 堆排序
```python
# 时间复杂度：O(nlogn)。初始化建堆的时间复杂度为 O(n)，建完堆以后需要进行 n-1次调整，一次调整（即 maxHeapify） 的时间复杂度为 O(logn)，那么 n-1n−1 次调整即需要 O(n\log n)O(nlogn) 的时间复杂度。因此，总时间复杂度为 O(n+nlogn)=O(n+nlogn)=O(nlogn)。


class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 大根堆排序
        # 根节点是0开始
        def maxheap(nums,index,end):
            # 访问index的子节点
            j = 2*index + 1
            while j <= end:
                # 选择index 较大的那个子节点
                if j <= end-1 and nums[j] < nums[j+1]:
                    j += 1
                # 如果该节点值小于最大子节点，则交换，让最大子节点作为根节点
                if j <= end and nums[index] < nums[j]:
                    nums[index],nums[j] = nums[j],nums[index]
                # 遍历到子节点
                    index = j
                    j = 2*index + 1
                else:
                    break 
        # 从下到上构建大根堆，找到最后一个非叶子节点 n//2 + 1
        n = len(nums)
        for i in range(n//2 + 1,-1,-1):
            maxheap(nums,i,n-1)
        # 遍历完以后已经构建出了大根堆,从上到下下沉，交换0和j，并且重新构建除去j的大根堆
        for j in range(n-1,-1,-1):
            nums[0],nums[j] = nums[j],nums[0]
            maxheap(nums,0,j-1)
        return nums
```
## 第k大的数
```python
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
```

# 链表
## 相交链表
```python
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
```

## 反转链表
```python
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
```
## 反转链表2
```python
class Solution:
    # 头插法
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        # 设置 dummyNode 是这一类问题的一般做法
        dummy_node = ListNode(-1)
        dummy_node.next = head
        pre = dummy_node # left之前的节点，计作0节点
        for _ in range(left - 1):
            pre = pre.next

        cur = pre.next # 左端节点
        for _ in range(right - left):
            next = cur.next # 第一次循环，next = 2节点
            cur.next = next.next # 第一次循环，1节点的next为3节点
            next.next = pre.next # 第一次循环，2节点的next为1节点
            pre.next = next # 第一次循环，0节点的next为2节点： 0->2->1-> 3 ->4 -> 把2插到1前面
            # 第二次循环把3插到2前面
        return dummy_node.next
```
## 删除链表指定值的节点
```python
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = ListNode(0,head)
        cur = dummy
        while cur.next:       # 防止cur遍历到尾节点 
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next
```
## 删除倒数第n个节点 记住要加头部的懒节点
```python
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
```
## 链表排序（用到了归并排序思想）
```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def sortFunc(head,tail):
            # 控制好递归出口
            if not head:
                return head
            # 最底层：
            # 当链表只剩下两个节点的时候，head.next = tail，此时将head断开，链表1是head，链表2是mid，进行Merge排序
            # 以此类推...
            if head.next == tail:
                head.next = None
                return head
            #找出中间节点
            fast = head
            slow = head
            # fast每次移动两个节点，slow每次移动一个节点
            while fast != tail:
                fast = fast.next
                slow = slow.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortFunc(head,mid),sortFunc(mid,tail))
        # 合并两个有序链表
        def merge(head1,head2):
            tmp = ListNode(0)
            res,cur1,cur2 = tmp,head1,head2
            while cur1 and cur2:
                if cur1.val > cur2.val:
                    res.next = cur2
                    cur2 = cur2.next
                else:
                    res.next = cur1
                    cur1 = cur1.next
                res = res.next
            if cur1:
                res.next = cur1
            elif cur2:
                res.next = cur2
            return tmp.next
        return sortFunc(head, None)
```

# 双指针

## 二分查找

```python
class Solution(object):
    def search(self, nums, target):
        l = 0
        r = len(nums) - 1
        for i in range(len(nums)):
            mid = (l - r)/2 + r
            if target > nums[mid]: l = mid + 1
            elif target< nums[mid]: r = mid - 1
            else:
                return mid
        return -1
```

## x的平方根

```PYTHON
class Solution:
    def mySqrt(self, x: int) -> int:
        l,r,ans = 0,x,-1
        while l <= r:
            mid = (l + r)//2
            if mid*mid <= x:
                res = mid
                l = mid + 1
            elif mid*mid > x:
                r = mid -1
        return res
```

## 三数之和
```python
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
```


## 四数之和
```python
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
```

# 贪心算法

## 合并区间

```PYTHON
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if len(intervals) == 0: return intervals
        res = []
        # 按左区间排序
        intervals.sort(key=lambda x: x[0])
        res.append(intervals[0])
        for i in range(1,len(intervals)):
            # 拿出上一个被合并的区间
            last = res[-1]
            if intervals[i][0] <= last[1]:# 如果第i个区间的左边界小于第i-1个区间的右边界
                res[-1] = [last[0],max(last[1],intervals[i][1])]
            else:
                res.append(intervals[i])
        return res
```



# 合并有序链表

```python
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
```

# 合并非递减数组
``` python
    def merge(self, nums1, m, nums2, n):
        tmp = []
        i = 0
        j = 0
        while i < m and j < n:
            if nums1[i] < nums2[j]:
                tmp.append(nums1[i])
                i += 1
            else:
                tmp.append(nums2[j])
                j += 1
        if i == m:
            tmp = tmp + nums2[j:]
        elif j == n:
            tmp = tmp + nums1[i:m]
        nums1[:] = tmp
        return nums1
```

# 哈希表
## 两数之和
```python
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
```

# 二叉树
## 二叉树中序遍历
```python
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
```
## 二叉树层序遍历
### 递归法
```python
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
```
### 迭代法
```python
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
```
## 二叉树锯齿形层序遍历
```python
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
```
## 二叉树右视图
```python
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
```
## 二叉树最大深度
```python
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
```
## 二叉树最小深度
```python
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
```
## 迭代法的最小深度
```python
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
    
        return 
```
# 最长回文子串
```python
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
```

# 回溯
## 全排列
```python
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
```
## 带有target的组合 如[2,2,2,3]之和等于7的所有组合
```python
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
```

# 动态规划
## 带障碍的路径规划
```python
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
```

## 爬楼梯
```python
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
```

## 拆分两数
```python
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
```
## 最大子序和
```python
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
```
## 买入股票的时机
```python
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
```
# dfs岛屿问题
```python
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
```
# 岛屿周长
```python
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
```



