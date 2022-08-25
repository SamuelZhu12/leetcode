# 目录
[TOC]
# 栈与队列

## 最小栈

https://leetcode.cn/problems/min-stack/solution/zui-xiao-zhan-by-leetcode-solution/

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]

    def push(self, x: int) -> None:
        self.stack.append(x)
        self.min_stack.append(min(x, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]

```

## 用栈实现队列

https://leetcode.cn/problems/implement-queue-using-stacks/submissions/

```python
class MyQueue:

    def __init__(self):
        self.que = []
        self.back_que = []
    def push(self, x: int) -> None:
        self.que.append(x)

    def pop(self) -> int:
        if not self.que: return None
        else:
            while self.que:
                self.back_que.append(self.que.pop()) # 将栈的元素倒到另一个栈中，栈顶元素即是之前的栈底元素，也是队列头元素
            res = self.back_que.pop()
            while self.back_que:
                self.que.append(self.back_que.pop())# 将栈中元素倒回去
            return res

    def peek(self) -> int:
        if not self.que: return None
        else:
            return self.que[0]

    def empty(self) -> bool:
        if self.que or self.back_que:
            return False
        else: return True
```





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

# 数组

## 轮转数组

https://leetcode.cn/problems/rotate-array

```python
class Solution(object):
    def rotate(self, nums, k):
        def rotateNums(nums,i,j):
            while i <= j:
                nums[i],nums[j] = nums[j],nums[i]
                i += 1
                j -= 1
        k = k % len(nums)
        rotateNums(nums,0,len(nums)-1)
        rotateNums(nums,0,k-1)
        rotateNums(nums,k,len(nums)-1)
        return nums
```

## 寻找旋转排序数组中的最小值

https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/

```python
class Solution(object):
    def findMin(self, nums):
        l = 0
        r = len(nums) - 1
        while l < r:
            mid = (l + r)//2
            # mid数字大于右端数字，说明最小值在右侧
            if nums[r] < nums[mid]:
                l = mid + 1
            # mid数字小于等于右侧数字，说明最小值在左侧
            else:
                r = mid
        return nums[l]
```



## 搜索旋转排序数组2⃣️

https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/

```python
class Solution(object):
    def search(self, nums, target):
        if not nums:
            return False
        l = 0
        r = len(nums) - 1
        while l <= r:
            # 消除重复元素的影响
            while l < r and nums[l] == nums[l+1]:
                l += 1
            while l < r and nums[r] == nums[r-1]:
                r -= 1
            # 取中点
            mid = (l+r)//2
            
            # 判断左右有序情况
            # 如果左侧有序
            if nums[mid] == target:
                return True
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            # 如果右侧有序
            else:
                if nums[mid] < target <= nums[len(nums)-1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return False
                    
```

## 搜索旋转数组（面试版）

https://leetcode.cn/problems/search-rotate-array-lcci/

```python
class Solution(object):
    def search(self, arr, target):
        if arr[0] == target:
            return 0
        l = 0
        r = len(arr) - 1
        # 左侧的最小值 > 右侧的最大值
        while l <= r:
            mid = (l + r) // 2
            # 中间值等于目标值，将右边界移到中间，因为左边可能还有相等的值
            if arr[mid] == target:
                while mid > 0 and arr[mid] == arr[mid-1]:
                    mid -= 1
                return mid
            # 判断mid在左侧还是右侧有序数组
            # mid大于左侧最小值，说明在左侧
            if arr[r] < arr[mid]:
                if arr[l] <= target < arr[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            # mid 小于左侧最小值，说明在右侧
            elif arr[r] > arr[mid]:
                if arr[mid] <target <= arr[r]:
                    l = mid + 1
                else:
                    r = mid - 1
            # 中间数字等于左边数字时，左边界右移
            else:
                r -= 1
        return -1
```





# 字符串

## 翻转字符串2⃣️

https://leetcode.cn/problems/reverse-string-ii/

```python
class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        i = 0
        while i < len(s):
            s = s[:i] + s[i:i+k][::-1] +s[i+k:]
            i += 2*k
        return s
```

## 替换空格

https://leetcode.cn/problems/ti-huan-kong-ge-lcof/

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        counter = s.count(' ')
        
        res = list(s)
        # 每碰到一个空格就多拓展两个格子，1 + 2 = 3个位置存’%20‘
        res.extend([' '] * counter * 2)
        
        # 原始字符串的末尾，拓展后的末尾
        left, right = len(s) - 1, len(res) - 1
        
        while left >= 0:
            if res[left] != ' ':
                res[right] = res[left]
                right -= 1
            else:
                # [right - 2, right), 左闭右开
                res[right - 2: right + 1] = '%20'
                right -= 3
            left -= 1
        return ''.join(res)
```

## 颠倒字符串中的单词

https://leetcode.cn/problems/reverse-words-in-a-string/submissions/

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ''
        start = 0
        end = 0
        # 去除开头和结尾的空格，截取开头第一个遇到的非空格
        for i in range(len(s)):
            if s[i] != ' ':
                s = s[i:]
                break
        # 截取倒排第一个遇到的非空格
        for i in range(len(s)-1,-1,-1):
            if s[i] != ' ':
                s = s[:i+1]
                break
        # 遇到空格 且空格去重
        while end < len(s):
            if s[end] == ' ' and s[end] != s[end-1]:
                res = ' ' + s[start:end] + res
                start = end + 1
            # 存在空格的时候，start指针进行移动
            if s[end] == ' ' and s[end] == s[end-1]:
                start += 1
            if end == len(s) - 1:
            # 最后一个单词
                res = s[start:end+1] + res
            end += 1
        return res
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
时间复杂度：O(nlogn)。由于归并排序每次都将当前待排序的序列折半成两个子序列递归调用，然后再合并两个有序的子序列，而每次合并两个有序的子序列需要 O(n)的时间复杂度，所以我们可以列出归并排序运行时间 T(n)的递归表达式：
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
                tmp = tmp + nums[j:r+1] # 注意不会包含右边界，故+1
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
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # 构建大根堆
        def maxHeap(nums,index,end):
          # 第一个左子结点
            j = 2*index + 1
            # 如果还没到end，则一直往下调整
            while j <= end:
              # 比较左右子结点
                if j <= end - 1 and nums[j] < nums[j+1]:
                    j += 1
             	# 比较最大的子结点和根节点大小，如果大于根节点则交换
                if j <= end and nums[index] < nums[j]:
                    nums[index],nums[j] = nums[j],nums[index]
                    index = j
                    j = 2*index + 1
                else: break
        n = len(nums)
        # 从最后一个非叶子节点构建
        for i in range(n//2 + 1,-1,-1):
            maxHeap(nums,i,n-1)
       	# 举特例，如果k = 1，则应该是n-1倒序遍历到n-2，k=2应该是n-1倒序遍历到n-3，并且取最后一次取的最大数，即是我们所要的第k大的数
        for j in range(n-1,n-k-1,-1):
            nums[0],nums[j] = nums[j],nums[0]
            maxHeap(nums,0,j-1)
        return nums[j]
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

## 两数相加

https://leetcode.cn/problems/add-two-numbers/

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# 题目要求：链表l1 2 -> 4 -> 3 和链表l2 5 -> 6 -> 4，计算342+564 = 807 输出为链表 7 -> 0 -> 8 
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = cur = ListNode(0) #  建立一个新的头节点来返回结果
        s = 0
        while l1 or l2 or s != 0:
            s += (l1.val if l1 else 0) + (l2.val if l2 else 0) # 链表对应位置节点为空时则补0
            cur.next = ListNode(s % 10) # 存储链表对应位置节点val之和，通过取模mod来取进位情况的个位数存储到下一个节点
            cur = cur.next # 三个链表同时访问下一个节点
            if l1:l1 = l1.next
            if l2:l2 = l2.next
            s = s // 10 #将进位数加到下一轮的计算中 '//'取的是除完以后的整数部分
        return dummy.next
```

## 重排链表

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if not head: return None
        nodeStack = []
        dummy = ListNode(0,head)
        cur1 = dummy.next
        while cur1.next:
            nodeStack.append((cur1,cur1.next)) # 要将前一个节点也带上
            cur1 = cur1.next
        cur2 = dummy.next
        while cur2.next and cur2.next.next: # 操作涉及到三个节点，所以确保后面两个节点不为空
            prenode,node = nodeStack.pop()
            node.next = cur2.next # 尾部 -> cur.next
            cur2.next = node # cur ->尾部
            prenode.next = None # 倒数第二个节点 -> None 这样来处理新的尾部节点
            cur2 = node.next
        return dummy.next
```

## 复制带随机指针的链表

https://leetcode.cn/problems/copy-list-with-random-pointer/

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
# 用哈希表存储新的节点。注意是要深拷贝
class Solution(object):
    def copyRandomList(self, head):
        if not head:return None
        dummy = Node(0,head)
        cur = dummy.next
        nodeMap = {}
        while cur:
            newNode = Node(cur.val)
            nodeMap.update({cur:newNode})
            cur = cur.next
        cur1 = dummy.next
        while cur1:
            if cur1.next:
                nodeMap[cur1].next = nodeMap[cur1.next]
            if cur1.random:
                nodeMap[cur1].random = nodeMap[cur1.random]
            cur1 = cur1.next
        return nodeMap[dummy.next]
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

## 和大于等于 target 的最短子数组

https://leetcode.cn/problems/2VG8Kg/

```python
# 类似于滑动窗口
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        if sum(nums) < target : return 0
        res = len(nums)
        start,end = 0, 1
        while end <= len(nums):
            if sum(nums[start:end]) >= target:
                res = min(res,end - start)
                start += 1
            else:
                end += 1
        return res
```

## 字符串相加

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        m = len(num1) - 1
        n = len(num2) - 1
        res = ''
        carry = 0
        while m >=0 or n >= 0 or carry > 0: # carry>0用于补齐最高位有进位的情况
            a = num1[m] if m >=0 else 0 # 为空的字符串补0
            b = num2[n] if n >=0 else 0 
            s = int(a)+int(b) + carry # 加入carry位
            s1 = s % 10 # 取模得到相加的个位数
            res = str(s1) + res #拼接字符
            carry  = s//10
            m -= 1
            n -= 1
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

## 分发饼干

```python
# g为小孩的胃口值：[7,8,9,10]
# s为饼干的尺寸，当尺寸>胃口值时可满足一个小孩的胃口，找到可满足小孩数量的最大值 s = [5,6,7,8]
# 将两个数组倒序排列，分别比较两个数组的最大值
class Solution(object):
    def findContentChildren(self, g, s):
        g = sorted(g,reverse=True)
        s = sorted(s,reverse=True)
        res = 0
        for i in range(len(g)):
            if res == len(s): return res
            if s[res] >= g[i]:
                res += 1
        return res
```

## 摆动序列

```python
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 如果是[2,3,6]，则默认[2,3]是摆动序列，因为有一个最低峰和一个最高峰
        # 计算有多少个波峰和波谷
        res = 1 # 默认
        pre = 0
        cur = 0
        for i in range(0,len(nums) - 1):
            cur = nums[i+1] - nums[i]
            if (cur < 0 and pre >=0) or (cur > 0 and pre <= 0): # 这里一定要带等号，这样数组个数为2时保证res = 2
                res += 1
                pre = cur
        return res
```

## 买卖股票的最佳时机

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0
        for i in range(len(prices)-1):
        # 对局部而言，每次都取差值大于0，即利润大于0的时间
        # profit[0:3] = profit[3] - profit[2] + profit[2] - profit[1] + profit[1] - profit[0]
            profit = prices[i+1] - prices[i]
            if profit >= 0:
                res += profit
        return res
```

## 跳跃游戏

https://leetcode.cn/problems/jump-game/

```python
## 1 0 1 3 4
## 每个值代表可向前跳跃的最大值，能否跳跃到终点？
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        cover = 0
        i = 0
        # 对cover可覆盖到的地方进行遍历,python中for不支持动态修改循环变量
        while i <= cover:
            # 每一个点的覆盖范围，更新最大跳跃覆盖范围
            cover = max(cover,nums[i] + i)
            if cover >= len(nums)-1:
                return True
            i += 1

        return False
```

## 跳跃游戏2

https://leetcode.cn/problems/jump-game-ii/

```python
## 假设这组数组可以最终跳到终点，求最小跳跃次数
## 在当前覆盖范围之内，更新该范围内每个点的最大覆盖范围，取最大值；当访问到当前范围最右侧的点时如果无法跳跃到中点，则进行下一个范围的覆盖,res++
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curcover = 0
        nextcover = 0
        res = 0
        if len(nums) == 1:
            return 0
        for i in range(len(nums)):
            nextcover = max(nextcover,i+nums[i]) # 当前覆盖范围内的下一步跳跃最大覆盖范围
            if i == curcover:# 如果走到右边界还没有到最大覆盖范围，则需要跳一步来更新最大覆盖范围
                res += 1
                curcover = nextcover
                if nextcover >= len(nums) - 1:break
            else:
                continue
        return res
```

## 加油站

https://leetcode.cn/problems/gas-station/

![image-20220807225947320](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/image-20220807225947320.png)

```python
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        rest = 0 #油箱剩余油量
        restMin = 0 #油箱最小剩余油量
        restSum = 0
        res = 0
        resList = []
        for i in range(len(gas)):
            rest = gas[i] - cost[i] # 从第i个点出发能否到达下一个点
            restSum += rest # rest的累计值
            if restMin > restSum:
                restMin = restSum
                res = i+1 # 如果restSum>0，则说明一定可以满足环绕条件，而最小的restSum的下一个加油站就是开始的加油站，因为前面的restSum是小于0的，往后面走一定可以把前面的坑补上，而如果先走了正确起点加油站的后面的加油站，中途会因油不足而无法满足条件

        if restSum < 0: return -1 # 此时不管从哪个点出发都无法环绕一圈
        
        if restMin >= 0: return 0 # 此时说明每次的备油量都是充足的，都可以从0出发到达下一个点

        return res
```

## K 次取反后最大化的数组和

https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/

```python
class Solution(object):
    def largestSumAfterKNegations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums = sorted(nums,key = abs,reverse = True)
        for i in range(len(nums)):
            if nums[i] < 0 and k > 0:
                nums[i] = -nums[i]
                k -= 1
        if k > 0:
            nums[-1] *= pow(-1,k)
        return sum(nums)
```

## 分发糖果

https://leetcode.cn/problems/candy/

```python
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        # 局部最优:右边的孩子比左边的孩子大，则多一颗糖果。左边的孩子比右边的大，则左边比右边多一颗糖果；全局最优：所有的孩子中，相邻的孩子中表现好的会多分配一些糖果。 没有必要考虑相等的因素，因为题目中没有提及，只需要满足右边界ratings[i+1]>ratings[i]则res[i+1] = res[i] + 1；左边界ratings[i] > ratings[i+1]，res[i]= max(res[i],res[i+1]+1)
        res = [1] * len(ratings)
        for i in range(len(ratings)-1):
            if ratings[i+1] > ratings[i]:
                res[i+1] = res[i] + 1
        for i in range(len(ratings)-2,-1,-1):
            if ratings[i] > ratings[i+1]:
                res[i] = max(res[i],res[i+1]+1) # 维护满足右边界的res条件

        return sum(res)
```

## 柠檬水找零

https://leetcode.cn/problems/lemonade-change/submissions/

```python
# 局部：解决每次的找零问题，分支付5、10、20元三种情况解决，当支付金额为20元时，要么找10+5，要么找5+5+5，但是5更利于找钱，所以优先考虑10+5，尽可能少消耗5元，手上的5元越多，满足每一种支付金额的可能性就越大，这就是局部最优的情况。
# 整体：每一次都能顺利找钱，最后可以满足条件
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five, ten, twenty = 0,0,0
        for i in range(len(bills)):
            if bills[i] == 5:
                five += 1
            elif bills[i] == 10:
                if five < 1 : return False
                five -= 1
                ten += 1
            elif bills[i] == 20:
                if ten > 0 and five > 0:
                    ten -= 1
                    five -= 1
                elif five > 2:
                    five -= 3
                else:return False
        return True
```

## 根据身高重建队列

https://leetcode.cn/problems/queue-reconstruction-by-height/

<img src="../../Library/Application Support/typora-user-images/image-20220809194334683.png" alt="image-20220809194334683" style="zoom:67%;" />

```python
class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        # 先按照身高h排列由大到小排列，再按照序号k由小到大排列
        people.sort(key=lambda x: (-x[0],x[1]))
        res = []
        # 排列结束后，发现每一次按照由小到大插入res中的顺序都满足
        for i in range(len(people)):
            res.insert(people[i][1],people[i])
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
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def getDepth(node):
            if not node:
                return 0
            # 解决只有一边子树的情况
            if not node.left and node.right: 
                return getDepth(node.right) + 1
            if node.left and not node.right:
                return getDepth(node.left) + 1
            return min(getDepth(node.left),getDepth(node.right)) + 1
        return getDepth(root)
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
## 翻转二叉树

```python
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        def inverse(node):
            if not node:
                return
            
            tmp = node.right
            node.right = node.left
            node.left = tmp
            inverse(node.left)
            inverse(node.right)
        inverse(root)
        return root
```

## 对称二叉树

```python
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def compare(left,right):
          # 先判断为空的情况
            if not left and right: return False 
            elif left and not right: return False
            elif not left and not right: return True
            elif left.val != right.val: return False
          # 不符合上述带空值或不相等的情况，就计算该节点的左右子节点是否相等
            outside = compare(left.left,right.right)
            inside =  compare(left.right,right.left)
            return outside and inside

        if not root:
            return True
        return compare(root.left,root.right)
```

## 平衡二叉树

https://leetcode.cn/problems/balanced-binary-tree/

```python
class Solution(object):
    def isBalanced(self, root):
        def traverse(node):
            if not node:
                return 0
            leftHeight = traverse(node.left) # 左子树高度
            rightHeight = traverse(node.right) # 右子树高度
            if leftHeight == -1: # 如果左子树不满足平衡二叉树，就返回-1
                return -1
            if rightHeight == -1:# 如果右子树不满足平衡二叉树，就返回-1
                return -1
            if abs(leftHeight - rightHeight) > 1: # 如果左右子树高度差 >1 本层就返回-1
                return -1
            else:
                return 1 + max(traverse(node.left),traverse(node.right)) # 如果左右子树高度差<1 即满足平衡条件，则
        flag = traverse(root)
        return True if flag != -1 else False
```

## 二叉树的所有路径

https://leetcode.cn/problems/binary-tree-paths/

```python
class Solution(object):
    def binaryTreePaths(self, root):
      
        res = []
        def traverse(node,path,res):
           # path连接每个node的值
            path += str(node.val)
            if not node.left and not node.right:
                res.append(str(path))
            
            if node.left:
              # 由于在path加上了"->"，实际上是隐藏回溯了，因为回溯时撤销的也是"->"
                traverse(node.left,path + "->",res)
            if node.right:
                traverse(node.right,path + "->",res)
            return res

        return traverse(root,'',res)
```

## 左叶子之和

```python
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def traverse(node):
            if not node:
                return 0
            res = 0
            # 要从左叶子节点的父节点来进行判断，否则无法判断是否是左叶子还是右叶子
            if node.left and (not node.left.left) and (not node.left.right): 
                res = node.left.val
            return res + traverse(node.left) + traverse(node.right)
        return traverse(root)
```

## 路径总和

https://leetcode.cn/problems/path-sum

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        def traverse(node,pathSum,targetSum):
            pathSum += node.val # 对每个节点进行处理，加上每个节点的值，隐形回溯了
            if (not node.left) and (not node.right) and pathSum == targetSum:
                return True
            if node.left: # 该节点的左子树如果有一条路径可以满足target，则返回true
                if traverse(node.left,pathSum,targetSum):
                    return True
                
            if node.right:
                if traverse(node.right,pathSum,targetSum):
                    return True
                
            return False
        return traverse(root,0,targetSum)
```

## 路径总和2

https://leetcode.cn/problems/path-sum-ii/

```python

class Solution(object):
    def pathSum(self, root, targetSum):
        if not root:
            return []
        def traverse(node,path):
            if not node.left and not node.right and sum(path) == targetSum: # 添加的条件
                res.append(path[:])
            if node.left: # 节点的回溯处理
                path.append(node.left.val)
                traverse(node.left,path)
                path.pop()
            if node.right:
                path.append(node.right.val)
                traverse(node.right,path)
                path.pop()
            return res
        path = [root.val]
        res = []
        return traverse(root,path)
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

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/fig14.png" alt="img" style="zoom:50%;" />

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
            for i in range(first, n): # 每一层的层序遍历
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1) # 每一层的下一层遍历
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first] # 保留 first-1 的状态，让i+1，到first同一层的另外一个子节点
        
        n = len(nums)
        res = []
        backtrack()
        return res
```
## 组合

https://leetcode.cn/problems/combinations

![image-20220813235359765](C:\Users\56268\Desktop\leetcode\leetcode\code.assets\image-20220813235359765.png)

```py
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        result = []
        path = []
        def backtracking(n, k, startidx):
            if len(path) == k:
                result.append(path[:])
                return

            # 剪枝， 最后k - len(path)个节点直接构造结果，无需递归
            last_startidx = n - (k - len(path)) + 1
            result.append(path + [idx for idx in range(last_startidx, n + 1)])

            for x in range(startidx, last_startidx):
                path.append(x)
                backtracking(n, k, x + 1)  # 递归
                path.pop()  # 回溯

        backtracking(n, k, 1)
        return result
```



## 组合总和

https://leetcode.cn/problems/combination-sum/

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/1598091943-hZjibJ-file_1598091940241.png)

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

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        path = []
        sum_ = 0
        res = []
        def backtracking(first,sum_):
            if sum_ == target:
                res.append(path[:])
                return 
            if sum_ > target:
                return
            for i in range(first,len(candidates)):
                sum_ += candidates[i]
                path.append(candidates[i])
                backtracking(i,sum_)
                sum_ -= candidates[i]
                path.pop()
            return res
        return backtracking(0,sum_)
```



## 分隔回文串

```python
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        path = []
        res = []
        def backtracking(first,path,res):
            if first >= len(s): # 分割线到字符串尾则完成一次分割
                res.append(path[:])
                return
            for i in range(first,len(s)):
                temp = s[first:i+1] # 截取该层字符串，从这一层的first到i为截取部分
                if temp == temp[::-1]: # 判断是否为回文串
                    path.append(temp)
                    backtracking(i + 1,path,res)
                    path.pop()
                else:
                    continue
            return res
        return backtracking(0,path,res)
```

## 复原ip地址

https://leetcode.cn/problems/restore-ip-addresses/

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208171144860.png" alt="93.复原IP地址" style="zoom:50%;" />

```python
# 本质上是对ip字符串插入'.'，也就是分割三次，并对每次的分割结果进行判断
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res= []
        def isIp(ipStr, start, end): 
            if start > end:
                return False
            if ipStr[start] == '0' and start != end: # 0开头，不合法
                return False
            if not 0 <= int(ipStr[start:end+1]) <= 255: # 当前截取字符串范围不合法
                return False
            return True

        def backtracking(first, pointNum, s):
            if pointNum == 3: # 当'.'的数量等于3时，返回插入好'.'的字符串
                if isIp(s,first, len(s)-1):
                    res.append(s[:])
                return
            for i in range(first, len(s)): # 循环控制横向遍历
                if isIp(s, first, i):  # [first,i]是截取的字符串范围
                    s = s[:i + 1] + '.' + s[i + 1:] # 对字符串整体进行'.'插入
                    backtracking(i+2, pointNum + 1, s) # 递归纵向遍历，由于插入了点，所以i+2，且点计数器变量+1
                    s = s[:i + 1] + s[i+2:] # 复原
                else: # 如果s[first,i+1]不满足ip要求，则break跳出该层的横向遍历，因为位数超过
                    break
            return res
        return backtracking(0,0,s)
```

## 电话号码的字母组合

https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

```python
class Solution(object):
    def letterCombinations(self,digits):
        tb = {'2': "abc", '3': "def", '4': "ghi", '5': "jkl", '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz"} # 设置号码映射表
        res = []
        path = ''
        if not digits: return []
        def backtracking(digits, first,path):
            if first == len(digits): # 组合数等于输入数字个数则加入
                res.append(path)
                return
            letters = tb[digits[first]] # 获取该节点的数字
            for letter in letters: # 遍历该节点所有的字母
                path = path + letter
                backtracking(digits,first + 1,path) # 下一层是下一个数字的字母，跟上一行的代码进行组合
                path = path[:-1]

            return res

        return backtracking(digits, 0,path)
```



## 组合总和

https://leetcode.cn/problems/combination-sum

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        path = []
        sum_ = 0
        res = []
        def backtracking(first,sum_):
            if sum_ == target:
                res.append(path[:])
                return 
            if sum_ > target:
                return
            for i in range(first,len(candidates)): # 二叉树横向遍历
                sum_ += candidates[i]
                path.append(candidates[i])
                backtracking(i,sum_) # 由于可以取自身的值，所以从i开始。二叉树纵向遍历
                sum_ -= candidates[i]
                path.pop()
            return res
        return backtracking(0,sum_)
```

## 组合总和Ⅱ

https://leetcode.cn/problems/combination-sum-ii/

```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        path = []
        sum_ = 0
        res = []
        candidates = sorted(candidates)
        def backtracking(first,sum_):
            if sum_ == target:
                res.append(path[:])
                return 
            if sum_ > target:
                return
            for i in range(first,len(candidates)):
                if i > first and candidates[i] == candidates[i-1]: # 类似于三数之和、四数之和，将同一层中相同的数字给去掉，如target = 8 [2,5,1]和另一个[2]构成的[2,5,1]
                    continue
                sum_ += candidates[i]
                path.append(candidates[i])
                backtracking(i+1,sum_)
                sum_ -= candidates[i]
                path.pop()
            return res
        return backtracking(0,sum_)
```

## 子集

https://leetcode.cn/problems/subsets/

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208161126643.png" alt="78.子集" style="zoom:67%;" />

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        path = []
        def backtracking(first,res,path):
            if len(path) <= first: # 不用做break操作，因为first >= nums.size()，本层for循环本来也结束了
                res.append(path[:])
            for i in range(first,len(nums)):
                path.append(nums[i])
                backtracking(i+1,res,path)
                path.pop()
            return res
        return backtracking(0,res,path)

# dp做法
class Solution(object):
    def subsets(self, nums):
        res = [[]]

        for i in range(len(nums)):
            for j in res[:]:
                tmp = j[:] # 每次遍历上一次的res集合，分别插入本轮的nums[i]，再插入res集合中
                tmp.append(nums[i])
                res.append(tmp[:])
        return res
```





# 动态规划

## 不同路径

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for i in range(n):
            dp[0][i] = 1
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
```

## 不同路径 II （带障碍的路径规划）

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

## 整数拆分



```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in range(3, n + 1):
            # 假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i），则有以下两种方案：
            #当j固定时，有 dp[i]=max(j*(i-j), j*dp[i-j])。由于j的取值范围是1到i-1，需要遍历所有的j得到dp[i]的最大值，因此可以得到状态转移方程如下：
            # 1) 将 i 拆分成 j 和 i−j 的和，且 i−j 不再拆分成多个正整数，此时的乘积是 j * (i-j)
            # 2) 将 i 拆分成 j 和 i−j 的和，且 i−j 继续拆分成多个正整数，此时的乘积是 j * dp[i-j]
            for j in range(1, i - 1):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[n]
```

## 搜索二叉树数量
```python
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
# 动态规划
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

# 贪心
    def maxSubArray(self, nums):
        res = -100000
        tmp = 0
        for i in range(len(nums)):
            tmp += nums[i]  # 每次都记录子序和的最大值，如果子序和的值小于0了就重新令其等于0
            if tmp >= res:
                res = tmp
            if tmp < 0:
                tmp = 0
        return res
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
## 分割等和子集

https://leetcode.cn/problems/partition-equal-subset-sum/

```python
# 当成01背包问题，物品的重量为nums[i]，价值也为nums[i]，背包的容量是sum(nums)/2
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        a = sum(nums)
        if a % 2 == 1: return False
        target = a // 2
        dp = [0 for _ in range(10001)] # 根据题目条件进行初始化，要考虑到sum//2的最大可能
        for i in range(n):
            for j in range(target,nums[i]-1,-1):
                dp[j] = max(dp[j - nums[i]] + nums[i],dp[j])

        return dp[target] == target
```

## 最后一块石头的重量 II

https://leetcode.cn/problems/last-stone-weight-ii/

```python
# 两块石头两两相碰，相等时抵消，跟上一题原理相同
# dp[j]的含义：背包装满j容量时的最大价值（该题的价值就是石头重量）
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        dp = [0]*1501 # 初始化，最大的背包容量可能值
        target = sum(stones)//2
        n = len(stones)
        for i in range(n):
            for j in range(target,stones[i]-1,-1): # 倒序遍历，这样不会重复刷新dp[j]
                dp[j] = max(dp[j-stones[i]]+stones[i],dp[j])
        return sum(stones) - 2*dp[target]
```

## 找零钱1(用的最少的硬币数量)

https://leetcode.cn/problems/coin-change/

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[j] 总金额为j时所需要的最少硬币个数，所以dp要做限制，当只有2块的时候，遇到3块则会因为dp[3-2] = 4而 = amount + 1 ，返回-1
        # dp[j] = min(dp[j],dp[j-coins[i]] + 1)
        dp = [a for a in range(amount+1)] # 第一行初始化
        for i in range(len(coins)): # 遍历物品
            for j in range(coins[i],amount+1): # 遍历背包
                dp[j] = min(dp[j],dp[j-coins[i]] + 1)
        return dp[-1] if dp[-1] < amount + 1 else -1


    #     0   1   2   3   4   5
    # 1   0   1   2   3   4   5
    # 2   0   1   1   2   2   3       
    # 5   0   0   0   0   0   1                    
```

## 找零钱2（所有可能的硬币组合）

https://leetcode.cn/problems/coin-change-2

```python

class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # dp[j]代表凑成j的总金额的组合数
        # dp[j] += dp[j-coins[i]]
        dp = [0]*(amount+1)
        dp[0] = 1
        for i in range(len(coins)):
            # 多重背包 正序遍历
            for j in range(coins[i],amount+1):
                dp[j] += dp[j-coins[i]]  # dp[2][4] = dp[1][4] + dp[2][4 - 2]  在放入1、2硬币的情况下，总金额凑成4的组合数 = 在放入1的情况下总金额凑成4的组合数 + 在放入1、2的情况下总金额凑成2的组合数，因为2和4之间只差了一个2元，即放入这个硬币的组合数加上不放这个硬币的组合数
        return dp[-1]
    #     0   1   2   3   4   5
    # 1   1   1   1   1   1   1
    # 2           2   2   3   3
    # 5                       4
```

## 最长递增子序列

https://leetcode.cn/problems/longest-increasing-subsequence/

```python
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i,n):
                if nums[j] > nums[i]:
                    dp[j] =max(dp[i] + 1 ,dp[j]) # dp[j]跟自己和上一个小的数i的dp[i]做比较，有可能在其他序列中已经最大，所以加上max
        return max(dp)
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

