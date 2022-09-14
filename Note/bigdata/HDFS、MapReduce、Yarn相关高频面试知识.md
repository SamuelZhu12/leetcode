# HDFS、MapReduce、Yarn面试知识
## 1. HDFS

### 1.1 基本概念

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208231442251.png" alt="img" style="zoom:67%;" />

HDFS是Hadoop下分布式文件存储系统，其具有高吞吐、高容错的特性，可以部署在低成本的硬件上。

#### 架构

HDFS遵从主、从架构，由单个NameNode和多个DataNode组成。

- NameNode：负责有关`文件系统命名空间`的操作，例如打开、关闭、重命名文件和目录等。同时还负责集群元数据的存储，记录着文件中各个数据块的位置信息。
- DataNode：负责提供来自文件系统客户端的读写请求，执行块的创建，删除等操作。
- SecondaryNameNode：NameNode的热备份。

#### 数据复制

HDFS提供了数据复制机制，将每一个文件存储为一系列Block，每个块由多个副本（默认为3）来保证容错，块的默认大小为128M。

#### 机架感知副本放置策略

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208231449495.png" alt="img" style="zoom:67%;" />

在写入程序位于DataNode上时，就优先将写入文件的一个副本放置在该DataNode上，否则放在随机DataNode。之后在另一个远程机架上的任意一个节点上放置另一个副本，并在该机架上的另一个节点上放置最后一个副本。此策略可以减少机架间的写入流量，从而提高写入性能。

为了最大限度地减少带宽消耗和读取延迟，HDFS 在执行读取请求时，优先读取距离**读取器最近**和**访问量最小**的副本。如果在与读取器节点相同的机架上存在副本，则优先选择该副本。如果 HDFS 群集跨越多个数据中心，则优先选择本地数据中心上的副本。 

由于Block在不同的Rack上都有备份，所以不再是单数据访问，所以速度和效率是非常快的。另外HDFS可以并行从服务器集群中读写，增加了文件读写的访问带宽。

#### HDFS架构的稳定性

1. 心跳机制和重新复制

每个 DataNode 定期向 NameNode 发送心跳消息，如果超过指定时间没有收到心跳消息，则将 DataNode 标记为死亡。NameNode 不会将任何新的 IO 请求转发给标记为死亡的 DataNode，也不会再使用这些 DataNode 上的数据。 由于数据不再可用，可能会导致某些块的复制因子小于其指定值，NameNode 会跟踪这些块，并在必要的时候进行重新复制。

2. 数据的完整性

由于存储设备故障等原因，存储在 DataNode 上的数据块也会发生损坏。为了避免读取到已经损坏的数据而导致错误，HDFS 提供了数据完整性校验机制来保证数据的完整性，具体操作如下：

当客户端创建 HDFS 文件时，它会计算文件的每个块的 `校验和`，并将 `校验和` 存储在同一 HDFS 命名空间下的单独的隐藏文件中。当客户端检索文件内容时，它会验证从每个 DataNode 接收的数据是否与存储在关联校验和文件中的 `校验和` 匹配。如果匹配失败，则证明数据已经损坏，此时客户端会选择从其他 DataNode 获取该块的其他可用副本。

3. 元数据的磁盘故障

`FsImage` 和 `EditLog` 是 HDFS 的核心数据，这些数据的意外丢失可能会导致整个 HDFS 服务不可用。为了避免这个问题，可以配置 NameNode 使其支持 `FsImage` 和 `EditLog` 多副本同步，这样 `FsImage` 或 `EditLog` 的任何改变都会引起每个副本 `FsImage` 和 `EditLog` 的同步更新。

4. 支持快照

快照支持在特定时刻存储数据副本，在数据意外损坏时，可以通过回滚操作恢复到健康的数据状态。

### 1.2 读流程
![HDFS读流程](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291733035.png)

1. 客户端(HDFS Client)通过Distributed FileSystem分布式文件系统向NameNode请求**下载**文件，由NameNode返回目标文件的元数据；
2. 由FSDataInputStream请求读数据DataNode1上的block1，由DataNode1传输回数据；
3. 再向block2发送读数据请求...再向block3发送请求...

### 1.3 写流程
![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291733048.png)

1. 客户端(HDFS Client)通过Distributed FileSystem分布式文件系统向NameNode请求**上传**文件，由NameNode响应上传文件；
2. 请求上传第一个Block，请返回DataNode；
3. 返回data1, data2 ,data3表示用这三个DataNode存储数据；
4. FSDataOutputStream请求建立Block传输通道，将三个DataNode进行串联；
5. 由末端DataNode向回传输应答成功信息；
6. FsDataOutputStream向前传输数据Packet。



### 1.4 Secondary NameNode

NameNode主要是用来保存HDFS的元数据信息，比如命名空间信息，块信息等。当它运行的时候，这些信息是存在内存中的。但是这些信息也可以持久化到磁盘上。

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208231514871.png" alt="img"  />

上面的这张图片展示了NameNode怎么把元数据保存到磁盘上的。这里有两个不同的文件：

```
1. fsimage - 它是在NameNode启动时对整个文件系统的快照
2. edit logs - 它是在NameNode启动后，对文件系统的改动序列
```

只有在NameNode重启时，edit logs才会合并到fsimage文件中，从而得到一个文件系统的最新快照。但是在产品集群中NameNode是很少重启的，这也意味着当NameNode运行了很长时间后，edit logs文件会变得很大。在这种情况下就会出现下面一些问题：

1. edit logs文件会变的很大，怎么去管理这个文件是一个挑战。
2. NameNode的重启会花费很长时间，因为edit logs很多改动要合并到fsimage文件上。
3. 如果NameNode挂掉了，那我们就丢失了很多改动，因为此时的fsimage文件非常旧。[笔者注: 笔者认为在这个情况下丢失的改动不会很多, 因为丢失的改动应该是还在内存中但是没有写到edit logs的这部分。

因此为了克服这个问题，我们需要一个易于管理的机制来帮助我们减小edit logs文件的大小和得到一个最新的fsimage文件，这样也会减小在NameNode上的压力。这跟Windows的恢复点是非常像的，Windows的恢复点机制允许我们对OS进行快照，这样当系统发生问题时，我们能够回滚到最新的一次恢复点上。

**SecondaryNameNode**就是来帮助解决上述问题的，它的职责是合并NameNode的edit logs到fsimage文件中。

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208231519489.png" alt="img"  />

```
1. 首先，它定时到NameNode去获取edit logs，并更新到fsimage上。[笔者注：Secondary NameNode自己的fsimage]
2. 一旦它有了新的fsimage文件，它将其拷贝回NameNode中。
3. NameNode在下次重启时会使用这个新的fsimage文件，从而减少重启的时间。
```

### 1.5 HDFS的优点

- 高吞吐量访问：由于Block在不同的Rack上都有备份，所以不再是单数据访问，所以速度和效率是非常快的。另外HDFS可以并行从服务器集群中读写，增加了文件读写的访问带宽。
- 高容错性：系统故障是不可避免的，如何做到故障之后的数据恢复和容错处理是至关重要的。HDFS通过多方面保证数据的可靠性，多份复制并且分布到物理位置的不同服务器上，数据校验功能、后台的连续自检数据一致性功能都为高容错提供了可能。
- 线性扩展：因为HDFS的Block信息存放到NameNode上，文件的Block分布到DataNode上，当扩充的时候仅仅添加DataNode数量，系统可以在不停止服务的情况下做扩充，不需要人工干预。

### 1.6 HDFS故障检测机制

1）节点失败检测机制

- 如果NameNode故障，整个集群会宕机，因为NameNode是唯一单点。
- DataNode以固定周期向NameNode发送心跳信息，通过该方式证明其健康。超过10min没有收到DN的心跳信息，则NN会认为该DN挂掉。

2）通信故障检测机制

- 只要DN发送了数据，接收方就会返回确认码，如果经过几次重试依旧没有收到确认码，则可认为主机挂掉。

3）数据错误检测机制

- 在传输数据的时候，同时会发送总和校验码，当硬盘存储数据的时候也会存储总校验码。
- DN发送数据块报告之前，会检查总和校验码，如果数据存在错误则不发送该数据块。NN也能根据发送数据的校验码判断DN的健康状态。

　　HDFS存储理念是以最少的钱买最烂的机器并实现最安全、难度高的分布式文件系统(高容错性低成本)。从上可以看出，HDFS认为机器故障是种常态，所以在设计时充分考虑到单个机器故障，单个磁盘故障，单个文件丢失等情况。



## 2. HDFS小文件处理

**问题：**一个Block占用**NameNode** 150字节，那么当小文件过多的时候则会占用大量内存空间

**解决方法** 1. 采用har归档方式，将小文件归类； 2. 采用CombineTextInputFormat，将小文件逻辑划分到一个切片中，使多个小文件被一个MapTask处理； 3. 在有小文件的情况下开启JVM重用，使JVM实例在同一个job中被重新使用N次，N的值可以在Hadoop的mapred-site.xml中进行配置。

*CombineTextInputFormat切片机制分为虚拟存储过程和切片过程(注意区分切片和block划分的概念)：

1. 虚拟存储过程：将输入目录下所有文件跟已设置的setMaxInputSplitSize值进行比较，如果不大于该值则逻辑上划分一个块，如果**大于该值且大于两倍**，则以最大值切割一块，当剩余数据**大于该值且小于该值2倍**，则将剩余数据平均划分；
2. 切片过程：判断虚拟存储的文件大小是否大于setMaxInputSplitSize，若**大于等于**则单独形成切片；如果**不大于**则跟下一个虚拟存储文件进行合并形成切片。

## 3. MapRuduce工作流程

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291733781.png)

### 3.1 Map阶段
1. **Split阶段**：客户端提交前对文件进行split操作，根据参数配置分片成若干个切片。输入分片(InputSplit)通常和HDFS的block关系很密切，假如设定的HDFS块大小为128MB，运行的大文件时1280MB，MapReduce会分为10个MapTask，每个MapTask都运行在block所在的Datanode上，体现移动计算不移动数据的思想；
2. 提交信息后由Yarn的Resource Manager对文件进行MapTask数量计算；
3. **Read阶段**：MapTask 通过用户编写的 RecordReader，按照 InputSplit记录的位置信息读取数据，解析出一个个<Key,Value>。
4. **Map阶段**：将解析出的<Key,Value>交给用户编写的Mapper类中map()函数处理，并产生一系列新的 <key,value>。
例如wordcount实例中，文本文件中每一行是一个单词，则Read阶段则产生<index,word>类型的键值对，而通过Map阶段后，则转换为<word,1>类型的键值对。
5. **Collector阶段**：在用户编写 map()函数中，当数据处理完成后，一般会调用 collect()输出结果。在该函数内部，它会将生成的<key,value>分区（调用 Partitioner），并写入一个环形内存缓冲区中，缓冲区默认100M，其中一部分存储元数据(metadata)，一部分存储数据；
6. **Shuffle阶段**（包括6-9步骤）：根据元数据中对数据标注的分区，在分区內部根据Key进行**快排**，保证每个分区内部数据有序；
7. **Combiner阶段**（可选）：Combiner会进行一些预聚合，将分区内数据进行合并(<a,1>,<a,1> -> <a,2>)；
8. 环形缓冲区写到80%时反向向磁盘进行溢写，在溢出过程的过程中将若干小文件合并成一个大文件，将产生的多个溢写小文件按照分区进行**归并排序**（因为数据已经有顺序）
9. 启动相应数量的ReduceTask，并告知数据处理范围（数据分区）。 **（可提前启动）**



### 3.2 Reduce阶段
![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291733697.png)

1. ReduceTask主动pull MapTask中数据，且拉取的是指定分区(ReduceTask1拉取不同MapTask中的partition0)中的数据；
2. 合并文件，**归并排序**；
3. ReduceTask使用outputFormat（默认为TextOutputFormat）向最终文件Part-r-000000进行write。**（一个ReduceTask对应一个part-r-0000x）**

**总结：一次快排，两次归并**

### 3.3 Map后shuffle机制
![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291733068.png)

![shuffle](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291757464.jpeg)

**Shuffle是map之后、reduce之前的一个混洗流程（细节与上相同)**
shuffle是按照key将数据发送到不同的reduce,产生磁盘与网络IO,如果key分布不均匀,会产生数据倾斜。

1. 将数据存入环形缓冲区；
2. 存储到80%进行反向溢写；
3. 反向溢写线程启动时，对缓冲区内部数据根据分区依据key进行快速排序，对溢写结果进行Combiner（可选）；
4. 小文件落盘后对若干小文件进行归并排序，按照分区合并成一个大文件；
5. 对分区内数据进行压缩，写入磁盘等待ReduceTask进行pull；
6. ReduceTask pull数据后先尝试放入内存，如果内存不够则放入磁盘。

### 3.4 Shuffle 优化

1） Map阶段

1. 增大环形缓冲区大小；
2. 增大环形缓冲区溢写比例，如80%->90%；
3. 减少对溢写文件的merge次数；
4. 采用Combiner提前进行合并。

2） Reduce阶段

1. 合理设置Map和Reduce数量：太少会导致Task等待时间过长，太多会导致Map、Reduce形成资源竞争；
2. 设置Map和Reduce共存：调整slowstart.completedmaps参数，使map运行一段时间后，reduce也开始运行；
3. 集群性能允许前提下，增大reduce端存储数据内存大小；

3） IO传输阶段

主要采用数据压缩方式：

1. map输入端考虑数据量大小和切片，使用如Bzip2、LZO等支持切片的压缩方式；
2. map输出端考虑速度快的snappy、LZO；
3. reduce输出端主要考虑整体需求，例如要作为下一个mr输入则需要考虑切片，如果要永久保存则考虑压缩率较大的gzip。

4） 整体

1. 增加NodeManager内存，根据服务器实际配置灵活调整；
2. 增加单任务内存，根据服务器实际配置灵活调整；
3. 控制MapTask内存上限，调整**mapreduce.map.memory.mb**参数：如果数据量过大（>128M)，则考虑增加内存上限；
4. 控制ReduceTask内存上限，调整**mapreduce.reduce.memory.mb**参数：如果数据量过大(>128M) ，则考虑增加内存上限；
5. 控制MapTask和ReduceTask的堆内存大小。mapreduce.map.java.opts、mapreduce.reduce.java.opts；
6. 增加MapTask和ReduceTask的CPU核数；
7. 增加Container的CPU核数和内存大小；
8. 在hdfs-site.xml文件中配置多目录；

## 4. Yarn资源调度工作机制

### 4.1 Yarn框架

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208081411082.png)

- ResourceManager

`ResourceManager` 通常在独立的机器上以后台进程的形式运行，它是整个集群资源的主要协调者和管理者。`ResourceManager` 负责给用户提交的所有应用程序分配资源，它根据应用程序优先级、队列容量、ACLs、数据位置等信息，做出决策，然后以共享的、安全的、多租户的方式制定分配策略，调度集群资源。

- NodeManager



<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051653127.png" alt="NodeManager1.png" style="zoom:50%;" />

`NodeManager` 是 YARN 集群中的每个具体节点的管理者。主要负责该节点内所有容器的生命周期的管理，监视资源和跟踪节点健康。具体如下：

​		1. 启动时向 `ResourceManager` 注册并定时发送心跳消息，等待 `ResourceManager` 的指令；

​		2. 维护 `Container` 的生命周期，监控 `Container` 的资源使用情况；

​		3. 管理任务运行时的相关依赖，根据 `ApplicationMaster` 的需要，在启动 `Container` 之前将需要的程序及其依赖拷贝到本地。

​		4. NodeManager通过两个RPC协议与RM和各个ApplicationMaster进行通信：

- ApplicaitonMaster

在用户提交一个应用程序时，YARN 会启动一个轻量级的进程 `ApplicationMaster`。`ApplicationMaster` 负责协调来自 `ResourceManager` 的资源，并通过 `NodeManager` 监视容器内资源的使用情况，同时还负责任务的监控与容错。具体如下：

​		1. 根据应用的运行状态来决定动态计算资源需求；

​		2. 向 `ResourceManager` 申请资源，监控申请的资源的使用情况；

​		3. 跟踪任务状态和进度，报告资源的使用情况和应用的进度信息；

​		4. 负责任务的容错。

- Container

​		`Container` 是 YARN 中的资源抽象，它封装了某个节点上的多维度资源，如内存、CPU、磁盘、网络等。当 AM 向 RM 申请资源时，RM 为 AM 返回的资源是用 `Container` 表示的。YARN 会为每个任务分配一个 `Container`，该任务只能使用该 `Container` 中描述的资源。`ApplicationMaster` 可在`Container`内运行任何类型的任务。例如，`MapReduce ApplicationMaster` 请求一个容器来启动 map 或 reduce 任务，而 `Giraph ApplicationMaster` 请求一个容器来运行 Giraph 任务。

### 4.2 Yarn工作机制

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208081423625.png)







![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291733560.png)


1. Mr程序提交到客户端所在节点后会创建一个YarnRunner（本地模式时是LocalRunner）；
2. 由Yarn向ResourceManager申请一个Application；
3. RM返回Application资源提交路径和application_id；
4. Yarn向集群提交job运行所需资源：**切片文件job.split，参数文件job.xml，代码xx.jar**;
5. 客户端向RM申请mapreduceApplicationMaster(AM)，RM将请求初始化成一个Task，并放入任务队列(Scheduler)中；
6. NodeManager从RM中领取任务；
7. NM创建一个Container，创建一个MRAppmaster，并从application集群路径中读取切片信息job.split；
8. Application Master根据切片信息向RM申请相应数量的MapTask；
9. 其他NM也领取相应任务，但不建立mrAppMaster，由第一个NM中的mrAppMaster发起启动其他NM中MapTask和Yarn Child的指令；
10. 不同的NM的Container中执行各自的MapTask，将各自的数据按照分区拉到磁盘；（**此时Map阶段已经结束**）
11. NM向RM申请Container运行ReduceTask，对应进程也是Yarn Child；
12. mpAppMaster向RM申请注销。

### 4.3 YARN优势
1、YARN的设计减小了JobTracker的资源消耗，并且让监测每一个Job子任务（tasks)状态的程序分布式化了，更安全、更优美。

2、在新的Yarn中，ApplicationMaster是一个可变更的部分，用户可以对不同的编程模型写自己的AppMst，让更多类型的编程模型能够跑在Hadoop集群中。

3、对于资源的表示以内存为单位，比之前以剩余slot数目更加合理。

4、MRv1中JobTracker一个很大的负担就是监控job下的tasks的运行状况，现在这个部分就扔给ApplicationMaster做了，
而ResourceManager中有一个模块叫做ApplicationManager，它是监测ApplicationMaster的运行状况，如果出问题，会在其他机器上重启。

5、Container用来作为YARN的一个资源隔离组件，可以用来对资源进行调度和控制。

## 5. Yarn调度器

调度器是在Yarn工作流程中管理**调度任务队列**的工具

1）Hadoop调度器主要分为三类：

	FIFO、Capacity Scheduler（容量调度器）和Fair Sceduler（公平调度器）；
	
	Apache默认的资源调度器是容量调度器；
	
	CDH默认的资源调度器是公平调度器。

2） 调度器区别：

	FIFO调度器：支持单队列、先进先出，但生产环境不会使用。
	
	容量调度器：支持多队列，保证先进入的任务优先执行。
	
	公平调度器：支持多队列，保证每个任务公平享有队列资源。

3） 调度器的选择：

	大厂：对并发度要求较高，选择公平调度器，因为其对服务器性能有一定要求。
	
	中小厂：选择容量调度器，因为集群服务器资源不太充裕。

4） 在生产环境怎么创建队列？

		调度器默认时只有一个default队列，不能满足生产需求。
	1. 按照框架：hive/spark/flink每个框架的任务放入指定的队列。（企业用的不多）
	2. 按照业务模块：如登录注册、购物车、下单等业务单独创建队列。

5） 多队列的好处？
 	1. 避免员工不小心写出死循环代码，将资源全部占用；
 	2. 实现任务的降级使用，特殊时期保证重要的任务队列资源充足。



## 6. Hadoop解决数据倾斜的方法

**数据倾斜：Hadoop中为了方便分布式计算对数据进行切分，但出现因数据本身而导致切分不均匀的情况，有些数据多，有些数据少，从而导致后续计算中不同reducer的负载不同，影响性能。具体来说，当map阶段结束时，由于环形缓冲区会对数据的key进行partition操作，所以同一个分区中数据的key多半相同，而reducer会拉取不同maptask中指定分区的数据，这意味着相同的key会进入同一个reducer进行聚合，如果key之间数量差距过大，则会造成不同的reducer之间的负载不均衡**

1) 提前在map进行combine，减少数据的传输量

	在Mapper端加上combiner相当于提前进行了reduce，即把一个Mapper中相同的key进行聚合，减少了shuffle过程中传输的数据量以及reducer端的计算量。但如果数据倾斜的key大量分布在不同的mapper时效果不好。

2. 导致数据倾斜的key大量分布在不同的mapper

	①局部聚合加全局聚合

	第一次在map阶段对那些导致了数据倾斜的key加上1到n的随机前缀（比如<a,1>变成<1-a,1>），这样本来相同的key也会分到多个reducer中进行局部聚合，降低了数据倾斜数量；

	第二次mapreduce去掉key的随机前缀，进行全局聚合。

**该方法进行了两次mapreduce，性能稍差**

3. 增加reducer，提升并行度

	JobConf.setNumReduceTasks(int)

4. 实现自定义分区

	根据数据分布情况，自定义散列函数，将key均匀分配到不同的Reducer。

5. 自定义Shuffle分区算法

	根据数据应用属性将数据重新进行分区划分。

## 7. Hadoop高可用

### HA(High Avaliablity)高可用

HA（High Availablity），即高可用（7*24小时不中断服务）。

实现高可用最关键的策略是消除单点故障。HA严格来说应该分成各个组件的HA机制：HDFS的HA和YARN的HA。

Hadoop2.0之前，在HDFS集群中NameNode存在单点故障（SPOF）。

NameNode主要在以下两个方面影响HDFS集群：

- NameNode机器发生意外，如宕机，集群将无法使用，直到管理员重启

- NameNode机器需要升级，包括软件、硬件升级，此时集群也将无法使用

HDFS HA功能通过配置Active/Standby两个NameNodes实现在集群中对NameNode的热备来解决上述问题。如果出现故障，如机器崩溃或机器需要升级维护，这时可通过此种方式将NameNode很快的切换到另外一台机器。

HDFS-HA工作机制：通过多个NameNode消除单点故障

### HDFS-HA工作要点

**1）元数据管理方式需要改变**

内存中各自保存一份元数据；

**Edits**日志只有Active状态的NameNode节点可以做写操作；

**所有的NameNode都可以读取Edits**；

共享的Edits放在一个共享存储中管理（qjournal和NFS两个主流实现）；

**2）需要一个状态管理功能模块**

实现了一个zkfailover，常驻在每一个namenode所在的节点，每一个zkfailover负责监控自己所在NameNode节点，利用zk进行状态标识，当需要进行状态切换时，由zkfailover来负责切换，切换时需要防止brain split现象的发生。

**3）必须保证两个NameNode之间能够ssh无密码登录**

**4）隔离（Fence），即同一时刻仅仅有一个NameNode对外提供服务**

### HDFS-HA自动故障转移工作机制

自动故障转移为HDFS部署增加了两个新组件：ZooKeeper和ZKFailoverController（ZKFC）进程， **ZooKeeper是维护少量协调数据，通知客户端这些数据的改变和监视客户端故障的高可用服务**。HA的自动故障转移依赖于ZooKeeper的以下功能：

**1．故障检测**

集群中的每个NameNode在ZooKeeper中维护了一个会话，如果机器崩溃，ZooKeeper中的会话将终止，ZooKeeper通知另一个NameNode需要触发故障转移。

**2．现役NameNode选择**

ZooKeeper提供了一个简单的机制用于唯一的选择一个节点为active状态。如果目前现役NameNode崩溃，另一个节点可能从ZooKeeper获得特殊的排外锁以表明它应该成为现役NameNode。

ZKFC是自动故障转移中的另一个新组件，是ZooKeeper的客户端，也监视和管理NameNode的状态。每个运行NameNode的主机也运行了一个ZKFC进程，ZKFC负责：

**1）健康监测**

ZKFC使用一个健康检查命令定期地ping与之在相同主机的NameNode，只要该NameNode及时地回复健康状态，ZKFC认为该节点是健康的。如果该节点崩溃，冻结或进入不健康状态，健康监测器标识该节点为非健康的。

**2）ZooKeeper会话管理**

当本地NameNode是健康的，ZKFC保持一个在ZooKeeper中打开的会话。如果本地NameNode处于active状态，ZKFC也保持一个特殊的znode锁，该锁使用了ZooKeeper对短暂节点的支持，如果会话终止，锁节点将自动删除。

**3）基于ZooKeeper的选择**

如果本地NameNode是健康的，且ZKFC发现没有其它的节点当前持有znode锁，它将为自己获取该锁。如果成功，则它已经赢得了选择，并负责运行故障转移进程以使它的本地NameNode为Active。

**HDFS-HA故障转移机制**

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208261038986.png)

**参考：尚硅谷Hadoop教程**