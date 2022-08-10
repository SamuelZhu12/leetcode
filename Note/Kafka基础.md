# Kafka基础知识总结
[TOC]

**Kafka是一个分布式的基于发布/订阅模式的消息队列(Message Queue)，主要应用于大数据实时处理领域**

## 1 消息队列

### 1.1 消息队列应用场景

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101518791.png)


**使用消息队列的好处：**

1. 解耦

	允许独立扩展或修改两边的处理过程，只要确保它们遵守同样的接口约束；

2. 可恢复性

	系统的一部分组件失效时，不会影响到整个系统。消息队列降低了进程间的耦合度，所以即使一个处理消息的进程挂掉，加入队列中的消息仍然可以在系统恢复后被处理；

3. 缓冲

	有助于控制和优化数据流过系统的速度，解决生产消息和消费消息的处理速度不一致的情况；

4. 灵活性和峰值处理能力

	在访问量剧增的情况下，应用仍然需要继续发挥作用，但是这样的突发流量并不常见。如果为以能处理这类峰值访问为标准来投入资源随时待命无疑是巨大的浪费。使用消息队列能够使关键组件顶住突发的访问压力，而不会因为突发的超负荷的请求而完全崩溃；

5. 异步通信

	用户有时不需要立即处理消息。消息队列提供了异步处理机制，允许用户把多个消息放入队列，但并不立即处理它。

### 1.2 消息队列的两种模式

1）点对点模式

	生产者生产消息发送到Queue中，然后消费者从Queue中拉取并且消费信息。信息被消费后从queue中删除，其他消费者就无法消费。一个Queue支持存在多个消费者，但对于一个消息而言只有一个消费者可消费。

2） 发布/订阅模式

	生产者将信息发布到topic中，同时有多个消费者消费该消息。和点对点模式不同，发布到topic的消息可被所有消费者消费。

## 2 Kafka基础架构

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101518920.png)


1） **Producer**：消息生产者，向kafka broker发消息的客户端。

2） **Consumer**：消息消费者，向kafka broker取消息的客户端。

3） **Consumer Group**：消费者组，由多个comsumer组成。**消费者组内每个消费者负责消费不同分区的数据，一个分区只能由一个消费者进行消费；消费者组之间互不影响**。所有的消费者组都属于某个消费者组，**即消费者组使逻辑上的一个订阅者**。

4） **Broker**：一台kafka服务器就是一个broker。一个集群由多个broker组成。一个broker可以容纳多个topic。

5） **Topic**： 可以理解为一个队列，消费者和生产者面向的都是一个topic。

6） **Partition**：为了实现扩展性，一个非常大的topic可以分布到多个broker上。

7） **Replica**： 副本，为保证集群中某个节点发生故障时，该节点的partition数据不丢失，且kafka仍然能正常工作。kafka提供了副本机制，一个topic的每个分区都有若干个副本，一个leader和若干followers。

8） **leader**： 每个分区多个副本的“主”，生产者发送数据的对象，以及消费者消费数据的对象都是leader。

9） **follower**：每个分区多个副本的“从”，实时从leader中同步数据，保持和leader数据的同步。leader发生故障时，某个follower会成为新的follower。



### 2.1 Kafka工作流程及文件存储机制

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101518135.png)


- Kafka中消息是以**topic**进行分类的，生产者生产消息，消费者消费消息，都是面向topic。

- topic是逻辑上的概念，而partition分片则是物理上的概念。每个partition对应一个log文件，该log文件中存储的就是producer生产的数据。Producer生产的数据会不断追加到该log文件末端，且每条数据都有自己的offset。消费者组中的每个消费者，都会实时记录自己消费到了哪个offset，以便出错回复时，从上次的位置继续消费。

- Kafka采取**分片+索引**的机制。一个topic被分为多个partition，而一个partition又被分为多个segment，一个segment对应着.log文件和.index文件。

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101518484.png)


### 2.2 Kafka生产者

#### 2.2.1 分区策略

1）分区原因：

①方便在集群中进行扩展。每个partition可通过调整而适应所在的机器，一个topic又有多个partition，因而集群可以适应任意大小的数据；

②可以提高并发，因为可以以partition作为单位读写。

2）分区原则：

①指明partition的情况下，直接把指明的值作为partition值；

②没有指明partition但有key的情况下，将key的hash值与topic的partition值取余数得到partition值；

③既没有partition也没有key的情况下，第一次调用时随机生成一个整数（后面以此整数为基础自增），将这个值与topic可用的partition总数取余得到partition值，即round-roubin算法。

#### 2.2.2 数据可靠性保证

	为保证数据的完整性、可靠性，topic的每个partition收到producer发送的数据后，都需要向producer发送ack，如果producer收到了ack则进行下一轮数据的发送，否则重新发送数据。

 1）副本数据同步策略

①半数follower以上完成同步，就发送ack。这种方式延迟低，但是leader挂掉时，选举新的leader需要容忍n台节点的故障，需要增加2n+1台副本。

②**全部follower完成同步**，发送ack。这种方式当leader挂掉时，选举新的leader容忍n台节点故障，但只需要增加n+1台副本，缺点是延迟高。

Kafka选择第二种策略，因为副本多容易造成数据冗余，且kafka对网络延迟要求较低。

2）ISR

ISR(in-sync replica set)， 指与leader保持同步的follower集合。当ISR中的follower完成数据的同步之后，leader会给follower发送ack，如果follower长时间没有向leader同步数据，则该follower会被提出，放入osr中。

 3）ack应答机制

针对不同的情况，ack有三种不同机制。 

① ack = 0 时，producer不等待broker的ack，这种操作提供了最低延迟，broker一接收数据还未写入磁盘就返回，当broker故障时，可能会造成**数据丢失**。

② ack = 1 时，producer等待broker的ack，partition的leader落盘成功后返回ack，如果在follower同步之前leader发生故障，将会**数据丢失**。

③ ack = -1 时，producer等待broker的ack，partition的leader和follower全部落盘成功后才返回ack。但是如果在follower同步完成后，broker发送ack前，leader发生故障，那么会造成**数据重复**。

4）故障处理细节

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101518699.png)


**LEO：指每个副本最大的offset**；

**HW：指消费者可见的最大的offset，ISR队列中最小的LEO。**

① follower故障

	follower故障时，会被临时提出ISR，等待该follower恢复后follower会读取本地磁盘记录的上次的HW，并将log文件高于HW的部分截取掉，从HW开始向leader进行同步。等待该follower的LEO大于等于该partition的HW，即follower追上leader后，就可以重新加入ISR。

② leader故障

	leader发生故障时，会从ISR中选出一个新的leader，之后为保证多个副本之间的数据一致性，其余的follower会将各自的log文件高于HW的部分截取掉，然后从新的leader同步数据。

**这种处理机制只能保证数据的一致性，并不能保证数据不丢失，数据的不丢失由ack确保。**

#### 2.2.3 Exactly Once语义

	当 ack = -1时，可保证producer到server之间不丢失数据，只可能造成**数据重复**，即**At Least Once**语义。相对的，$$ack = 0$$时，可以保证生产者每条信息只会被发送一次，即**At Most Once**语义，这样可以避免数据重复，但不能避免数据丢失。因此，kafka中加入了**幂等性**，以此保证数据既不丢失也不重复，即**Exactly Once**语义。
$$
At Least Once + 幂等性 = Exactly Once
$$
Kafka的幂等性实现方式是将原本下游要做的去重工作放在了数据上游。开启了幂等性的producer在初始化时会被分配一个pid，发往同一个partition的消息会附带Sequence Number。而Broker端会对<pid,topic,seqNumber>做缓存，当具有相同主键的消息提交时，Broker只会持久化一条。但重启时pid会变化，不同partition也具有不同的主键，因此幂等性无法保证跨分区跨会话的exactly once。



### 2.3 Kafka消费者



#### 2.3.1 消费方式 

	**消费者使用pull模式从broker中拉取数据**。push模式很难适应消费速率不同的消费者，因为消息发送速率是由broker决定的。它的目标是尽可能快的传递消息，很可能造成consumer来不及消费信息，典型的表现就是拒绝服务以及网络阻塞。而pull模式则可以根据consumer的消费能力以适应当前的速率来消费信息。
	
	pull模式的缺点是：如果kafka没有数据，则consumer可能会陷入循环中，一直返回空数据。针对这一点，kafka的消费者在消费数据时会传入一个时长参数timeout，如果当前没有数据可供消费，consumer会等待一段时间后再返回，这段时长即为timeout。

#### 2.3.2 分区分配策略

	确定哪个parition由哪个consumer来消费，有两种分配策略：Round-Robin和Range策略。
	
	①Round-Robin
	
	Round-Robin策略就是简单的将所有的partition和consumer按照字典序进行排序之后，然后依次将partition分配给各个consumer，如果当前的consumer没有订阅当前的partition，那么就会轮询下一个consumer，直至最终将所有的分区都分配完毕。但是轮询的方式会导致每个consumer所承载的分区数量不一致，从而导致各个consumer压力不均一。
	
	②**Range**
	
	Range策略是按照topic依次进行分配，先计算各个consumer将会承载的分区数量，然后将已订阅的topic的partition按照指定数量分配给该consumer。如果有两个Consumer：c1,c2，两个topic：topic1, topic2 ，每个topic有三个partition：topic1_0, topic_1, topic1_2,topic2_0, topic2_1, topic2_2 ，则会按照以下步骤进行分配：

- 对于topic1，获取topic1所有分区：topic1_0, topic_1, topic1_2 和订阅该分区的消费者c1,c2，将其按照字典进行排序；
- 按照平均分配的方式计算每个消费者会消费多少个分区，如果没有除尽，则将剩下的部分依次分配给前面的消费者。如上述例子三个主题，两个消费者，每个消费者分一个partition，多出来一个则分配到c1。
- topic2中的partition如上进行分配。最后的结果如下：

| $$C0$$ | topic1_0,topic1_1,topic2_0,topic2_1 |
| ------ | :---------------------------------: |
| $C1$   |          topic1_2,topic2_2          |

	从上述图结果可以看出，Range策略会导致排序在前面的消费者消费更多partition，从而导致各个consumer的压力不均衡。

#### 2.3.3 消费者offset的存储

由于 consumer 在消费过程中可能会出现断电宕机等故障， consumer 恢复后，需要从故障前的位置的继续消费，所以 **consumer 需要实时记录自己消费到了哪个 offset**，以便故障恢复后继续消费。Kafka 0.9 版本之前， consumer 默认将 offset 保存在 Zookeeper 中，从 0.9 版本开始，consumer 默认将 offset 保存在 Kafka 一个内置的 topic 中，该 topic 为__consumer_offsets。

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101518906.png)


### 2.4 Kafka高效读写数据

1）顺序写磁盘

	Kafka的producer生产数据，要写入到log文件中，写的过程是追加到文件末端，是顺序写。

2）零复制技术

### 2.5 Zookeeper在Kafka中的之作用

	1）Controller选举
	
	Kafka集群中会有一个Broker会被选举为Controller，对zookeeper进行监听，负责管理**broker的上下线**，所有**topic的分区副本分配**和**leader选举工作**。	
	
	2）配置管理
	
	Topic的配置之所以可以动态更新就是基于zookeeper做了一个动态全局配置管理。
	
	3）负载均衡
	
	基于zookeeper的消费者，实现了该特性，动态感知分区变动，将负载使用到既定分配策略分不到的消费者上。
	
	4）分布式通知
	
	如分区增加、Broker下线、topic变动等；
	
	5）集群管理和master选举
	
	可以通过命令行，对kafka集群上的topic partition分布，进行迁移管理，也可以对partition leader选举进行干预。

## 3 相关面试知识

1. Kafka 都有哪些特点？

- 高吞吐量、低延迟：kafka每秒可以处理几十万条消息，它的延迟最低只有几毫秒，每个topic可以分多个partition, consumer group 对partition进行consume操作；
- 可扩展性：kafka集群支持热扩展；
- 持久性、可靠性：消息被持久化到本地磁盘，并且支持数据备份防止数据丢失；
- 容错性：允许集群中节点失败（若副本数量为n,则允许n-1个节点失败）；
- 高并发：支持数千个客户端同时读写。

2. 请简述下你在哪些场景下会选择 Kafka？

- 日志收集：一个公司可以用Kafka可以收集各种服务的log，通过kafka以统一接口服务的方式开放给各种consumer，例如hadoop、HBase、Solr等。
- 消息系统：解耦和生产者和消费者、缓存消息等。
- 用户活动跟踪：Kafka经常被用来记录web用户或者app用户的各种活动，如浏览网页、搜索、点击等活动，这些活动信息被各个服务器发布到kafka的topic中，然后订阅者通过订阅这些topic来做实时的监控分析，或者装载到hadoop、数据仓库中做离线分析和挖掘。
- 运营指标：Kafka也经常用来记录运营监控数据。包括收集各种分布式应用的数据，生产各种操作的集中反馈，比如报警和报告。
- 流式处理：比如spark streaming和 Flink

3. Kafka 分区的目的？

	分区对于 Kafka 集群的好处是：实现负载均衡。分区对于消费者来说，可以提高并发度，提高效率。

4.  Kafka 是如何做到消息的有序性？

	kafka 中的每个 partition 中的消息在写入时都是有序的，而且单独一个 partition 只能由一个消费者去消费，可以在里面保证消息的顺	序性。但是分区之间的消息是不保证有序的。

5. Kafka 的高可靠性是怎么实现的？

	回答上文中关于ack和hw、leo的内容，分别保证了数据的可靠性和一致性。

6. ISR、OSR、AR 是什么？

	ISR：In-Sync Replicas 副本同步队列

	OSR：Out-of-Sync Replicas

	AR：Assigned Replicas 所有副本

	ISR是由leader维护，follower从leader同步数据有一些延迟，超过相应的阈值会把 	follower 剔除出 ISR, 存入OSR（Out-of-Sync Replicas ）列表，新加入的follower也会先存放在OSR中。AR=ISR+OSR。

7. LEO、HW、LSO、LW等分别代表什么？

- LEO：是 LogEndOffset 的简称，代表当前日志文件中下一条
- HW：水位或水印（watermark）一词，也可称为高水位(high watermark)，通常被用在流式处理领域（比如Apache Flink、Apache Spark等），以表征元素或事件在基于时间层面上的进度。在Kafka中，水位的概念反而与时间无关，而是与位置信息相关。严格来说，它表示的就是位置信息，即位移（offset）。取 partition 对应的 ISR中 最小的 LEO 作为 HW，consumer 最多只能消费到 HW 所在的位置上一条信息。
- LSO：是 LastStableOffset 的简称，对未完成的事务而言，LSO 的值等于事务中第一条消息的位置(firstUnstableOffset)，对已完成的事务而言，它的值同 HW 相同
- LW：Low Watermark 低水位, 代表 AR 集合中最小的 logStartOffset 值。

8. 数据传输的事务有几种？

	数据传输的事务定义通常有以下三种级别：

（1）At most once: 消息不会被重复发送，最多被传输一次，但也有可能一次不传输。

（2）At least once: 消息不会被漏发送，最少被传输一次，但也有可能被重复传输。

（3）Exactly once: 不会漏传输也不会重复传输,每个消息都传输被传输。

9. Kafka 高效文件存储设计特点

- Kafka把topic中一个parition大文件分成多个小文件段，通过多个小文件段，就容易定期清除或删除已经消费完文件，减少磁盘占用。
- 通过索引信息可以快速定位message和确定response的最大大小。
- 通过index元数据全部映射到memory，可以避免segment file的IO磁盘操作。
- 通过索引文件稀疏存储，可以大幅降低index文件元数据占用空间大小

10. Kafka创建Topic时如何将分区放置到不同的Broker中

- 副本因子不能大于 Broker 的个数；
- 第一个分区（编号为0）的第一个副本放置位置是随机从 brokerList 选择的；
- 其他分区的第一个副本放置位置相对于第0个分区依次往后移。也就是如果我们有5个 Broker，5个分区，假设第一个分区放在第四个 Broker 上，那么第二个分区将会放在第五个 Broker 上；第三个分区将会放在第一个 Broker 上；第四个分区将会放在第二个 Broker 上，依次类推；
- 剩余的副本相对于第一个副本放置位置其实是由 nextReplicaShift 决定的，而这个数也是随机产生的

11. 谈一谈 Kafka 的再均衡(rebalance)

	在Kafka中，当有新消费者加入或者订阅的topic数发生变化时，会触发Rebalance(再均衡：在同一个消费者组当中，分区的所有权从一个消费者转移到另外一个消费者)机制，Rebalance顾名思义就是重新均衡消费者消费。Rebalance的过程如下：

	①所有成员都向coordinator发送请求，请求入组。一旦所有成员都发送了请求，coordinator会从中选择一个consumer担任leader的角色，并把组成员信息以及订阅信息发给leader。

	②leader开始分配消费方案，指明具体哪个consumer负责消费哪些topic的哪些partition。一旦完成分配，leader会将这个方案发给coordinator。coordinator接收到分配方案之后会把方案发给各个consumer，这样组内的所有成员就都知道自己应该消费哪些分区了。所以对于Rebalance来说，Coordinator起着至关重要的作用。

12. Kafka 是如何实现高吞吐率的？

- 顺序读写；
- 零拷贝
- 文件分段
- 批量发送
- 数据压缩

13. Kafka缺点

- 由于是批量发送，数据并非真正的实时；
- 对于mqtt协议不支持；
- 不支持物联网传感数据直接接入；
- 仅支持统一分区内消息有序，无法实现全局消息有序；
- 监控不完善，需要安装插件；
- 依赖zookeeper进行元数据管理；sKafka目录结构

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101518253.png)

**参考：** 
尚硅谷大数据kafka教程

[32 道常见的 Kafka 面试题你都会吗？附答案	](https://zhuanlan.zhihu.com/p/82998949)