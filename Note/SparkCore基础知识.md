# SparkCore基础知识
## 1 Spark运行环境

1） Local模式

不需要其他任何节点资源就可以在本地执行Spark代码的环境，一般用于教学、调试、演示。

2） StandAlone模式

StandAlone模式即独立部署模式，体现了经典的master-slave模式。在集群中选择一个节点作为Master，另外其他节点都可以作为Worker节点。

3） Yarn模式

StandAlone模式由Spark自身提供计算资源，无需其他框架提供资源。这种方式降低了和其他第三方资源框架的耦合性，独立性很强。但Spark主要是一个计算框架，而不是资源调度框架，在实际的生产环境中，它在Yarn环境下运行的情况更为常见。

**Spark常用端口号：**

- Spark查看当前Spark-shell运行任务情况端口号：4040（计算）
- Spark Master内部通信服务端口号：7077
- Standalone模式下，Spark Master Web端口号：8080（资源）
- Spark历史服务器端口号：18080
- Hadoop Yarn任务运行情况查看端口号：8088

## 2 Spark运行架构

### 2.1 运行架构

**Spark框架的核心是一个计算引擎，整体来说它采用了标准的master-slave结构**
![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612150.png)


### 2.2 核心组件

1）Driver

Spark的驱动器节点，用于执行Spark任务中的main方法，负责实际代码执行工作。
	
Driver在Spark作业执行时主要负责：

- 将用户程序转换为job
- 在Executor之间调度task
- 跟踪Exectutor的执行情况
- 通过UI展示查询运行情况

2） Executor

Spark Executor是集群中工作节点（Work）中的一个JVM进程，负责在Spark作业中运行具体Task，任务之间彼此独立。Spark应用启动时，Executor节点被同时启动，并且伴随着整个Spark应用的生命周期而存在。如果由Executor节点发生了故障或崩溃，Spark应用也可以继续执行，会将出错节点上的任务调度到其他Executor节点上继续运行。
	
Executor的核心功能：

- 负责运行Spark应用的任务，并将结果返回给Driver
- 通过自身的Block Manager为用户程序中要求缓存的RDD提供内存式存储。RDD是直接缓存在Executor进程内的，因此任务可以在运行时充分利用缓存数据加速。

3）Master&Worker

Spark集群的独立部署环境中，不需要依赖其他的资源调度框架，自身就实现了资源调度的功能，所以环境中还有两个其他的核心组件：Master和Worker，这里的Master是一个进程，主要负责资源的调度和分配，并进行集群的监控等职责，类似于Yarn环境中的RM；Worker也是一个进程，运行在集群中的一台服务器上，由Master分配资源对数据进行并行的计算和处理，类似于Yarn环境中的NM。

4）Application Master

Hadoop用户向YARN集群提交应用程序时，提交程序中应该包含ApplicationMaster，用于向资源调度器申请执行任务的资源容器Container，运行用户自己的程序任务job，监控整个任务的执行，跟踪整个任务的状态，处理任务失败等异常情况。
	
简而言之，RM和Driver之间的解耦合靠的是ApplicationMaster。

### 2.3 核心概念

1）Executor和Core

Spark Executor是集群中运行在工作节点（Worker）中的一个JVM进程，是整个集群中专门用于计算的节点。在提交应用中，可以提供参数指定计算节点的个数，以及对应的资源。这里的资源一般指的是工作节点Executor的内存大小和虚拟CPU（Core）数量。

2）并行度(Parallelism)

在分布式计算框架中一般都是多个任务同时执行，由于任务分布在不同的计算节点进行计算，所以能够真正地实现多任务**并行**执行。将整个集群并行执行任务的数量称之为**并行度**。

3）有向无环图(DAG)

大数据计算引擎框架根据使用方式不同分为四类，其中第一类就是Hadoop所承载的MapReduce，它将计算分为Map和Reduce两个阶段，对于上层应用来说要实现这两个阶段，就需要实现多个job的串联。这样的弊端催生了支持DAG框架的产生。支持DAG的框架被划分为第二代计算引擎。Spark是第三代计算引擎，主要特点是Job内部的DAG支持（不跨越job），以及实时计算。

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612158.png)


DAG是由Spark程序直接映射成的数据流的高级抽象模型。简单理解就是将整个程序计算的执行过程用图形表示出来，更易于理解。

### 2.4 提交流程

提交流程指的是开发人员根据需求写的应用程序通过Spark客户端提交给Spark运行环境执行计算的流程。以下是基于Yarn环境的提交流程：

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612224.png)


Spark应用程序提交到Yarn环境中执行的时候，一般会有两种部署执行的方式：Client和Cluster。两种模式的区别主要在于：Driver程序运行节点位置

1）Yarn Client模式

Client模式将用于监控和调度的**Driver模块在客户端执行**，而不是在Yarn中，一般用于测试：

- Driver在任务提交的本地机器上运行
- Driver启动后会和RM通讯申请启动Application Master
- RM分配container，在合适的NM上启动Application Master，负责向RM申请Executor内存
- RM接到ApplicationMaster的资源后会分配container，然后ApplicationMaster在资源分配指定的NM上启动Executor进程
- Executor进程启动后会向Driver反向注册，Executor全部注册完Driver开始执行main函数
- 之后执行到Action算子时，触发一个job，并根据宽依赖开始划分stage，每个stage生成对应的TaskSet，之后将task分发到各个Executor上执行。

2）Yarn Cluster模式

Cluster模式将用于监控和调度的Driver模块启动在Yarn集群资源中执行。一般应用于实际生产环境。

- 在Yarn Cluster模式下，任务提交后会和RM通讯申请启动ApplicationMaster
- 随后RM分配container，在合适的NM上启动Application Master，此时的Application Master就是Driver
- Driver启动后向RM申请Executor内存，RM接到ApplicationMaster的资源后会分配container，然后在合适的NodeManager上启动Executor进程；
- Executor进程启动后会向Driver反向注册，Executor全部注册完成后Driver开始执行main函数
- 之后执行到Action算子时，触发一个job，并根据宽依赖开始划分stage，每个stage生成对应的TaskSet，之后将task分发到各个Executor上执行。

## 3 Spark核心编程

**Spark计算框架中有三大数据结构：RDD，累加器，广播变量**

### 3.1 RDD

RDD叫做弹性分布数据集，是Spark中最基本的**数据处理模型**。代码中是一个抽象类，它代表一个弹性的、不可变、可分区、里面元素可并行计算的集合。

- **弹性：**存储的弹性：内存与磁盘的自动切换

容错的弹性：数据丢失可恢复
		
计算的弹性：计算出错重试机制
		
切片的弹性：可根据需要重新分片

- 分布式：数据存储在大数据集群不同节点上
- 数据集：RDD封装了计算逻辑，并不保存数据
- 数据抽象：RDD是一个抽象类，需要子类具体实现
- 不可变：RDD封装了计算逻辑，是不可以改变的，想要改变只能产生新的RDD，在新的RDD里面封装计算逻辑

**RDD核心属性：**

1）分区列表：RDD数据结构中存在分区列表，用于执行任务时并行计算，是实现分布式计算的重要属性。

```scala
protected def getPartitions: Array[Parititon]
```

2） 分区计算函数：Spark在计算时，使用分区函数对每一个分区进行计算。

```scala
def compute(split:Partition, context:TaskContext):Iterator[T]
```

3）RDD之间依赖关系：RDD是计算模型的封装，当需求中需要将多个计算模型进行组合时，就需要将多个RDD建立依赖关系。

4）分区器：当数据为KV类型数据时，可以通过设定分区器自定义数据的分区。

5）首选位置：计算数据时，可以根据计算节点的状态选择不同的节点位置进行计算。

### 3.2 RDD的创建和分区操作

RDD使用分区来分布式并行处理数据，并且要做到尽量少的在不同的Executor之间使用网络交换数据。所以当使用RDD读取数据的时候，会尽量的在物理上靠近数据源。比如说在读取HDFS中数据的时候，会尽量地保持RDD的分区和数据源的分区数、分区模式等一一对应。

Spark中RDD有四种创建模式：
	
1）从集合（内存）中创建RDD：

```scala
sparkContext.parallelize(List(1,2,3,4)，minPartition = x)
sparkContext.makeRDD(List(1,2,3,4))
// minPartition可以用来设定分区数量
```

2）从外部存储（文件）创建RDD
由外部存储系统的数据集创建RDD，包括：本地文件系统，Hadoop支持的数据集，如Hbase、HDFS等。

```
sparkContext.textFile(Path)
```

3）从其他RDD创建

通过一个RDD运算完以后，将数据模型保存为新的RDD

4）直接创建RDD

使用new的方式直接构造RDD

### 3.3 RDD工作原理

在Yarn环境中，RDD的工作原理如下：
	
1）启动Yarn集群环境

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612232.png)


2）Spark通过申请资源创建调度节点和计算节点

![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612271.png)

3）Spark框架根据需求将计算逻辑根据分区划分成不同的任务
![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612311.png)


4）调度节点将任务根据计算节点状态发送到对应的计算节点进行计算
![在这里插入图片描述](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612610.png)

**从上述流程可以看出RDD在整个流程中主要用于将逻辑封装，并生成Task发送给Executor节点执行计算。**

### 3.4 RDD转换算子

| Transformation 算子                                          | Meaning（含义）                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **map**(*func*)                                              | 对原 RDD 中每个元素运用 *func* 函数，并生成新的 RDD          |
| **filter**(*func*)                                           | 对原 RDD 中每个元素使用*func* 函数进行过滤，并生成新的 RDD   |
| **flatMap**(*func*)                                          | 与 map 类似，但是每一个输入的 item 被映射成 0 个或多个输出的 items（ *func* 返回类型需要为 Seq ）。 |
| **mapPartitions**(*func*)                                    | 与 map 类似，但函数单独在 RDD 的每个分区上运行， *func*函数的类型为 Iterator<T> => Iterator<U> ，其中 T 是 RDD 的类型，即 RDD[T] |
| **mapPartitionsWithIndex**(*func*)                           | 与 mapPartitions 类似，但 *func* 类型为 (Int, Iterator<T>) => Iterator<U> ，其中第一个参数为分区索引 |
| **sample**(*withReplacement*, *fraction*, *seed*)            | 数据采样，有三个可选参数：设置是否放回（withReplacement）、采样的百分比（*fraction*）、随机数生成器的种子（seed）； |
| **union**(*otherDataset*)                                    | 合并两个 RDD                                                 |
| **intersection**(*otherDataset*)                             | 求两个 RDD 的交集                                            |
| **distinct**([*numTasks*]))                                  | 去重                                                         |
| **groupByKey**([*numTasks*])                                 | 按照 key 值进行分区，即在一个 (K, V) 对的 dataset 上调用时，返回一个 (K, Iterable<V>) **Note:** 如果分组是为了在每一个 key 上执行聚合操作（例如，sum 或 average)，此时使用 `reduceByKey` 或 `aggregateByKey` 性能会更好。 **Note:** 默认情况下，并行度取决于父 RDD 的分区数。可以传入 `numTasks` 参数进行修改。 |
| **reduceByKey**(*func*, [*numTasks*])                        | 按照 key 值进行分组，并对分组后的数据执行归约操作。          |
| **aggregateByKey**(*zeroValue*,*numPartitions*)(*seqOp*, *combOp*, [*numTasks*]) | 当调用（K，V）对的数据集时，返回（K，U）对的数据集，其中使用给定的组合函数和 zeroValue 聚合每个键的值。与 groupByKey 类似，reduce 任务的数量可通过第二个参数进行配置。 |
| **sortByKey**([*ascending*], [*numTasks*])                   | 按照 key 进行排序，其中的 key 需要实现 Ordered 特质，即可比较 |
| **join**(*otherDataset*, [*numTasks*])                       | 在一个 (K, V) 和 (K, W) 类型的 dataset 上调用时，返回一个 (K, (V, W)) pairs 的 dataset，等价于内连接操作。如果想要执行外连接，可以使用 `leftOuterJoin`, `rightOuterJoin` 和 `fullOuterJoin` 等算子。 |
| **cogroup**(*otherDataset*, [*numTasks*])                    | 在一个 (K, V) 对的 dataset 上调用时，返回一个 (K, (Iterable<V>, Iterable<W>)) tuples 的 dataset。 |
| **cartesian**(*otherDataset*)                                | 在一个 T 和 U 类型的 dataset 上调用时，返回一个 (T, U) 类型的 dataset（即笛卡尔积）。 |
| **coalesce**(*numPartitions*)                                | 将 RDD 中的分区数减少为 numPartitions。                      |
| **repartition**(*numPartitions*)                             | 随机重新调整 RDD 中的数据以创建更多或更少的分区，并在它们之间进行平衡。 |
| **repartitionAndSortWithinPartitions**(*partitioner*)        | 根据给定的 partitioner（分区器）对 RDD 进行重新分区，并对分区中的数据按照 key 值进行排序。这比调用 `repartition` 然后再 sorting（排序）效率更高，因为它可以将排序过程推送到 shuffle 操作所在的机器。 |

RDD根据数据处理方式的不同将算子整体上分为Value类型、双Value类型和Key-Value类型

**value类型**

1） map

函数说明：将处理的数据逐条进行映射转换。这里的转换可以是类型的转换也可以是值的转换。要求数据经过map后不会增多或者减少。

```scala
val list = List(1,2,3)
sc.parallelize(list).map(_ * 10).foreach(println)

// 输出结果： 10 20 30 （这里为了节省篇幅去掉了换行,后文亦同）
```

2） mapPartitions

函数说明：将待处理的数据以分区为单位发送到计算节点进行处理，与 map 类似，但函数单独在 RDD 的每个分区上运行， *func*函数的类型为 `Iterator<T> => Iterator<U>` (其中 T 是 RDD 的类型)，即输入和输出都必须是可迭代类型。

```scala
val list = List(1, 2, 3, 4, 5, 6)
sc.parallelize(list, 3).mapPartitions(iterator => {
  val buffer = new ListBuffer[Int]
  while (iterator.hasNext) {
    buffer.append(iterator.next() * 100)
  }
  buffer.toIterator
}).foreach(println)
//输出结果
100 200 300 400 500 600
```

**Map算子是以分区内一个数据为单位依次执行，类似于串行操作；mapPartitions算子是以分区为单位进行批处理操作。**

3）mapPartitionWithIndex

函数说明：将待处理的数据以分区为单位发送到计算节点进行处理，这里的处理可以是任意的处理哪怕是过滤数据，**在处理的同时可以获取当前分区索引。**

```scala
val list = List(1, 2, 3, 4, 5, 6)
sc.parallelize(list, 3).mapPartitionsWithIndex((index, iterator) => {
  val buffer = new ListBuffer[String]
  while (iterator.hasNext) {
    buffer.append(index + "分区:" + iterator.next() * 100)
  }
  buffer.toIterator
}).foreach(println)
//输出
0 分区:100
0 分区:200
1 分区:300
1 分区:400
2 分区:500
2 分区:600
```

4）flatMap

函数说明：将处理后的数据进行扁平化后再进行映射处理（继续拆分），所以算子也称扁平映射。将一对多拆分成一对一。每一个输入的 item 会被映射成 0 个或多个输出的 items（ *func* 返回类型需要为 `Seq`

```scala
val list = List(List(1, 2), List(3), List(), List(4, 5))
sc.parallelize(list).flatMap(_.toList).map(_ * 10).foreach(println)

// 输出结果 ： 10 20 30 40 50
```

5）glom

函数说明：将同一个分区的数据直接转换为相同类型的内存数组进行处理，分区不变。

```scala
def glom():RDD[Array[T]]
//函数签名

val dataRDD = sparkContext.makeRDD(List(1,2,3,4)
                                   ,1)
val dataRDD1:RDD[Array[Int]] = dataRDD.glom()
```

6）groupBy

函数说明：将数据根据指定的规则进行分组，分区默认不变，但是数据会被打乱重新组合，我们将这样的操作成为**shuffle**。极限情况下数据可能被分在同一个分区中。

```scala
def groupBy[K](f:T => K)(implicit kt:ClassTag[K]): RDD[(K,Iterable[T])]
//函数签名

val dataRDD = sparkContext.makeRDD(List(1,2,3,4),1)
val dataRDD1 = dataRDD.groupBy(
    _%2
)
```

7）filter

函数说明：将数据根据指定的规则进行筛选过滤，符合规则的数据保留，不符合规则的数据丢弃。当数据进行筛选过滤后，分区不变，但是分区内的数据可能不均衡。在生产环境下，可能会出现数据倾斜。

```scala
val list = List(3, 6, 9, 10, 12, 21)
sc.parallelize(list).filter(_ >= 10).foreach(println)

// 输出： 10 12 21
```

8）sample

函数说明：根据指定的规则从数据集中抽取数据。有三个可选参数：设置是否放回 (withReplacement)、采样的百分比 (fraction)、随机数生成器的种子 (seed)

```scala
val list = List(1, 2, 3, 4, 5, 6)
sc.parallelize(list).sample(withReplacement = false, fraction = 0.5).foreach(println)
```

9）distinct

函数说明：将数据集中重复的数据去重

```scala
val list = List(1, 2, 2, 4, 4)
sc.parallelize(list).distinct().foreach(println)
// 输出: 4 1 2
```

10）coalesce

函数说明：根据数据量缩减分区，用于大数据过滤后，提高小数据的执行效率。

```scala
def coalesce(numPartitions:Int,shuffle:Boolean = false,
			partitionCoalescer:Option[PartitionCoalescer] = Option.empty)
			(implicit ord: Ordering[T] = null)
			:RDD[T]
```

11）repartition

函数说明：该操作内部其实执行的是coalesce操作，参数shuffle的默认值为true。无论是将分区多的RDD转换为分区数少的RDD转换为分区数多的RDD，repartition操作都可以完成，因为无论如何都会经过shuffle过程。

12）sortBy

函数说明：该操作用于排序数据。再排序之前可以将数据通过f函数进行处理，之后按照f函数处理的结果进行排序，默认为升序排序。排列后新产生的RDD的分区数与原RDD的分区数一致，中间存在shuffle过程。

```scala
val list02 = List(("hadoop",100), ("spark",90), ("storm",120))
sc.parallelize(list02).sortBy(x=>x._2,ascending=false).foreach(println)
// 输出
(storm,120)
(hadoop,100)
(spark,90)
```

**双value类型**

13） intersection

函数说明：对源RDD和参数RDD求交集后返回一个新的RDD

```scala
def intersection(other:RDD[T]):RDD[T]
//函数签名

val dataRDD1 = sparkContext.makeRDD(List(1,2,3,4))
val dataRDD2 = sparkContext.makeRDD(List(3,4,5,6))
val dataRDD = dataRDD1.intersection(dataRDD2)
```

14）union

函数说明：对源RDD和参数RDD求并集后返回一个新的RDD

```scala
val list1 = List(1, 2, 3)
val list2 = List(4, 5, 6)
sc.parallelize(list1).union(sc.parallelize(list2)).foreach(println)
// 输出: 1 2 3 4 5 6
```

15）subtract

函数说明：以一个RDD元素为主，去除两个RDD中重复元素，将其他元素保留下来，求差集。

```scala
def subtract(other:RDD[T]):RDD[T]
//函数签名

val dataRDD1 = sparkContext.makeRDD(List(1,2,3,4))
val dataRDD2 = sparkContext.makeRDD(List(3,4,5,6))
val dataRDD = dataRDD1.subtract(dataRDD2)
```

16）zip

函数说明：将RDD中的元素以键值对的形式进行合并。其中，键值对中的Key为第1个RDD中的元素，Value为第2个RDD中的相同位置的元素。

```scala
def zip[U:ClassTag](ohter:RDD[U]):RDD[(T,U)]
//函数签名

val dataRDD1 = sparkContext.makeRDD(List(1,2,3,4))
val dataRDD2 = sparkContext.makeRDD(List(3,4,5,6))
val dataRDD = dataRDD1.zip(dataRDD2)
```

**key-value类型**

17）partitionBy

函数说明：将数据按照指定Partitioner重新进行分区。Spark默认的分区器是HashPartioner

```scala
def partitionBy([partition:Partitioner]):RDD[(K,V)]
//函数签名

val rdd:RDD[(Int,String)] = 
	sc.makeRDD(Array((1,"aaa"),(2,"bbb"),(3,"ccc")),3)
import org.apache.spark.HashPartitioner
val rdd2: RDD[(Int,String)] = rdd.partitionBy(new HashPartitioner(2))
```

**18）reduceByKey**

```scala
val list = List(("hadoop", 2), ("spark", 3), ("spark", 5), ("storm", 6), ("hadoop", 2))
sc.parallelize(list).reduceByKey(_ + _).foreach(println)

//输出
(spark,8)
(hadoop,4)
(storm,6)
```

19）groupByKey

函数说明：将数据源的数据根据key对value进行分组

```scala
val list = List(("hadoop", 2), ("spark", 3), ("spark", 5), ("storm", 6), ("hadoop", 2))
sc.parallelize(list).groupByKey().map(x => (x._1, x._2.toList)).foreach(println)

//输出：
(spark,List(3, 5))
(hadoop,List(2, 2))
(storm,List(6))
```

**reduceByKey和groupByKey的区别？**

reduceByKey和groupByKey都存在shuffle操作，但是reduceByKey可以在shuffle前对分区进行预聚合功能，这样可以减少落盘的数据量；groupByKey只进行分组，不会让数据量减少。

20）aggregateByKey

函数说明：将数据根据不同的规则进行**分区内计算和分区间计算**。

当调用（K，V）对的数据集时，返回（K，U）对的数据集，其中使用给定的组合函数和 zeroValue 聚合每个键的值。与 `groupByKey` 类似，reduce 任务的数量可通过第二个参数 `numPartitions` 进行配置。示例如下：

```scala
// 为了清晰，以下所有参数均使用具名传参
val list = List(("hadoop", 3), ("hadoop", 2), ("spark", 4), ("spark", 3), ("storm", 6), ("storm", 8))
sc.parallelize(list,numSlices = 2).aggregateByKey(zeroValue = 0,numPartitions = 3)(
      seqOp = math.max(_, _),
      combOp = _ + _
    ).collect.foreach(println)
//输出结果：
(hadoop,3)
(storm,8)
(spark,7)
```

这里使用了 `numSlices = 2` 指定 aggregateByKey 父操作 parallelize 的分区数量为 2，其执行流程如下：

[![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208081730931.png)](https://github.com/SamuelZhu12/God-Of-BigData/blob/master/pictures/spark-aggregateByKey.png)

基于同样的执行流程，如果 `numSlices = 1`，则意味着只有输入一个分区，则其最后一步 combOp 相当于是无效的，执行结果为：

```
(hadoop,3)
(storm,8)
(spark,4)
```

同样的，如果每个单词对一个分区，即 `numSlices = 6`，此时相当于求和操作，执行结果为：

```
(hadoop,5)
(storm,14)
(spark,7)
```

`aggregateByKey(zeroValue = 0,numPartitions = 3)` 的第二个参数 `numPartitions` 决定的是输出 RDD 的分区数量，想要验证这个问题，可以对上面代码进行改写，使用 `getNumPartitions` 方法获取分区数量：

```scala
sc.parallelize(list,numSlices = 6).aggregateByKey(zeroValue = 0,numPartitions = 3)(
  seqOp = math.max(_, _),
  combOp = _ + _
).getNumPartitions
```

[![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208081730929.png)](

21）foldByKey

函数说明：当分区内计算规则和分区间计算规则相同时，aggregateByKey就可以简化为foldByKey

```scala
def foldByKey(zeroValue: V)(func:(V,V)=>V):RDD[(K,V)]
//函数签名


val dataRDD1 = sparkContext.makeRDD(List(("a",1),("b",2),("c",3)))
val dataRDD2 = dataRDD1.foldByKey(0)(_+_)
```

22） CombineByKey

函数说明：最通用的对key-value型RDD进行聚集操作的聚集函数（aggregation function）。类似于aggregate()，combineByKey()允许用户返回值与输入不一致。

```scala
def combineByKey[C](
	createCombiner:V=>C,
	mergeValue:(C,V)=>C.
	mergeCombiners:(C,C)=>C):RDD[(K,C)]
)
//函数签名

val list:List[(String,Int)] = List(("a",88),("b",95),("a",91),("b",93),("a",95),("b",98))
val input: RDD[(String,Int)] = sc.makeRDD(list,2)
val combineRDD: RDD[(String,(Int,Int))] = input.combineByKey(
	(_, 1),
    (acc:(Int,Int),v) => (acc._1+v,acc._2 + 1),
    (acc1:(Int,Int),acc2:(Int,Int)) => (acc1._1+acc2._1,acc1._2+acc2._2)
)
```

23）sortByKey

函数说明：在一个（K,V）的RDD上调用，K必须实现Orderd接口（特质），返回一个按照key进行排序的RDD

```scala
val list01 = List((100, "hadoop"), (90, "spark"), (120, "storm"))
sc.parallelize(list01).sortByKey(ascending = false).foreach(println)
// 输出
(120,storm)
(100,hadoop)
(90,spark)
```

24）join

函数说明：在类型为(K,V)和(K,W)的RDD上调用，返回一个相同key对应的所有元素连接在一起的(K,(V,W))的RDD

```scala
def join[W](other:RDD[(K,w)]):RDD[(K,(V,W))]
//函数签名

val rdd:RDD[(Int,String)] = sc.makeRDD(Array((1,"a"),(2,"b"),(3,"c")))
val rdd1:RDD[(Int,Int)] = sc.makeRDD(Array((1,4),(2,5),(3,6)))
rdd.join(rdd1).collect().foreach(println)
```

25）cogroup

函数说明：在类型为(K,V)和(K,W)的RDD上调用，返回一个(K,(Iterable<V>,iterable<W>))类型的RDD

```scala
val list01 = List((1, "a"),(1, "a"), (2, "b"), (3, "e"))
val list02 = List((1, "A"), (2, "B"), (3, "E"))
val list03 = List((1, "[ab]"), (2, "[bB]"), (3, "eE"),(3, "eE"))
sc.parallelize(list01).cogroup(sc.parallelize(list02),sc.parallelize(list03)).foreach(println)

// 输出： 同一个 RDD 中的元素先按照 key 进行分组，然后再对不同 RDD 中的元素按照 key 进行分组
(1,(CompactBuffer(a, a),CompactBuffer(A),CompactBuffer([ab])))
(3,(CompactBuffer(e),CompactBuffer(E),CompactBuffer(eE, eE)))
(2,(CompactBuffer(b),CompactBuffer(B),CompactBuffer([bB])))
```

**cogroup返回的是一个元组，元组的value是List的集合（Iterable），如(K,(Iterable[V],Iterable[W]))，Iterable[V]中是第一个RDD中key相同的value，Iterable[W]中是第二个RDD中key相同的value。一般在开发过程中用的较少，作为中间过程存在。**

### 3.5 RDD行动算子

| Action（动作）                                     | Meaning（含义）                                              |
| -------------------------------------------------- | ------------------------------------------------------------ |
| **reduce**(*func*)                                 | 使用函数*func*执行归约操作                                   |
| **collect**()                                      | 以一个 array 数组的形式返回 dataset 的所有元素，适用于小结果集。 |
| **count**()                                        | 返回 dataset 中元素的个数。                                  |
| **first**()                                        | 返回 dataset 中的第一个元素，等价于 take(1)。                |
| **take**(*n*)                                      | 将数据集中的前 *n* 个元素作为一个 array 数组返回。           |
| **takeSample**(*withReplacement*, *num*, [*seed*]) | 对一个 dataset 进行随机抽样                                  |
| **takeOrdered**(*n*, *[ordering]*)                 | 按自然顺序（natural order）或自定义比较器（custom comparator）排序后返回前 *n* 个元素。只适用于小结果集，因为所有数据都会被加载到驱动程序的内存中进行排序。 |
| **saveAsTextFile**(*path*)                         | 将 dataset 中的元素以文本文件的形式写入本地文件系统、HDFS 或其它 Hadoop 支持的文件系统中。Spark 将对每个元素调用 toString 方法，将元素转换为文本文件中的一行记录。 |
| **saveAsSequenceFile**(*path*)                     | 将 dataset 中的元素以 Hadoop SequenceFile 的形式写入到本地文件系统、HDFS 或其它 Hadoop 支持的文件系统中。该操作要求 RDD 中的元素需要实现 Hadoop 的 Writable 接口。对于 Scala 语言而言，它可以将 Spark 中的基本数据类型自动隐式转换为对应 Writable 类型。(目前仅支持 Java and Scala) |
| **saveAsObjectFile**(*path*)                       | 使用 Java 序列化后存储，可以使用 `SparkContext.objectFile()` 进行加载。(目前仅支持 Java and Scala) |
| **countByKey**()                                   | 计算每个键出现的次数。                                       |
| **foreach**(*func*)                                | 遍历 RDD 中每个元素，并对其执行*fun*函数                     |

1）reduce

函数说明：聚集RDD中的所有元素，先聚合分区内数据，再聚合分区间数据

```scala
 val list = List(1, 2, 3, 4, 5)
sc.parallelize(list).reduce((x, y) => x + y)
sc.parallelize(list).reduce(_ + _)

// 输出 15
```

2）collect

函数说明：在驱动程序中，以数组Array的形式返回数据集的所有元素

```scala
def collect():Array[T]
//函数签名

val rdd: RDD[Int] = sc.makeRDD(List(1,2,3,4))
rdd.collect.foreach(println)
```

3）count

函数说明：返回RDD中元素的个数

```scala
def count():Long
//函数签名

val rdd:RDD[Int] = sc.makeRDD(List(1,2,3,4))
//返回RDD中元素的个数

val countResult: Long = rdd.count()
```

4）first

函数说明：返回RDD中第一个元素

```scala
def first():T
//

val rdd:RDD[Int] = sc.makeRDD(List(1,2,3,4))
val firstResult: Int = rdd.first()
println(firstResult)
```

5）take

函数说明：返回一个由RDD的前n个元素组成的数组

```scala
def take(num:Int) :Array[T]
//

val rdd:RDD[Int] = sc.makeRDD(List(1,2,3,4))
val takeResult: Array[Int] = rdd.take(2)
println(takeResult.mkString(","))
```

6）takeOrdered

函数说明：按自然顺序（natural order）或自定义比较器（custom comparator）排序后返回前 *n* 个元素。需要注意的是 `takeOrdered` 使用隐式参数进行隐式转换，以下为其源码。所以在使用自定义排序时，需要继承 `Ordering[T]` 实现自定义比较器，然后将其作为隐式参数引入。

```scala
def takeOrdered(num: Int)(implicit ord: Ordering[T]): Array[T] = withScope {
  .........
}
// 继承 Ordering[T],实现自定义比较器，按照 value 值的长度进行排序
class CustomOrdering extends Ordering[(Int, String)] {
    override def compare(x: (Int, String), y: (Int, String)): Int
    = if (x._2.length > y._2.length) 1 else -1
}

val list = List((1, "hadoop"), (1, "storm"), (1, "azkaban"), (1, "hive"))
//  引入隐式默认值
implicit val implicitOrdering = new CustomOrdering
sc.parallelize(list).takeOrdered(5)

// 输出： Array((1,hive), (1,storm), (1,hadoop), (1,azkaban)
```

7）aggregate

函数说明：分区的数据通过初始值和分区内的数据进行聚合，然后再和初始值进行分区间的数据聚合。

```scala
def aggregate[U:ClassTag](zeroValue:U)(seqOp:(U,T)=>U),combOp:(U,U)=>U
//
val rdd:RDD[Int] = sc.makeRDD(List(1,2,3,4))
val result: Int = rdd.aggregate(0)(_+_,_+_)
val result: Int = rdd.aggregate(10)(_+_,_+_)
```

8）fold

函数说明：折叠操作，aggregate的简化版操作，区内和区间操作相同

```scala
def fold(zeroValue:T)(op:(T,T)=>T):T
//

val rdd:RDD[Int] = sc.makeRDD(List(1,2,3,4))
val foldResult: Int = rdd.fold(0)(_+_)
```

9）countByKey

函数说明：统计每种key的个数

```scala
val list = List(("hadoop", 10), ("hadoop", 10), ("storm", 3), ("storm", 3), ("azkaban", 1))
sc.parallelize(list).countByKey()

// 输出： Map(hadoop -> 2, storm -> 2, azkaban -> 1)
```

10）save相关算子

函数说明：将数据保存到不同格式的文件中

```scala
//保存为Text
rdd.saveAsTextFile("output")

//序列化成对象保存到文件
rdd.saveAsObjectFile("output1")

//保存成Sequencefile文件
rdd.map((_,1)).saveAsSequenceFile("output2")
```

11）foreach

函数说明：分布式遍历RDD中的每一个元素，调用指定函数

```scala
val rdd:RDD[Int] = sc.makeRDD(List(1,2,3,4))

//收集后打印
rdd.map(num=>num).collect().foreach(println)


//分布式打印
rdd.foreach(println)
```

### 3.6 RDD序列化

从计算角度来看，算子以外的代码都是在driver端执行，算子内的代码都是在executor端执行。在scala函数式编程中，会导致算子内需要用到算子外的数据，如果算子外的数据无法序列化，则无法传值给executor端执行。因此每次执行任务计算前，检查闭包内的对象是否可以继续序列化，这个操作称为**闭包检测**。

因此，在spark中要对rdd对象中的数据进行读取，需要将类继续序列化来访问：

```scala
class Search(query:String) extends Serializable{
	def isMatch(s:String): Boolean = {
		s.contains(query)
	}
	//函数序列化案例
	def getMatch(rdd:RDD[String]): RDD[String] = {
		rdd.filter(isMatch)
}

}

```

### 3.7 RDD依赖关系

1）RDD血缘关系

RDD只支持粗粒度转换，即在大量记录上执行的单个操作。将创建RDD的一系列Linage(血统)记录下来，以便恢复丢失的分区。

2）RDD窄依赖

窄依赖表示每一个父RDD的Partition最多被子RDD的一个Partition使用。如map、filter、union等操作。

3）RDD宽依赖

宽依赖表示同一个父RDD的Partition被多个子RDD的Partition依赖，会引起Shuffle。如groupByKey、reduceByKey、sortByKey等。

### 3.8 RDD任务划分

RDD任务切分中间分为：Application、Job、Stage和Task

- Application：初始化一个SparkContext即生成一个Application
- Job：一个Action算子就会生成一个Job
- Stage：一种并行计算的task。Stage等于宽依赖(ShuffleDependency)的个数加1；
- Task：在map(reduce)阶段并行的个数。一个Stage的阶段中，最后一个RDD的分区个数就是Task的个数。

```Application->Job->Stage->Task 每一层都是 1 对 n 的关系```

分组会产生shuffle，shuffle会落地产生磁盘文件，如果要进行网络传输那么就会有一个序列化的过程，在数据落到磁盘的时候会进行压缩（默认hash分区）

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208091034165.png" alt="在这里插入图片描述" style="zoom:50%;" />

### 3.9 RDD持久化

1）RDD Cache缓存

	RDD通过Cache或者Persist方法将前面的计算结果缓存，默认情况下会把数据以缓存在JVM的堆内存中。但是并不是这两个方法被调用时立即缓存，而是触发后面的action算子时，该RDD将会被缓存在计算节点的内存中，并供后面重用。

2）RDD CheckPoint

	CheckPoint实际上是将RDD中间结果写入磁盘。由于血缘依赖过长会造成容错成本过高，这样就不如在中间阶段做检查点容错，如果检查点之后有节点出现问题，可以从检查点开始重做血缘，减少开销。

3）Cacha和CheckPoint区别

	1）Cache缓存只是将数据保存起来，不切断血缘依赖
	
	2）Cache缓存的数据通常存储在磁盘、内存等地方，可靠性低。CheckPoint的数据通常存在HDFS等高容错、高可用的文件系统，可靠性高。
	
	3）建议对checkpoint()的RDD使用Cache缓存，这样checkpoint的job只需要从Cache缓存中读取数据即可，否则需要再从头计算一次RDD。

### 3.10 RDD分区器

	Spark目前支持Hash分区和Range分区，和用户自定义分区。Hash为当前的默认分区。分区器决定了RDD中分区的个数，RDD中每条数据经过Shuffle后进入哪个分区，进而决定Reduce个数。

1）Hash分区：对于给定的key，计算其hashCode，并除以分区个数取余。

2）Range分区：将一定范围内的数据映射到一个分区中，尽量保证每个分区数据均匀，而且分区间有序。


## 4 累加器

	累加器用来把Executor端变量信息聚合到Driver端。在Driver端程序中定义的变量，在Executor端的每个Task都会得到这个变量的一份新的副本，每个task更新这些副本的值后，传回Driver端进行Merge操作。

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612661.png" style="zoom:33%;" />

```scala
longAccumulator()
```

## 5 广播变量

	广播变量用来高效分发比较大的对象。向所有工作节点发送一个较大的只读值，以供一个或多个Spark操作使用。比如，应用要向所有节点发送一个较大的只读查询表，用广播变量就很合适。在多个并行操作中使用同一个变量，但是Spark会为每个任务分别发送。

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208051612709.png" style="zoom:33%;" />

**参考：尚硅谷Spark教程**
