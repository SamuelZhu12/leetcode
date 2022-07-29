# 数据仓库和Hive

## 1. 数据仓库

### 1.1 基本概念

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291058685.webp)

- 数据仓库(Data Warehouse)的目的是构建面向分析的集成化数据环境，为企业提供决策支持。它出于分析性报告和决策支持的目的而创建。

- 数据仓库本身并不生产任何数据，也不消费任何数据，数据全部来源于外部，并且开放给外部应用。

### 1.2 与数据库的区别

- 数据仓库与传统数据库的区别实际上是OLTP(On-line Transaction Processing)和OLAP(On-line Analytical Processing)的区别，前者侧重于事务，例如对于交易事件的数据记录；而后者则侧重于分析，面向主题进行设计，例如出报表等。
- 数据库是面向事务的设计，数据仓库是面向主题设计的。
- **数据库**设计是尽量避免冗余，一般针对某一业务应用进行设计；比如一张简单的User表，记录用户名、密码等简单数据即可，符合业务应用，但是不符合分析；**数据仓库**在设计是有意引入冗余，依照分析需求，分析维度、分析指标进行设计。

- 虽然传统的OLTP(例如MySQL、Oracle)型数据库也可以进行数据的分析，但随着数据量级的增加，很多数据存储于分布式存储系统中，想要跨集群、关联多种数据，OLTP就显得无能为力。
- 数据仓库，是在数据库已经大量存在的情况下，为了进一步挖掘数据资源、为了决策需要而产生的，它决不是所谓的“大型数据库”。

### 1.3 数据仓库分层

数仓的分层多种多样，按照数据流入流出的顺序，大致可分为：**源数据层、数据仓库层、数据应用层**。其中数据仓库层又分为DWD(etail)(明细数据层)和DWS(sum)(数据汇总层)。

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291128347.webp)

- 减少重复开发，在数据开发的过程中可以产生中间层，将公共逻辑下沉，减少重复计算；

- 清晰数据结构，每个分层分工明确，方便开发人员理解；

- 方便定位问题，通过分层了解数据血缘关系，在出问题的时候通过回溯定位问题；

- 简单化复杂问题，和分治法思想类似，分而治之，将复杂的问题简单化，还能解耦

## 2 Hive

**官方链接：**https://hive.apache.org/

### 2.1 基本概念

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291154800.png" alt="绘图1" style="zoom:67%;" />

- Hive是Hadoop生态圈中的一个**数据仓库工具**。

- Hive可以将**结构化的数据文件映射成一张数据库表**，并提供类SQL查询功能。
- Hive的本质是将HiveSQL这一类SQL查询语言转换为MapReduce任务进行运算。底层由HDFS来提供数据的存储支持，类似于MapReduce的客户端一样。

### 2.2 Hive的优缺点

**优点**：

- 类SQL语法，提供快速开发上手能力，免去开发人员写mapreduce的繁琐操作
- 支持用户自定义函数，可根据自己的需求来实现自己的函数
- 适用于处理大数据

**缺点**：

- 自动生成的MapReduce作业不够智能化
- 调优比较困难，粒度粗
- 延迟高，处理小规模数据集较慢

### 2.3 Hive架构原理

官网架构图：

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291558859.png)

简化架构图：

![](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202207291418070.png)

1. **用户接口：Client**

​		CLI(command-line interface)、JDBC/ODBC、WEBUI；

2. **元数据：Metastore**

​		元数据包括：表名、表所属的数据库（默认是default）、表的拥有者、列/分区字段、表的类型（内外表）、		表的数据所在目录等；

3. **Hadoop：**

​		使用HDFS进行存储，使用MapReduce进行计算；

4. **Driver：驱动器**

​		（1）解析器(SQL Parser)：将SQL字符串转换成抽象语法树，这一步一般都用第三方工具库完成，对AST进行			语法分析，比如表是否存在，字段是否存在，SQL语法是否有误等；

​		（2）编译器(Physical Plan)：将AST编译生成逻辑执行计划；

​		（3）优化器(Query Optimizer)：对逻辑执行计划进行优化；

​		（4）执行器(Execution)：把逻辑执行计划转换成可以运行的物理计划。对于Hive来说就是MR/Spark。

​		总之，Hive通过给用户提供一系列交互接口，接收到用户的指令(SQL)，使用自己的Driver，结合元数据(		MetaStore)，将这些指令翻译成MapReduce，提交到Hadoop中执行，最后将执行返回的结果输出到用户交		互接口。

### 2.3 Hive与数据库的比较

- Hive只是一种**数仓工具**，并不是数仓本身，更不是数据库。其查询语句类似于SQL语句

- Hive是针对数仓进行设计的，而数仓内容是读多写少，因此**Hive中不建议对数据进行改写**，所有的数据都是在加载的时候确定好的。而数据库中的数据通常是需要经常进行修改的，因此可以使用INSERT INTO ... VALUES添加数据，使用UPDATE ... SET修改数据
- Hive底层基于MapReduce，（3.0版本中索引被删除）查询数据时需要扫描整张表，同时由于MapReduce本身具有较高的延迟，因此Hive查询整体延迟较高。但当数据量庞大的时候，由于Hive可以进行并行计算，性能会远超于数据库。

### 2.4 HQL的执行流程

#### 2.4.1 Hive Stage的划分
1. 在生成map-reduce任务时，Hive深度优先方式遍历Operator tree（操作符树），遇到第一个reduceSink操作符时，该操作符之前的操作符便划分到一个map-reduce任务的Map任务中，然后该reduceSink到下一个reduceSink操作符之间的部分划分为map-reduce任务的Reduce任务。
2. 一个完整的MapReduce阶段代表一个stage。Hive中还有非MapReduce的stage，在MapReduce为计算框架时，基本以MapReduce的stage为主。
3.  由于Hive使用MapReduce计算引擎时无法直接使用不同阶段的结果。因此，每个阶段完成之后的结果都要输出到临时目录，供下一阶段读取，因此便将Operator tree分解成不同的stage。

#### 2.4.2 HQL具体执行流程

```
1) 一条hivesql -> 多个stage (深度优先遍历)（explain）
2) 每个stage可能会转化成一个job ( 一般来说,stage 数量会多于 job 数量 ) 
3) 每一个job对应yarn上的一个application
4) 每个application会有多个map task和多个reduce task (mr任务为细粒度资源申请)
5) 通常来说，每个task(map or reduce) 会对应一个 container,application的map task数量取决于split数量
   application的reduce task数量由hive自动推测
                  totalInputFileSize, 
                  hive.exec.reducers.bytes.per.reducer, 
                  hive.exec.reducers.max
                  double bytes = Math.max(totalInputFileSize, bytesPerReducer);
                  int reducers = ( int ) Math.ceil(bytes / bytesPerReducer);
                  reducers = Math.max( 1 , reducers);
                  reducers = Math.min(maxReducers, reducers);
6) 正在运行的任务，在yarn 的8088查看,运行完成的任务,在 job history(19090)里面查看
```

### 2.5 Hive数据类型

1. **基本数据类型**

| Hive      | Java    | 长度                                                 |
| --------- | ------- | ---------------------------------------------------- |
| TINYINT   | byte    | 1byte有符号整数                                      |
| SMALINT   | short   | 2byte有符号整数                                      |
| INT       | int     | 4byte有符号整数                                      |
| BIGINT    | long    | 8byte有符号整数                                      |
| BOOLEAN   | boolean | 布尔类型，true或者false                              |
| FLOAT     | float   | 单精度浮点数                                         |
| DOUBLE    | double  | 双精度浮点数                                         |
| STRING    | string  | 字符系列。可以指定字符集。可以使用单引号或者双引号。 |
| TIMESTAMP |         | 时间类型                                             |
| BINARY    |         | 字节数组                                             |

2. **集合数据类型**

   | 数据类型 | 描述                                                         | 语法示例 |
   | :------: | ------------------------------------------------------------ | -------- |
   |  STRUCT  | 和c语言中的struct类似，都可以通过“点”符号访问元素内容。例如，如果某个列的数据类型是STRUCT{first STRING, last STRING},那么第1个元素可以通过字段.first来引用。 | struct() |
   |   MAP    | MAP是一组键-值对元组集合，使用数组表示法可以访问数据。例如，如果某个列的数据类型是MAP，其中键->值对是’first’->’John’和’last’->’Doe’，那么可以通过字段名[‘last’]获取最后一个元素 | map()    |
   |  ARRAY   | 数组是一组具有相同类型和名称的变量的集合。这些变量称为数组的元素，每个数组元素都有一个编号，编号从零开始。例如，数组值为[‘John’, ‘Doe’]，那么第2个元素可以通过数组名[1]进行引用。 | Array()  |

3. **类型强制转换**

```
Hive的原子数据类型是可以进行隐式转换的，类似于Java的类型转换，例如某表达式使用INT类型，TINYINT会自动转换为INT类型，但是Hive不会进行反向转化，例如，某表达式使用TINYINT类型，INT不会自动转换为TINYINT类型，它会返回错误，除非使用CAST 操作。

1）隐式类型转换规则如下。
（1）任何整数类型都可以隐式地转换为一个范围更广的类型，如TINYINT可以转换成INT，INT可以转换成BIGINT。
（2）所有整数类型、FLOAT和STRING类型都可以隐式地转换成DOUBLE。
（3）TINYINT、SMALLINT、INT都可以转换为FLOAT。
（4）BOOLEAN类型不可以转换为任何其它的类型。

2）可以使用CAST操作显示进行数据类型转换，例如CAST('1' AS INT)将把字符串'1' 转换成整数1；
   如果强制类型转换失败，如执行CAST('X' AS INT)，表达式返回空值 NULL。
```

### 2.6 DDL操作

#### 2.6.1 表定义

1. 外部表

Hive并非认为完全拥有这份数据，删除该表只会删除metastore中表的元数据，并不会删除HDFS中的数据。多用来存储一些需要长久保存的日志信息等。

2. 内部表

Hive会（或多或少）控制着数据的生命周期，删除一个内部表，Hive表的元数据和HDFS中对应的数据会被一起删除。多用来存储逻辑过程中的中间表，临时表。

3. 分区表

分区表对应一个HDFS上的文件夹，该文件夹是该分区下所有的数据文件。Hive中的分区就是分目录。

4. 临时表（Temporary）

临时表仅对当前会话可见。数据将存储在用户的临时目录中，在会话结束时删除，不支持分区，不支持创建索引。

#### 2.6.2 DDL操作

具体操作见官方文档：[https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL](https://gitee.com/link?target=https%3A%2F%2Fcwiki.apache.org%2Fconfluence%2Fdisplay%2FHive%2FLanguageManual%2BDDL)

#### 2.6.3 DML操作

官方文档：[https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DML](https://gitee.com/link?target=https%3A%2F%2Fcwiki.apache.org%2Fconfluence%2Fdisplay%2FHive%2FLanguageManual%2BDML)

1. 排序


**Order by **

```
order by会对所给的全部数据进行全局排序,默认只有一个reduce处理数据，谨慎使用
```
**Sort by**

order by会对所给的全部数据进行全局排序,默认只有一个reduce处理数据，谨慎使用	

```
分区内有序，保证局部有序，有多个reduce处理任务
```

**Distribute by**

```
distribute by 控制map结果的分发，会将具有相同字段的map输出分发到一个reduce节点上做处理。类似MR中partition，进行分区，结合sort by使用
```

**Cluster by**

```
当distribute by和sorts by字段相同时，可以使用cluster by方式
```

2. 全局排序的思路

```
(1) 如果在数据处理过程中必须要用到全局排序，则最好使用UDF转换为局部排序。
(2) 先预估数据范围，假设这里数据范围是0-100，然后在每个Map作业中，使用Partitioner对数据进行自定义
		分发，0-10的数据分发到一个Reduce中，10-20的到一个Reduce中，依次类推，然后在每个Reduce作业中进
		行局部排序即可
```

 3. 抽样

```
Hive提供了数据取样（SAMPLING）的功能，能够根据一定的规则进行数据抽样，目前支持数据块抽样，分桶抽样和随机抽样
```

- 数据块抽样

```
/*
(1) tablesample(n percent) 抽取原hive表中10%的数据
(2) tablesample(n M) 指定抽样数据的大小，单位为M
(3) tablesample(n rows) 指定抽样数据的行数，其中n代表每个map任务均取n行数据
*/
select * from t1 tablesample(0.1 percent)
```

- 分桶抽样

**分桶**

```
hive中分桶其实就是根据某一个字段Hash取模，放入指定数据的桶中，比如将表table_1按照ID分成100个桶，其算法是hash(id) % 100，这样，hash(id) % 100 = 0的数据被放到第一个桶中，hash(id) % 100 = 1的记录被放到第二个桶中
```

**抽样**

```
/*
TABLESAMPLE (BUCKET x OUT OF y [ON colname]) 
其中x是要抽样的桶编号，桶编号从1开始，colname表示抽样的列，y表示桶的数量。 
例如：将表随机分成10组，抽取其中的第一个桶的数据 
*/
select * from table_01 tablesample(bucket 1 out of 10 on rand())
```

- 随机抽样

```
/*使用rand()函数进行随机抽样，limit关键字限制抽样返回的数据*/
select * from table_name where col=xxx distribute by rand() sort by rand() limit num
```

4. 窗口范围函数

```hive
/*
(1) OVER 指定分析函数工作的数据窗口大小
(2) CURRENT ROW:当前行
(3) PRECEDING n:往前n行数据
(4) FOLLOWING n:往后n行数据
(5) unbounded preceding 表示该窗口最前面的行
(6) unbounded following 表示该窗口最后面的行

rows between unbounded preceding and current row 和默认的一样
rows between 2 preceding and 1 following表示在当前行的前2行和后1行中计算
rows between current row and unbounded following表示在当前行和到最后行中计算
*/
select country,time,charge,
max(charge) over (partition by country order by time) as normal,
/* rows between unbounded preceding and current row 和默认的一样 */
max(charge) over (partition by country order by time 
                  rows between unbounded preceding and current row) as unb_pre_cur,
/*rows between 2 preceding and 1 following 表示在当前行的前2行和后1行中计算*/
max(charge) over (partition by country order by time 
                  rows between 2 preceding and 1 following) as pre2_fol1,
/*rows between current row and unbounded following 表示在当前行和到最后行中计算*/
max(charge) over (partition by country order by time 
                  rows between current row and unbounded following) as cur_unb_fol 
from temp
```

5. 窗口统计

```HIVE
(1) LAG(col,n,DEFAULT) 用于统计窗口内往上第n行值
(2) LEAD(col,n,DEFAULT) 用于统计窗口内往下第n行值
(3) LAST_VALUE 取分组内排序后，截止到当前行，最后一行的值
(4) FIRST_VALUE 取分组内排序后，截止到当前行，第一行的值
```

6. 多维分区函数

- **GROUPING SETS和GROUPING__ID**

```hive
/*
在一个GROUP BY查询中，根据不同的维度组合进行聚合，等价于将不同维度的GROUP BY结果集进行
UNION ALL,GROUPING__ID，表示结果属于哪一个分组集合*/
select
  month,
  day,
  count(distinct cookieid) as uv,
  GROUPING__ID
from cookie.cookie5
group by month,day
grouping sets (month,day)
order by GROUPING__ID;

/*等价于*/
SELECT month,NULL,COUNT(DISTINCT cookieid) AS uv,1 AS GROUPING__ID FROM cookie5 GROUP BY month
UNION ALL
SELECT NULL,day,COUNT(DISTINCT cookieid) AS uv,2 AS GROUPING__ID FROM cookie5 GROUP BY day
```

- **CUBE**

```hive
/*根据GROUP BY的维度的所有组合进行聚合*/

SELECT  month, day,
COUNT(DISTINCT cookieid) AS uv,
GROUPING_ID
FROM cookie5
GROUP BY month,day
WITH CUBE
ORDER BY GROUPING__ID;

/*等价于*/

SELECT NULL,NULL,COUNT(DISTINCT cookieid) AS uv,0 AS GROUPING__ID FROM cookie5
UNION ALL
SELECT month,NULL,COUNT(DISTINCT cookieid) AS uv,1 AS GROUPING__ID FROM cookie5 GROUP BY month
UNION ALL
SELECT NULL,day,COUNT(DISTINCT cookieid) AS uv,2 AS GROUPING__ID FROM cookie5 GROUP BY day
UNION ALL
SELECT month,day,COUNT(DISTINCT cookieid) AS uv,3 AS GROUPING__ID FROM cookie5 GROUP BY month,day
```

- **ROLLUP**

```hive
/*是CUBE的子集，以最左侧的维度为主，从该维度进行层级聚合,以month维度进行层级聚合*/
SELECT  month, day, COUNT(DISTINCT cookieid) AS uv, GROUPING__ID 
FROM cookie5
GROUP BY month,day WITH ROLLUP  ORDER BY GROUPING__ID;
```

