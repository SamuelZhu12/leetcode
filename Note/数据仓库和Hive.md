# 数据仓库和Hive

## 1 数据仓库

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

## 2 Hive原理及概念

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



## 3 HQL
### 3.1 Hive数据类型
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

### 3.2 DDL操作

#### 2.6.1 表定义

1. 外部表

Hive并非认为完全拥有这份数据，删除该表只会删除metastore中表的元数据，并不会删除HDFS中的数据。多用来存储一些需要长久保存的日志信息等。

2. 内部表

Hive会（或多或少）控制着数据的生命周期，删除一个内部表，Hive表的元数据和HDFS中对应的数据会被一起删除。多用来存储逻辑过程中的中间表，临时表。

3. 分区表

分区表对应一个HDFS上的文件夹，该文件夹是该分区下所有的数据文件。Hive中的分区就是分目录。

4. 临时表（Temporary）

临时表仅对当前会话可见。数据将存储在用户的临时目录中，在会话结束时删除，不支持分区，不支持创建索引。

#### 3.2.2 DDL操作

具体操作见官方文档：[https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL](https://gitee.com/link?target=https%3A%2F%2Fcwiki.apache.org%2Fconfluence%2Fdisplay%2FHive%2FLanguageManual%2BDDL)

#### 3.2.3 DML操作

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

7. 窗口排序

- **row_number**

```hive
-- 按照性别进行分组，并按照成绩进行排序，序号连续且唯一：1、2、3、4、5
select id,name,row_number() over (partition by sex order by grade)
```

- **rank**

```hive
-- 按照性别进行分组，并按照成绩进行排序，成绩相同时序号相同，且不连续：1、1、3、3、5
select id,name,row_number() over (partition by sex order by grade)
```

- **dense_rank**

``` hive
-- 按照性别进行分组，并按照成绩进行排序，成绩相同时序号相同，但序号连续：1、1、2、2、3
select id,name,row_number() over (partition by sex order by grade)
```

###  3.3 Hive函数

#### 3.3.1 Hive自带函数

- Hive自带一些基本函数，例如max/min等，当内置函数无法满足业务需求时，可考虑使用UDF(用户自定义函数)

官方文档：https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF

#### 3.3.2 自定义函数

官方文档：https://cwiki.apache.org/confluence/display/Hive/HivePlugins

1. UDF(User-Defined-Function)

```
 UDF（User-Defined-Function）一进一出
```

```java
@UdfMeta(
        funcName = "func_name",
        funcDesc = "xxxx",
        funcExample = "select func_name() from xxx ",
        funcReturn = @DataType(base = DataTypeEnum.STRING,map ={DataTypeEnum.STRING,DataTypeEnum.INT}),
        funcParam = {
                @Param(paramType = @DataType(array = DataTypeEnum.STRING),paramDesc = "xxx"),
                @Param(paramType = @DataType(base = DataTypeEnum.STRING),paramDesc = "xxx"))
public class func_name extends GenericUDF {
    ObjectInspectorConverters.Converter[] converters;
    @Override
  
  	/*初始化*/
    public ObjectInspector initialize(ObjectInspector[] args) throws UDFArgumentException {
        // 1.检查参数个数
        if (args.length != 1) {
            throw new UDFArgumentException("Param must be x argu");
        }
        // 2.检查参数类型
        converters = new ObjectInspectorConverters.Converter[args.length] ;
        converters[0] = ObjectInspectorConverters.getConverter(args[0] ,
                ObjectInspectorFactory.getStandardMapObjectInspector(
                        PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                        PrimitiveObjectInspectorFactory.writableStringObjectInspector)) ;
        // 3.返回检查器的返回值
        return PrimitiveObjectInspectorFactory.writableStringObjectInspector;
    }

    @Override
  	/*处理数据*/
    public Object evaluate(DeferredObject[] args) throws HiveException {
        // 1.获取参数
        Map<IntWritable,Text> posValue = (Map<IntWritable,Text>)(converters[0].convert(args[0].get()));
        // 2. 调用业务逻辑...
        // 3.返回结果
        return new Text('xxxx');
    }

    @Override
    public String getDisplayString(String[] children) {
        return children[0];
    }
}
```

2. UDAF(User-Defined Aggregation Function)

```
UDAF（User-Defined Aggregation Function） 聚集函数，多进一出,类似于：count/max/min
```

3. UDTF(User-Defined Table-Generating Functions)

```
UDTF（User-Defined Table-Generating Functions）一进多出,如lateral view explore()
```

```java
@UdfMeta(
        funcName = "func_name",
        funcDesc = "xxxx",
        funcExample = "select func_name() from xxx ",
        funcReturn = @DataType(base = DataTypeEnum.STRING,map ={DataTypeEnum.STRING,DataTypeEnum.INT}),
        funcParam = {
                @Param(paramType = @DataType(array = DataTypeEnum.STRING),paramDesc = "xxx"),
                @Param(paramType = @DataType(base = DataTypeEnum.STRING),paramDesc = "xxx"))
public class func_name extends GenericUDTF {

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {

      	 // 1.检查参数个数
        if (args.length != 1) {
            throw new UDFArgumentException("Param must be x argu");
        }
        // 2.检查参数类型
        converters = new ObjectInspectorConverters.Converter[args.length] ;
        converters[0] = ObjectInspectorConverters.getConverter(args[0] ,
                ObjectInspectorFactory.getStandardMapObjectInspector(
                        PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                        PrimitiveObjectInspectorFactory.writableStringObjectInspector)) ;
        // 3.返回检查器的返回值
        ArrayList<String> columns = new ArrayList<>();
        //有两列时，有几列，添加几列
        columns.add("c1");
        columns.add("c2");
        //设置每列的列类型，有几列就设置几列的列类型
        ArrayList<ObjectInspector> columnType = new ArrayList<ObjectInspector>();
        columnType.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        columnType.add(ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableStringObjectInspector,
                PrimitiveObjectInspectorFactory.writableIntObjectInspector));
        return ObjectInspectorFactory.getStandardStructObjectInspector(columns,columnType);
    }

    @Override
    public void process(Object[] args) throws HiveException {

        // 1 解析参数
        List<Text> inputlist=(ArrayList<Text>) args[0];
        String delimiter=args[1].toString();
        BooleanWritable removeEmptyStr=(BooleanWritable)args[3] ;
        // 2.调用业务逻辑
        Map<Text, IntWritable> map = ....
        // 3.输出结果
        ArrayList<Object> objects = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            objects.add(new Text("xxx")));
            objects.add(map);
            forward(objects);
        }
    }
    @Override
    public void close() throws HiveException {
    }

}
```

##  4 Hive压缩

### 4.1 压缩格式

| 压缩格式 | 工具  | 算法    | 文件扩展名 | 是否可切分 |
| -------- | ----- | ------- | ---------- | ---------- |
| DEFAULT  | 无    | DEFAULT | .deflate   | 否         |
| Gzip     | gzip  | DEFAULT | .gz        | 否         |
| bzip2    | bzip2 | bzip2   | .bz2       | 是         |
| LZO      | lzop  | LZO     | .lzo       | 是         |
| Snappy   | 无    | Snappy  | .snappy    | 否         |

### 4.2 编码/解码器

| 压缩格式 | 对应的编码/解码器                                            |
| -------- | ------------------------------------------------------------ |
| DEFLATE  | [org.apache.hadoop.io.compress.DefaultCodec](https://gitee.com/link?target=http%3A%2F%2Forg.apache.hadoop.io.compress.defaultcodec%2F) |
| gzip     | [org.apache.hadoop.io.compress.GzipCodec](https://gitee.com/link?target=http%3A%2F%2Forg.apache.hadoop.io.compress.gzipcodec%2F) |
| bzip2    | [org.apache.hadoop.io.compress.BZip2Codec](https://gitee.com/link?target=http%3A%2F%2Forg.apache.hadoop.io.compress.bzip2codec%2F) |
| LZO      | [com.hadoop.compression.lzo.LzopCodec](https://gitee.com/link?target=http%3A%2F%2Fcom.hadoop.compression.lzo.lzopcodec%2F) |
| Snappy   | [org.apache.hadoop.io.compress.SnappyCodec](https://gitee.com/link?target=http%3A%2F%2Forg.apache.hadoop.io.compress.snappycodec%2F) |

### 4.3 压缩性能的比较

| 压缩算法 | 原始文件大小 | 压缩文件大小 | 压缩速度 | 解压速度 |
| -------- | ------------ | ------------ | -------- | -------- |
| gzip     | 8.3GB        | 1.8GB        | 17.5MB/s | 58MB/s   |
| bzip2    | 8.3GB        | 1.1GB        | 2.4MB/s  | 9.5MB/s  |
| LZO      | 8.3GB        | 2.9GB        | 49.3MB/s | 74.6MB/s |

### 4.4 压缩参数配置

| 参数                                                         | 默认值                                                       | 阶段        | 建议                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------- | -------------------------------------------- |
| [io.compression.codecs](https://gitee.com/link?target=http%3A%2F%2Fio.compression.codecs%2F) | DefaultCodec,GzipCodec,BZip2Codec,Lz4Codec                   | 输入压缩    | Hadoop使用文件扩展名判断是否支持某种编解码器 |
| [mapreduce.map.output.compress](https://gitee.com/link?target=http%3A%2F%2Fmapreduce.map.output.compress%2F) | false                                                        | mapper输出  | 这个参数设为true启用压缩                     |
| [mapreduce.map.output.compress.codec](https://gitee.com/link?target=http%3A%2F%2Fmapreduce.map.output.compress.codec%2F) | [org.apache.hadoop.io.compress.DefaultCodec](https://gitee.com/link?target=http%3A%2F%2Forg.apache.hadoop.io.compress.defaultcodec%2F) | mapper输出  | 使用LZO、LZ4或snappy编解码器在此阶段压缩数据 |
| [mapreduce.output.fileoutputformat.compress](https://gitee.com/link?target=http%3A%2F%2Fmapreduce.output.fileoutputformat.compress%2F) | false                                                        | reducer输出 | 这个参数设为true启用压缩                     |
| [mapreduce.output.fileoutputformat.compress.codec](https://gitee.com/link?target=http%3A%2F%2Fmapreduce.output.fileoutputformat.compress.codec%2F) | [org.apache.hadoop.io.compress](https://gitee.com/link?target=http%3A%2F%2Forg.apache.hadoop.io.compress%2F). DefaultCodec | reducer输出 | 使用标准工具或者编解码器，如gzip和bzip2      |
| [mapreduce.output.fileoutputformat.compress.type](https://gitee.com/link?target=http%3A%2F%2Fmapreduce.output.fileoutputformat.compress.type%2F) | RECORD                                                       | reducer输出 | SequenceFile输出使用的压缩类型：NONE和BLOCK  |

### 4.5 Map输出阶段压缩

```
(1) 开启hive中间传输数据压缩功能  hive.exec.compress.intermediate=true;
(2) 开启mapreduce中map输出压缩功能 set mapreduce.map.output.compress=true;
(3) 设置mapreduce中map输出数据的压缩方式 mapreduce.map.output.compress.codec
		=org.apache.hadoop.io.compress.SnappyCodec
```

### 4.6 Reduce输出阶段压缩

```
(1) 开启hive最终输出数据压缩功 hive.exec.compress.output=true
(2) 开启mapreduce最终输出数据压缩 mapreduce.output.fileoutputformat.compress=true
(3) 设置mapreduce最终数据输出压缩方式mapreduce.output.fileoutputformat.compress.codec 
	  = org.apache.hadoop.io.compress.SnappyCodec
(4) 设置mapreduce最终数据输出压缩为块压缩mapreduce.output.fileoutputformat.compress.type=BLOCK
```

## 5 Hive存储
### 5.1 存储格式
1. Hive支持的存储数的格式主要有：TEXTFILE 、SEQUENCEFILE、ORC、PARQUET
2. TEXTFILE，SEQUENCEFILE的存储格式都是基于行存储的
3. ORC，PARQUET是基于列式存储的

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208011357492.png" alt="img" style="zoom:67%;" />

|                            行存储                            |                            列存储                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| **行存储：**查询满足条件的一整行数据的时候，列存储则需要去每个聚集的字段找到对应的每个列的值，行存储只需要找到其中一个值，其余的值都在相邻地方，所以此时行存储查询的速度更快。 | 因为每个字段的数据聚集存储，在查询只需要少数几个字段的时候，能大大减少读取的数据量；每个字段的数据类型一定是相同的，列式存储可以针对性的设计更好的设计压缩算法 |

- **TextFile**
默认格式，数据不做压缩，磁盘开销大，数据解析开销大。可结合Gzip、Bzip2使用，但使用Gzip这种方式，hive不会对数据进行切分，从而无法对数据进行并行操作

- **Orc**

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208011402374.png" alt="18bb4498d9c7d31588043814e355dcad.png" style="zoom:67%;" />

```
可以看到每个Orc文件由1个或多个stripe组成，每个stripe250MB大小，这个Stripe实际相当于RowGroup概念，
不过大小由4MB->250MB，这样应该能提升顺序读的吞吐率。每个Stripe里有三部分组成，
分别是Index Data，Row Data，Stripe Footer：

   1）Index Data：一个轻量级的index，默认是每隔1W行做一个索引。这里做的索引应该只是记录某行的各字段在Row Data中的offset。
   2）Row Data：存的是具体的数据，先取部分行，然后对这些行按列进行存储。对每个列进行了编码，分成多个Stream来存储。
   3）Stripe Footer：存的是各个Stream的类型，长度等信息。
每个文件有一个File Footer，这里面存的是每个Stripe的行数，每个Column的数据类型信息等；每个文件的尾部是一个PostScript，这里面记录了整个文件的压缩类型以及FileFooter的长度信息等。在读取文件时，会seek到文件尾部读PostScript，从里面解析到File Footer长度，再读FileFooter，从里面解析到各个Stripe信息，再读各个Stripe，即从后往前读。
```

- **Parquet**

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208011404738.png" alt="2340c32c18ad4e251a3756e18a8fa661.png" style="zoom:67%;" />

```
Parquet是面向分析型业务的列式存储格式，由Twitter和Cloudera合作开发，2015年5月从Apache的孵化器里毕业成为Apache顶级项目。
Parquet文件是以二进制方式存储的，所以是不可以直接读取的，文件中包括该文件的数据和元数据，因此Parquet格式文件是自解析的。
通常情况下，在存储Parquet数据的时候会按照Block大小设置行组的大小，由于一般情况下每一个Mapper任务处理数据的最小单位是一个Block，这样可以把每一个行组由一个Mapper任务处理，增大任务执行并行度

上图展示了一个Parquet文件的内容，一个文件中可以存储多个行组，文件的首位都是该文件的Magic Code，用于校验它是否是一个Parquet文件，Footer length记录了文件元数据的大小，通过该值和文件长度可以计算出元数据的偏移量，文件的元数据中包括每一个行组的元数据信息和该文件存储数据的Schema信息。除了文件中每一个行组的元数据，每一页的开始都会存储该页的元数据，在Parquet中，有三种类型的页：数据页、字典页和索引页。数据页用于存储当前行组中该列的值，字典页存储该列值的编码字典，每一个列块中最多包含一个字典页，索引页用来存储当前行组下该列的索引，目前Parquet中还不支持索引页。
```

### 5.2 元数据存储

- **Version表**

​		存储Hive版本信息

- **DB表**

​		DBS：存储数据库对应的hdfs地址

​		DATABASE_PARAMS：存储数据库相关的配置

- **Table表**

​		Table表主要有TBLS、TABLE_PARAMS、TBL_PRIVS

    SDS-- 该表保存文件存储的基本信息，如INPUT_FORMAT、OUTPUT_FORMAT、是否压缩,TBLS表中的SD_ID与该表关联
    SD_PARAMS-该表存储Hive存储的属性信息
    SERDES-该表存储序列化的一些属性、格式信息,比如：行、列分隔符

- **Hive表字段相关的元数据表**

```
COLUMNS_V2--该表存储表对应的字段信息
```

- **分区信息**

```
(1) PARTITIONS--该表存储表分区的基本信息
(2) PARTITION_KEYS--该表存储分区的字段信息
(3) PARTITION_KEY_VALS--该表存储分区字段值
(4) PARTITION_PARAMS--该表存储分区的属性信息
```

## 6 Hive的Join方式

### 6.1 Map Join

```
(1)Map端join是指有两种表，只是一张较小，一张较大（一般大于1万条数据），大表的信息完全可以覆盖小表，往往将较小的表以键值对的形式添加到内存中，然后只扫描大表：对于大表中的每一条记录key/value，在小表中查找是否有相同的key的记录，如果有，则连接后输出即可。
(2)Map端join是数据到达map处理函数之前进行合并的，效率要远远高于Reduce端join，因为Reduce端join是把所有的数据都经过Shuffle，非常消耗资源。所以一般都用Map端join。
```

![image-20220223185224945](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208011417720.png)

**操作流程**

```
1.以键值对形式存储小表信息；
2.在Mapper.setup(Context context)函数里读取该文件并存到Map中；
3.读大表信息，根据外键关联，大表的第二列就是小表的第一列，所以键相同，然后通过大表的第二列的键获取小表中的值，将得到的值将填充回大表中的第二列；
4.将最后得到的值转为字符串；
5.将结果输出（即没有Reduce任务）。
```

**JAVA代码**

```java
public class MyMapJoin {
    public static class MyMapper extends Mapper<LongWritable, Text,Text, NullWritable>{
        private Map myType = new HashMap();//存储小表信息
            @Override
            protected void setup(Context context) throws IOException, InterruptedException {
              	//获取缓存中的小表，返回的是小表的URI
                String fileName = context.getCacheFiles()[0].getPath();
           	final BufferedReader read = new BufferedReader(new FileReader(fileName));//buffer读数据增加效率
            	String str = null;
            	while ((str=read.readLine())!=null){
                String [] sps = str.split(",");//通过逗号进行分割
                myType.put(sps[0],sps[1]);//分割后第一列放到键，第二列放到值
            	}	
            	read.close();
       	    }
       	    @Override
       	    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException 
           //读大表
           String [] goodInfos = value.toString().split(",");
           //根据主外键获取大表分类,外键就是第2列，根据外键获取小表第一列的值
           String type = myType.get(goodInfos[1]).toString();
           //将数据填充回大表数组中的第二列
           goodInfos[1] =type;
           //将数组转为字符串
           Text txt = new Text(Arrays.toString(goodInfos));
           context.write(txt,NullWritable.get());
       }
   }
    public static class MyReduce extends Reducer<Text,NullWritable,Text, NullWritable> {

    }
    //主启动类
    public class Driver{

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException, URISyntaxException {
	Configuration conf = new Configuration();
        //准备一个空任务
        Job job = Job.getInstance(conf);
        //设置该任务的主启动类
        job.setJarByClass(Driver.class);
        //设置任务的输入数据源文件,也就是大表的文件夹
        FileInputFormat.addInputPath(job,new Path("e://xxx"));
        //设置你的mapper任务类
        job.setMapperClass(MyMapper.class);
        //设置mapper任务类的输出数据类型 key和value类型
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(NullWritable.class);
        //放到linux系统下读hdfs文件的路径
        //job.addCacheFile(new URI("hdfs://192.168.xx.xxx:9000/mydemo/xxx.csv"));

        //读本地文件,也就是小表
        job.addFileToClassPath(new Path("e://xxx.csv"));
        //设置任务的输出数据源文件
        FileOutputFormat.setOutputPath(job,new Path("e://gg"));
        //启动任务并执行
        job.waitForCompletion(true);

    }
}
}
```

### 6.2 SMB Join

```
(1) SMB存在的目的主要是为了解决大表与大表间的 Join 问题，分桶其实就是把大表化成了“小表”，然后 Map-Side Join 解决之，
		这是典型的分而治之的思想
(2) SMB分桶字段一般为join的key
(3) 分桶后的文件大小一般为block的大小
```

### 6.3 Reduce Join

```
(1)Map:两张都是大表，所以同一个key对应的字段可能位于不同map中，把两张表放在同一目录下，map函数同时读取两个文件File1和File2，为了区分两张表的key/value数据对，对每条数据打一个tag，例如File1就用 type: File2就用context: 根据文件名进行判断File1和File2,由于有外键关联，输出的时候把相同的外键作为键输出。
(2)Shuffle:根据key的值进行hash,并将key/value按照hash值推送至不同的reduce中，这样确保两个表中相同的key位于同一个reduce中。
(3)Reduce:根据key的值完成join操作，期间通过Tag来识别不同表中的数据
```

![image-20220223185459546](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208011743072.png)

### 6.4 Semi Join

```
(1) SemiJoin就是所谓的半连接是reduce join的一个变种，就是在map端过滤掉一些数据，在网络中只传输参与连接的数据不参与连接的数据不必在网络中进行传输，从而减少了shuffle的网络传输量，使整体效率得到提高，思想和reduce join是一模一样的。

(2) SemiJoin将小表中参与join的key单独抽出来通过DistributedCach分发到相关节点，然后将其取出放到内存中(可以放到HashSet中)，在map阶段扫描连接表，将join key不在内存HashSet中的记录过滤掉，让那些参与join的记录通过shuffle传输到reduce端进行join操作，其他的和reduce join都是一样的
```

## 7 Hive中小文件的处理

### 7.1 小文件产生原因

```
(1) 动态分区插入数据的时候，会产生大量的小文件；
(2) 数据源本身就包含有大量的小文件；
(3) 做增量导入，比如Sqoop数据导入，一些增量insert等；
(4) 分桶表，分桶表通常也会遇到小文件，本质上还是增量导入的问题；
```

### 7.2 小文件危害

```
(1) namenode内存压力，如果namenode内存使用完了，每个文件的元数据信息大概150k
(2) hive 合并小文件会导致任务查询时间长 CombineHiveInputForma
```

### 7.3 小文件解决

- 预防小文件产生

```java
//每个Map最大输入大小(这个值决定了合并后文件的数量)
set mapred.max.split.size=256000000;  
//一个节点上split的至少的大小(这个值决定了多个DataNode上的文件是否需要合并)
set mapred.min.split.size.per.node=100000000;
//一个交换机下split的至少的大小(这个值决定了多个交换机上的文件是否需要合并)  
set mapred.min.split.size.per.rack=100000000;
//执行Map前进行小文件合并
set hive.input.format=org.apache.hadoop.hive.ql.io.CombineHiveInputFormat; 
//设置map端输出进行合并，默认为true
set hive.merge.mapfiles = true
//设置reduce端输出进行合并，默认为false
set hive.merge.mapredfiles = true
//设置合并文件的大小
set hive.merge.size.per.task = 256*1000*1000
//当输出文件的平均大小小于该值时，启动一个独立的MapReduce任务进行文件merge。
set hive.merge.smallfiles.avgsize=16000000
```

- 小文件解决

```java
/* 通过distribute by 进行文件合并，一般基于分区字段*/
insert overwrite table test [partition(hour=...)] select * from test distribute by floor (rand()*5);
```
