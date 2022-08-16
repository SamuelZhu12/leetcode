# Spark SQL

## 一. 简介

Spark SQL是spark中一个子模块，主要用于操作结构化数据。它具有以下特点：

- 能够将 SQL 查询与 Spark 程序无缝混合，允许您使用 SQL 或 DataFrame API 对结构化数据进行查询；
- 支持多种开发语言；
- 支持多达上百种的外部数据源，包括 Hive，Avro，Parquet，ORC，JSON 和 JDBC 等；
- 支持 HiveQL 语法以及 Hive SerDes 和 UDF，允许你访问现有的 Hive 仓库；
- 支持标准的 JDBC 和 ODBC 连接；
- 支持优化器，列式存储和代码生成等特性；
- 支持扩展并能保证容错。

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101120720.png)

## 二、DataFrame & DataSet

### 2.1 DataFrame

为了支持结构化数据的处理，Spark SQL 提供了新的数据结构 DataFrame。DataFrame 是一个由具名列组成的数据集。它在概念上等同于关系数据库中的表或 R/Python 语言中的 `data frame`。 由于 Spark SQL 支持多种语言的开发，所以每种语言都定义了 `DataFrame` 的抽象，主要如下：

| 语言   |                   主要抽象                   |
| ------ | :------------------------------------------: |
| Scala  | Dataset[T] & DataFrame (Dataset[Row] 的别名) |
| Java   |                  Dataset[T]                  |
| Python |                  DataFrame                   |
| R      |                  DataFrame                   |

### 2.2 DataFrame 对比 RDDs

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101121646.png)

DataFrame 内部的有明确 Scheme 结构，即列名、列字段类型都是已知的，这带来的好处是可以减少数据读取以及更好地优化执行计划，从而保证查询效率。

**DataFrame 和 RDDs 应该如何选择？**

- 如果你想使用函数式编程而不是 DataFrame API，则使用 RDDs；
- 如果你的数据是非结构化的 (比如流媒体或者字符流)，则使用 RDDs，
- 如果你的数据是结构化的 (如 RDBMS 中的数据) 或者半结构化的 (如日志)，出于性能上的考虑，应优先使用 DataFrame。

Dataset 也是分布式的数据集合，在 Spark 1.6 版本被引入，它集成了 RDD 和 DataFrame 的优点，具备强类型的特点，同时支持 Lambda 函数，但只能在 Scala 和 Java 语言中使用。在 Spark 2.0 后，为了方便开发者，Spark 将 DataFrame 和 Dataset 的 API 融合到一起，提供了结构化的 API(Structured API)，即用户可以通过一套标准的 API 就能完成对两者的操作。

> 这里注意一下：DataFrame 被标记为 Untyped API，而 DataSet 被标记为 Typed API，后文会对两者做出解释。

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101123553.png)

### 2.4 静态类型与运行时类型安全

静态类型 (Static-typing) 与运行时类型安全 (runtime type-safety) 主要表现如下:

在实际使用中，如果你用的是 Spark SQL 的查询语句，则直到运行时你才会发现有语法错误，而如果你用的是 DataFrame 和 Dataset，则在编译时就可以发现错误 (这节省了开发时间和整体代价)。DataFrame 和 Dataset 主要区别在于：

在 DataFrame 中，当你调用了 API 之外的函数，编译器就会报错，但如果你使用了一个不存在的字段名字，编译器依然无法发现。而 Dataset 的 API 都是用 Lambda 函数和 JVM 类型对象表示的，所有不匹配的类型参数在编译时就会被发现。

以上这些最终都被解释成关于类型安全图谱，对应开发中的语法和分析错误。在图谱中，Dataset 最严格，但对于开发者来说效率最高。

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101127473.png" alt="img" style="zoom:67%;" />

​	上面的描述可能并没有那么直观，下面的给出一个 IDEA 中代码编译的示例：

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101129092.png)

这里一个可能的疑惑是 DataFrame 明明是有确定的 Scheme 结构 (即列名、列字段类型都是已知的)，但是为什么还是无法对列名进行推断和错误判断，这是因为 DataFrame 是 Untyped 的。

### 2.5 Untyped & Typed

在上面我们介绍过 DataFrame API 被标记为 `Untyped API`，而 DataSet API 被标记为 `Typed API`。DataFrame 的 `Untyped` 是相对于语言或 API 层面而言，它确实有明确的 Scheme 结构，即列名，列类型都是确定的，但这些信息完全由 Spark 来维护，Spark 只会在运行时检查这些类型和指定类型是否一致。这也就是为什么在 Spark 2.0 之后，官方推荐把 DataFrame 看做是 `DatSet[Row]`，Row 是 Spark 中定义的一个 `trait`，其子类中封装了列字段的信息。

相对而言，DataSet 是 `Typed` 的，即强类型。如下面代码，DataSet 的类型由 Case Class(Scala) 或者 Java Bean(Java) 来明确指定的，在这里即每一行数据代表一个 `Person`，这些信息由 JVM 来保证正确性，所以字段名错误和类型错误在编译的时候就会被 IDE 所发现。

```scala
case class Person(name: String, age: Long)
val dataSet: Dataset[Person] = spark.read.json("people.json").as[Person]
```

## 三、DataFrame & DataSet & RDDs 总结

这里对三者做一下简单的总结：

- RDDs 适合非结构化数据的处理，而 DataFrame & DataSet 更适合结构化数据和半结构化的处理；
- DataFrame & DataSet 可以通过统一的 Structured API 进行访问，而 RDDs 则更适合函数式编程的场景；
- 相比于 DataFrame 而言，DataSet 是强类型的 (Typed)，有着更为严格的静态类型检查；
- DataSets、DataFrames、SQL 的底层都依赖了 RDDs API，并对外提供结构化的访问接口。

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101131848.png" alt="img" style="zoom:67%;" />

## 四、Spark SQL的运行原理

DataFrame、DataSet 和 Spark SQL 的实际执行流程都是相同的：

1. 进行 DataFrame/Dataset/SQL 编程；
2. 如果是有效的代码，即代码没有编译错误，Spark 会将其转换为一个逻辑计划；
3. Spark 将此逻辑计划转换为物理计划，同时进行代码优化；
4. Spark 然后在集群上执行这个物理计划 (基于 RDD 操作) 。

### 4.1 逻辑计划(Logical Plan)

执行的第一个阶段是将用户代码转换成一个逻辑计划。它首先将用户代码转换成 `unresolved logical plan`(未解决的逻辑计划)，之所以这个计划是未解决的，是因为尽管您的代码在语法上是正确的，但是它引用的表或列可能不存在。 Spark 使用 `analyzer`(分析器) 基于 `catalog`(存储的所有表和 `DataFrames` 的信息) 进行解析。解析失败则拒绝执行，解析成功则将结果传给 `Catalyst` 优化器 (`Catalyst Optimizer`)，优化器是一组规则的集合，用于优化逻辑计划，通过谓词下推等方式进行优化，最终输出优化后的逻辑执行计划。

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101132498.png)

### 4.2 物理计划(Physical Plan)

得到优化后的逻辑计划后，Spark 就开始了物理计划过程。 它通过生成不同的物理执行策略，并通过成本模型来比较它们，从而选择一个最优的物理计划在集群上面执行的。物理规划的输出结果是一系列的 RDDs 和转换关系 (transformations)。

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101132324.png)

### 4.3 执行

在选择一个物理计划后，Spark 运行其 RDDs 代码，并在运行时执行进一步的优化，生成本地 Java 字节码，最后将运行结果返回给用户。



# Structured API基本使用

## 一、创建DataFrame和Dataset

### 1.1 创建DataFrame

Spark 中所有功能的入口点是 `SparkSession`，可以使用 `SparkSession.builder()` 创建。创建后应用程序就可以从现有 RDD，Hive 表或 Spark 数据源创建 DataFrame。示例如下：

```scala
val spark = SparkSession.builder().appName("Spark-SQL").master("local[2]").getOrCreate()
val df = spark.read.json("/usr/file/json/emp.json")
df.show()

// 建议在进行 spark SQL 编程前导入下面的隐式转换，因为 DataFrames 和 dataSets 中很多操作都依赖了隐式转换
import spark.implicits._
```

可以使用 `spark-shell` 进行测试，需要注意的是 `spark-shell` 启动后会自动创建一个名为 `spark` 的 `SparkSession`，在命令行中可以直接引用即可：

![img](https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101159915.png)

### 1.2 创建Dataset

Spark 支持由内部数据集和外部数据集来创建 DataSet，其创建方式分别如下：

####  1. 由外部数据集创建

```scala
// 1.需要导入隐式转换
import spark.implicits._

// 2.创建 case class,等价于 Java Bean
case class Emp(ename: String, comm: Double, deptno: Long, empno: Long, 
               hiredate: String, job: String, mgr: Long, sal: Double)

// 3.由外部数据集创建 Datasets
val ds = spark.read.json("/usr/file/emp.json").as[Emp]
ds.show()
```

#### 2. 由内部数据集创建

```scala
// 1.需要导入隐式转换
import spark.implicits._

// 2.创建 case class,等价于 Java Bean
case class Emp(ename: String, comm: Double, deptno: Long, empno: Long, 
               hiredate: String, job: String, mgr: Long, sal: Double)

// 3.由内部数据集创建 Datasets
val caseClassDS = Seq(Emp("ALLEN", 300.0, 30, 7499, "1981-02-20 00:00:00", "SALESMAN", 7698, 1600.0),
                      Emp("JONES", 300.0, 30, 7499, "1981-02-20 00:00:00", "SALESMAN", 7698, 1600.0))
                    .toDS()
caseClassDS.show()
```

### 1.3 由RDD创建DataFrame

Spark 支持两种方式把 RDD 转换为 DataFrame，分别是使用反射推断和指定 Schema 转换：

#### 1. 使用反射推断

```scala
// 1.导入隐式转换
import spark.implicits._

// 2.创建部门类
case class Dept(deptno: Long, dname: String, loc: String)

// 3.创建 RDD 并转换为 dataSet
val rddToDS = spark.sparkContext
  .textFile("/usr/file/dept.txt")
  .map(_.split("\t"))
  .map(line => Dept(line(0).trim.toLong, line(1), line(2)))
  .toDS()  // 如果调用 toDF() 则转换为 dataFrame 
```

#### 2. 以编程方式指定Schema

```scala
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._


// 1.定义每个列的列类型
val fields = Array(StructField("deptno", LongType, nullable = true),
                   StructField("dname", StringType, nullable = true),
                   StructField("loc", StringType, nullable = true))

// 2.创建 schema
val schema = StructType(fields)

// 3.创建 RDD
val deptRDD = spark.sparkContext.textFile("/usr/file/dept.txt")
val rowRDD = deptRDD.map(_.split("\t")).map(line => Row(line(0).toLong, line(1), line(2)))


// 4.将 RDD 转换为 dataFrame
val deptDF = spark.createDataFrame(rowRDD, schema)
deptDF.show()<br/>
```

### 1.4  DataFrames与Datasets互相转换

Spark 提供了非常简单的转换方法用于 DataFrame 与 Dataset 间的互相转换，示例如下：

```shell
# DataFrames转Datasets
scala> df.as[Emp]
res1: org.apache.spark.sql.Dataset[Emp] = [COMM: double, DEPTNO: bigint ... 6 more fields]

# Datasets转DataFrames
scala> ds.toDF()
res2: org.apache.spark.sql.DataFrame = [COMM: double, DEPTNO: bigint ... 6 more fields]
```

## 二、Columns列操作

### 2.1 引用列

Spark 支持多种方法来构造和引用列，最简单的是使用 `col() ` 或 `column() ` 函数。

```scala
col("colName")
column("colName")

// 对于 Scala 语言而言，还可以使用$"myColumn"和'myColumn 这两种语法糖进行引用。
df.select($"ename", $"job").show()
df.select('ename, 'job).show()
```

### 2.2 新增列

```scala
// 基于已有列值新增列
df.withColumn("upSal",$"sal"+1000)
// 基于固定值新增列
df.withColumn("intCol",lit(1000))
```

### 2.3 删除列

```scala
// 支持删除多个列
df.drop("comm","job").show()
```

### 2.4 重命名列

```scala
df.withColumnRenamed("comm", "common").show()
```

需要说明的是新增，删除，重命名列都会产生新的 DataFrame，原来的 DataFrame 不会被改变。

<br/>

## 三、使用Structured API进行基本查询

```scala
// 1.查询员工姓名及工作
df.select($"ename", $"job").show()

// 2.filter 查询工资大于 2000 的员工信息
df.filter($"sal" > 2000).show()

// 3.orderBy 按照部门编号降序，工资升序进行查询
df.orderBy(desc("deptno"), asc("sal")).show()

// 4.limit 查询工资最高的 3 名员工的信息
df.orderBy(desc("sal")).limit(3).show()

// 5.distinct 查询所有部门编号
df.select("deptno").distinct().show()

// 6.groupBy 分组统计部门人数
df.groupBy("deptno").count().show()
```

<br/>

## 四、使用Spark SQL进行基本查询

### 4.1 Spark  SQL基本使用

```scala
// 1.首先需要将 DataFrame 注册为临时视图
df.createOrReplaceTempView("emp")

// 2.查询员工姓名及工作
spark.sql("SELECT ename,job FROM emp").show()

// 3.查询工资大于 2000 的员工信息
spark.sql("SELECT * FROM emp where sal > 2000").show()

// 4.orderBy 按照部门编号降序，工资升序进行查询
spark.sql("SELECT * FROM emp ORDER BY deptno DESC,sal ASC").show()

// 5.limit  查询工资最高的 3 名员工的信息
spark.sql("SELECT * FROM emp ORDER BY sal DESC LIMIT 3").show()

// 6.distinct 查询所有部门编号
spark.sql("SELECT DISTINCT(deptno) FROM emp").show()

// 7.分组统计部门人数
spark.sql("SELECT deptno,count(ename) FROM emp group by deptno").show()
```

### 4.2 全局临时视图

上面使用 `createOrReplaceTempView` 创建的是会话临时视图，它的生命周期仅限于会话范围，会随会话的结束而结束。

你也可以使用 `createGlobalTempView` 创建全局临时视图，全局临时视图可以在所有会话之间共享，并直到整个 Spark 应用程序终止后才会消失。全局临时视图被定义在内置的 `global_temp` 数据库下，需要使用限定名称进行引用，如 `SELECT * FROM global_temp.view1`。

```scala
// 注册为全局临时视图
df.createGlobalTempView("gemp")

// 使用限定名称进行引用
spark.sql("SELECT ename,job FROM global_temp.gemp").show()
```






# Spark SQL 外部数据源


## 一、简介

### 1.1 多数据源支持

Spark 支持以下六个核心数据源，同时 Spark 社区还提供了多达上百种数据源的读取方式，能够满足绝大部分使用场景。

- CSV
- JSON
- Parquet
- ORC
- JDBC/ODBC connections
- Plain-text files

> 注：以下所有测试文件均可从本仓库的[resources](https://github.com/heibaiying/BigData-Notes/tree/master/resources) 目录进行下载

### 1.2 读数据格式

所有读取 API 遵循以下调用格式：

```scala
// 格式
DataFrameReader.format(...).option("key", "value").schema(...).load()

// 示例
spark.read.format("csv")
.option("mode", "FAILFAST")          // 读取模式
.option("inferSchema", "true")       // 是否自动推断 schema
.option("path", "path/to/file(s)")   // 文件路径
.schema(someSchema)                  // 使用预定义的 schema      
.load()
```

读取模式有以下三种可选项：

| 读模式          | 描述                                                         |
| --------------- | ------------------------------------------------------------ |
| `permissive`    | 当遇到损坏的记录时，将其所有字段设置为 null，并将所有损坏的记录放在名为 _corruption t_record 的字符串列中 |
| `dropMalformed` | 删除格式不正确的行                                           |
| `failFast`      | 遇到格式不正确的数据时立即失败                               |

### 1.3 写数据格式

```scala
// 格式
DataFrameWriter.format(...).option(...).partitionBy(...).bucketBy(...).sortBy(...).save()

//示例
dataframe.write.format("csv")
.option("mode", "OVERWRITE")         //写模式
.option("dateFormat", "yyyy-MM-dd")  //日期格式
.option("path", "path/to/file(s)")
.save()
```

写数据模式有以下四种可选项：

| Scala/Java               | 描述                                                         |
| :----------------------- | :----------------------------------------------------------- |
| `SaveMode.ErrorIfExists` | 如果给定的路径已经存在文件，则抛出异常，这是写数据默认的模式 |
| `SaveMode.Append`        | 数据以追加的方式写入                                         |
| `SaveMode.Overwrite`     | 数据以覆盖的方式写入                                         |
| `SaveMode.Ignore`        | 如果给定的路径已经存在文件，则不做任何操作 |

## 二、CSV

CSV 是一种常见的文本文件格式，其中每一行表示一条记录，记录中的每个字段用逗号分隔。

### 2.1 读取CSV文件

自动推断类型读取读取示例：

```scala
spark.read.format("csv")
.option("header", "false")        // 文件中的第一行是否为列的名称
.option("mode", "FAILFAST")      // 是否快速失败
.option("inferSchema", "true")   // 是否自动推断 schema
.load("/usr/file/csv/dept.csv")
.show()
```

使用预定义类型：

```scala
import org.apache.spark.sql.types.{StructField, StructType, StringType,LongType}
//预定义数据格式
val myManualSchema = new StructType(Array(
    StructField("deptno", LongType, nullable = false),
    StructField("dname", StringType,nullable = true),
    StructField("loc", StringType,nullable = true)
))
spark.read.format("csv")
.option("mode", "FAILFAST")
.schema(myManualSchema)
.load("/usr/file/csv/dept.csv")
.show()
```

### 2.2 写入CSV文件

```scala
df.write.format("csv").mode("overwrite").save("/tmp/csv/dept2")
```

也可以指定具体的分隔符：

```scala
df.write.format("csv").mode("overwrite").option("sep", "\t").save("/tmp/csv/dept2")
```

### 2.3 可选配置

为节省主文篇幅，所有读写配置项见文末 9.1 小节。

<br/>

## 三、JSON

### 3.1 读取JSON文件

```json
spark.read.format("json").option("mode", "FAILFAST").load("/usr/file/json/dept.json").show(5)
```

需要注意的是：默认不支持一条数据记录跨越多行 (如下)，可以通过配置 `multiLine` 为 `true` 来进行更改，其默认值为 `false`。

```json
// 默认支持单行
{"DEPTNO": 10,"DNAME": "ACCOUNTING","LOC": "NEW YORK"}

//默认不支持多行
{
  "DEPTNO": 10,
  "DNAME": "ACCOUNTING",
  "LOC": "NEW YORK"
}
```

### 3.2 写入JSON文件

```scala
df.write.format("json").mode("overwrite").save("/tmp/spark/json/dept")
```

### 3.3 可选配置

为节省主文篇幅，所有读写配置项见文末 9.2 小节。

## 四、Parquet

 Parquet 是一个开源的面向列的数据存储，它提供了多种存储优化，允许读取单独的列非整个文件，这不仅节省了存储空间而且提升了读取效率，它是 Spark 是默认的文件格式。

### 4.1 读取Parquet文件

```scala
spark.read.format("parquet").load("/usr/file/parquet/dept.parquet").show(5)
```

### 2.2 写入Parquet文件

```scala
df.write.format("parquet").mode("overwrite").save("/tmp/spark/parquet/dept")
```

### 2.3 可选配置

Parquet 文件有着自己的存储规则，因此其可选配置项比较少，常用的有如下两个：

| 读写操作 | 配置项               | 可选值                                               | 默认值                                       | 描述                                                         |
| -------- | -------------------- | ---------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| Write    | compression or codec | None,uncompressed,bzip2,deflate, gzip,lz4, or snappy | None                                         | 压缩文件格式                                                 |
| Read     | mergeSchema          | true, false                                          | 取决于配置项 `spark.sql.parquet.mergeSchema` | 当为真时，Parquet 数据源将所有数据文件收集的 Schema 合并在一起，否则将从摘要文件中选择 Schema，如果没有可用的摘要文件，则从随机数据文件中选择 Schema。 |

> 更多可选配置可以参阅官方文档：https://spark.apache.org/docs/latest/sql-data-sources-parquet.html

## 五、ORC 

ORC 是一种自描述的、类型感知的列文件格式，它针对大型数据的读写进行了优化，也是大数据中常用的文件格式。

### 5.1 读取ORC文件 

```scala
spark.read.format("orc").load("/usr/file/orc/dept.orc").show(5)
```

### 4.2 写入ORC文件

```scala
csvFile.write.format("orc").mode("overwrite").save("/tmp/spark/orc/dept")
```

## 六、SQL Databases 

Spark 同样支持与传统的关系型数据库进行数据读写。但是 Spark 程序默认是没有提供数据库驱动的，所以在使用前需要将对应的数据库驱动上传到安装目录下的 `jars` 目录中。下面示例使用的是 Mysql 数据库，使用前需要将对应的 `mysql-connector-java-x.x.x.jar` 上传到 `jars` 目录下。

### 6.1 读取数据

读取全表数据示例如下，这里的 `help_keyword` 是 mysql 内置的字典表，只有 `help_keyword_id` 和 `name` 两个字段。

```scala
spark.read
.format("jdbc")
.option("driver", "com.mysql.jdbc.Driver")            //驱动
.option("url", "jdbc:mysql://127.0.0.1:3306/mysql")   //数据库地址
.option("dbtable", "help_keyword")                    //表名
.option("user", "root").option("password","root").load().show(10)
```

从查询结果读取数据：

```scala
val pushDownQuery = """(SELECT * FROM help_keyword WHERE help_keyword_id <20) AS help_keywords"""
spark.read.format("jdbc")
.option("url", "jdbc:mysql://127.0.0.1:3306/mysql")
.option("driver", "com.mysql.jdbc.Driver")
.option("user", "root").option("password", "root")
.option("dbtable", pushDownQuery)
.load().show()

//输出
+---------------+-----------+
|help_keyword_id|       name|
+---------------+-----------+
|              0|         <>|
|              1|     ACTION|
|              2|        ADD|
|              3|AES_DECRYPT|
|              4|AES_ENCRYPT|
|              5|      AFTER|
|              6|    AGAINST|
|              7|  AGGREGATE|
|              8|  ALGORITHM|
|              9|        ALL|
|             10|      ALTER|
|             11|    ANALYSE|
|             12|    ANALYZE|
|             13|        AND|
|             14|    ARCHIVE|
|             15|       AREA|
|             16|         AS|
|             17|   ASBINARY|
|             18|        ASC|
|             19|     ASTEXT|
+---------------+-----------+
```

也可以使用如下的写法进行数据的过滤：

```scala
val props = new java.util.Properties
props.setProperty("driver", "com.mysql.jdbc.Driver")
props.setProperty("user", "root")
props.setProperty("password", "root")
val predicates = Array("help_keyword_id < 10  OR name = 'WHEN'")   //指定数据过滤条件
spark.read.jdbc("jdbc:mysql://127.0.0.1:3306/mysql", "help_keyword", predicates, props).show() 

//输出：
+---------------+-----------+
|help_keyword_id|       name|
+---------------+-----------+
|              0|         <>|
|              1|     ACTION|
|              2|        ADD|
|              3|AES_DECRYPT|
|              4|AES_ENCRYPT|
|              5|      AFTER|
|              6|    AGAINST|
|              7|  AGGREGATE|
|              8|  ALGORITHM|
|              9|        ALL|
|            604|       WHEN|
+---------------+-----------+
```

可以使用 `numPartitions` 指定读取数据的并行度：

```scala
option("numPartitions", 10)
```

在这里，除了可以指定分区外，还可以设置上界和下界，任何小于下界的值都会被分配在第一个分区中，任何大于上界的值都会被分配在最后一个分区中。

```scala
val colName = "help_keyword_id"   //用于判断上下界的列
val lowerBound = 300L    //下界
val upperBound = 500L    //上界
val numPartitions = 10   //分区综述
val jdbcDf = spark.read.jdbc("jdbc:mysql://127.0.0.1:3306/mysql","help_keyword",
                             colName,lowerBound,upperBound,numPartitions,props)
```

想要验证分区内容，可以使用 `mapPartitionsWithIndex` 这个算子，代码如下：

```scala
jdbcDf.rdd.mapPartitionsWithIndex((index, iterator) => {
    val buffer = new ListBuffer[String]
    while (iterator.hasNext) {
        buffer.append(index + "分区:" + iterator.next())
    }
    buffer.toIterator
}).foreach(println)
```

执行结果如下：`help_keyword` 这张表只有 600 条左右的数据，本来数据应该均匀分布在 10 个分区，但是 0 分区里面却有 319 条数据，这是因为设置了下限，所有小于 300 的数据都会被限制在第一个分区，即 0 分区。同理所有大于 500 的数据被分配在 9 分区，即最后一个分区。

<div align="center"> <img src="../pictures/spark-mysql-分区上下限.png"/> </div>

### 6.2 写入数据

```scala
val df = spark.read.format("json").load("/usr/file/json/emp.json")
df.write
.format("jdbc")
.option("url", "jdbc:mysql://127.0.0.1:3306/mysql")
.option("user", "root").option("password", "root")
.option("dbtable", "emp")
.save()
```

<br/>

## 七、Text 

Text 文件在读写性能方面并没有任何优势，且不能表达明确的数据结构，所以其使用的比较少，读写操作如下：

### 7.1 读取Text数据

```scala
spark.read.textFile("/usr/file/txt/dept.txt").show()
```

### 7.2 写入Text数据

```scala
df.write.text("/tmp/spark/txt/dept")
```

## 八、数据读写高级特性

### 8.1 并行读

多个 Executors 不能同时读取同一个文件，但它们可以同时读取不同的文件。这意味着当您从一个包含多个文件的文件夹中读取数据时，这些文件中的每一个都将成为 DataFrame 中的一个分区，并由可用的 Executors 并行读取。

### 8.2 并行写

写入的文件或数据的数量取决于写入数据时 DataFrame 拥有的分区数量。默认情况下，每个数据分区写一个文件。

### 8.3 分区写入

分区和分桶这两个概念和 Hive 中分区表和分桶表是一致的。都是将数据按照一定规则进行拆分存储。需要注意的是 `partitionBy` 指定的分区和 RDD 中分区不是一个概念：这里的**分区表现为输出目录的子目录**，数据分别存储在对应的子目录中。

```scala
val df = spark.read.format("json").load("/usr/file/json/emp.json")
df.write.mode("overwrite").partitionBy("deptno").save("/tmp/spark/partitions")
```

输出结果如下：可以看到输出被按照部门编号分为三个子目录，子目录中才是对应的输出文件。

<img src="https://typora-1308702321.cos.ap-guangzhou.myqcloud.com/typora/202208101210491.png" alt="img"  />

### 8.3 分桶写入

分桶写入就是将数据按照指定的列和桶数进行散列，目前分桶写入只支持保存为表，实际上这就是 Hive 的分桶表。

```scala
val numberBuckets = 10
val columnToBucketBy = "empno"
df.write.format("parquet").mode("overwrite")
.bucketBy(numberBuckets, columnToBucketBy).saveAsTable("bucketedFiles")
```

### 8.5 文件大小管理

如果写入产生小文件数量过多，这时会产生大量的元数据开销。Spark 和 HDFS 一样，都不能很好的处理这个问题，这被称为“small file problem”。同时数据文件也不能过大，否则在查询时会有不必要的性能开销，因此要把文件大小控制在一个合理的范围内。

在上文我们已经介绍过可以通过分区数量来控制生成文件的数量，从而间接控制文件大小。Spark 2.2 引入了一种新的方法，以更自动化的方式控制文件大小，这就是 `maxRecordsPerFile` 参数，它允许你通过控制写入文件的记录数来控制文件大小。

```scala
 // Spark 将确保文件最多包含 5000 条记录
df.write.option(“maxRecordsPerFile”, 5000)
```

## 九、可选配置附录

### 9.1 CSV读写可选配置

| 读\写操作 | 配置项                      | 可选值                                                       | 默认值                     | 描述                                                         |
| --------- | --------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ |
| Both      | seq                         | 任意字符                                                     | `,`(逗号)                  | 分隔符                                                       |
| Both      | header                      | true, false                                                  | false                      | 文件中的第一行是否为列的名称。                               |
| Read      | escape                      | 任意字符                                                     | \                          | 转义字符                                                     |
| Read      | inferSchema                 | true, false                                                  | false                      | 是否自动推断列类型                                           |
| Read      | ignoreLeadingWhiteSpace     | true, false                                                  | false                      | 是否跳过值前面的空格                                         |
| Both      | ignoreTrailingWhiteSpace    | true, false                                                  | false                      | 是否跳过值后面的空格                                         |
| Both      | nullValue                   | 任意字符                                                     | “”                         | 声明文件中哪个字符表示空值                                   |
| Both      | nanValue                    | 任意字符                                                     | NaN                        | 声明哪个值表示 NaN 或者缺省值                                  |
| Both      | positiveInf                 | 任意字符                                                     | Inf                        | 正无穷                                                       |
| Both      | negativeInf                 | 任意字符                                                     | -Inf                       | 负无穷                                                       |
| Both      | compression or codec        | None,<br/>uncompressed,<br/>bzip2, deflate,<br/>gzip, lz4, or<br/>snappy | none                       | 文件压缩格式                                                 |
| Both      | dateFormat                  | 任何能转换为 Java 的 <br/>SimpleDataFormat 的字符串            | yyyy-MM-dd                 | 日期格式                                                     |
| Both      | timestampFormat             | 任何能转换为 Java 的 <br/>SimpleDataFormat 的字符串            | yyyy-MMdd’T’HH:mm:ss.SSSZZ | 时间戳格式                                                   |
| Read      | maxColumns                  | 任意整数                                                     | 20480                      | 声明文件中的最大列数                                         |
| Read      | maxCharsPerColumn           | 任意整数                                                     | 1000000                    | 声明一个列中的最大字符数。                                   |
| Read      | escapeQuotes                | true, false                                                  | true                       | 是否应该转义行中的引号。                                     |
| Read      | maxMalformedLogPerPartition | 任意整数                                                     | 10                         | 声明每个分区中最多允许多少条格式错误的数据，超过这个值后格式错误的数据将不会被读取 |
| Write     | quoteAll                    | true, false                                                  | false                      | 指定是否应该将所有值都括在引号中，而不只是转义具有引号字符的值。 |
| Read      | multiLine                   | true, false                                                  | false                      | 是否允许每条完整记录跨域多行                                 |

### 9.2 JSON读写可选配置

| 读\写操作 | 配置项                             | 可选值                                                       | 默认值                           |
| --------- | ---------------------------------- | ------------------------------------------------------------ | -------------------------------- |
| Both      | compression or codec               | None,<br/>uncompressed,<br/>bzip2, deflate,<br/>gzip, lz4, or<br/>snappy | none                             |
| Both      | dateFormat                         | 任何能转换为 Java 的 SimpleDataFormat 的字符串                 | yyyy-MM-dd                       |
| Both      | timestampFormat                    | 任何能转换为 Java 的 SimpleDataFormat 的字符串                 | yyyy-MMdd’T’HH:mm:ss.SSSZZ       |
| Read      | primitiveAsString                  | true, false                                                  | false                            |
| Read      | allowComments                      | true, false                                                  | false                            |
| Read      | allowUnquotedFieldNames            | true, false                                                  | false                            |
| Read      | allowSingleQuotes                  | true, false                                                  | true                             |
| Read      | allowNumericLeadingZeros           | true, false                                                  | false                            |
| Read      | allowBackslashEscapingAnyCharacter | true, false                                                  | false                            |
| Read      | columnNameOfCorruptRecord          | true, false                                                  | Value of spark.sql.column&NameOf |
| Read      | multiLine                          | true, false                                                  | false                            |

### 9.3 数据库读写可选配置

| 属性名称                                   | 含义                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| url                                        | 数据库地址                                                   |
| dbtable                                    | 表名称                                                       |
| driver                                     | 数据库驱动                                                   |
| partitionColumn,<br/>lowerBound, upperBoun | 分区总数，上界，下界                                         |
| numPartitions                              | 可用于表读写并行性的最大分区数。如果要写的分区数量超过这个限制，那么可以调用 coalesce(numpartition) 重置分区数。 |
| fetchsize                                  | 每次往返要获取多少行数据。此选项仅适用于读取数据。           |
| batchsize                                  | 每次往返插入多少行数据，这个选项只适用于写入数据。默认值是 1000。 |
| isolationLevel                             | 事务隔离级别：可以是 NONE，READ_COMMITTED, READ_UNCOMMITTED，REPEATABLE_READ 或 SERIALIZABLE，即标准事务隔离级别。<br/>默认值是 READ_UNCOMMITTED。这个选项只适用于数据读取。 |
| createTableOptions                         | 写入数据时自定义创建表的相关配置                             |
| createTableColumnTypes                     | 写入数据时自定义创建列的列类型                               |

> 数据库读写更多配置可以参阅官方文档：https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html










## 参考资料

1. [Spark SQL, DataFrames and Datasets Guide > Getting Started](https://spark.apache.org/docs/latest/sql-getting-started.html)
2. Matei Zaharia, Bill Chambers . Spark: The Definitive Guide[M] . 2018-02 
3. https://spark.apache.org/docs/latest/sql-data-sources.html
