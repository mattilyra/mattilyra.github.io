---
layout: post
title: The Correct Way to Connect to Cassandra in Spark Shell
published: true
---

tl;dr

- using the cassandra connector in the spark-shell is fairly straightforward
- setting up the connection in a way that doens't break the existing `sc` is not documented anywhere
- [the correct solution](#solution) is to not call `sc.stop` but provide the cassandra host on startup of the shell

# Spark and Cassandra

Apache Cassandra is a NoSQL distributed database that's been gaining popularity recently. It's also pretty high performance, scoring very high in a (not so) recent [comparison of key-value stores][key_value_stores] (PDF) for different workloads. Among the contenders were HBase, Cassandra, Voldemort, Redis, VoltDB and MySQL, HBase tends to be the winner (by one to two orders of magnitude) when it comes to latency and Cassandra when it comes to throughput - depending on the number of nodes in cluster. A key-value store is nice, but it isn't much use unless you have something doing reads and writes into it. That's where `spark` comes in.

Every data scientist's<sup>[[1]](https://en.wikipedia.org/wiki/Data_science) [[2]](https://www-01.ibm.com/software/data/infosphere/data-scientist/)[[3]](#footnote "I don't like the term either but that's what we seem to have settled for.")</sup> favourite new toy `spark` is a distributed in-memory data processing framework. Cassandra very helpfully comes with a `spark` connector that allows you to pull data into spark as `RDD`s or `DataFrame`s directly from Cassandra.

## Connection Issues

Connecting to a Cassandra host from `spark` isn't all that complicated, just import the connector and tell `SparkConf` where to find the Cassandra host from and you're off to the races. The Cassandra connector [docs](https://github.com/datastax/spark-cassandra-connector/#documentation) cover the basic usage pretty well. Aside from the bazillion different versions of the connector getting everything up and running is fairly straightforward.

Start the spark shell with the necessary Cassandra connector dependencies `bin/spark-shell --packages datastax:spark-cassandra-connector:1.6.0-M2-s_2.10`.

{% highlight scala linenos %}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import com.datastax.spark.connector._

# connect to a local cassandra instance
val conf = new SparkConf(true)
    .set("spark.cassandra.connection.host", "127.0.0.1")
val sc = new SparkContext(conf)
val sqlContext = new SQLContext(sc)

# read in some data as a DataFrame
val df = sqlContext
    .read
    .format("org.apache.spark.sql.cassandra")
    .options(Map("table" -> "fooTable", "keyspace" -> "bar")).load.cache()
{% endhighlight %}

Lovely, you now have a DataFrame that acts just like any other `spark` DataFrame. So far so good. Now let's say you wanted to test something in the `spark-shell` and pull in data from Cassandra. No problem, just do what you did before, except that you need to stop the existing `SparkContext` that is created automagically when the shell starts up, before you can create a new one. This isn't really documented anywhere, except sporadically on [StackOverflow](https://stackoverflow.com/questions/25837436/how-to-load-spark-cassandra-connector-in-the-shell). The accepted answer is actually the wrong way to do this.

{% highlight scala linenos %}
// DO NOT DO THIS
sc.stop // NOOOooo

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import com.datastax.spark.connector._

// connect to a local cassandra instance
val conf = new SparkConf(true)
    .set("spark.cassandra.connection.host", "127.0.0.1")
val sc = new SparkContext(conf)
val sqlContext = new SQLContext(sc)

// read in some data as a DataFrame
val df = sqlContext
    .read
    .format("org.apache.spark.sql.cassandra")
    .options(Map("table" -> "fooTable", "keyspace" -> "bar")).load.cache()
{% endhighlight %}

The `SparkContext` created above will not function like the old `SparkContext` created when the shell started up. This doesn't actually have anything to do the Cassandra connector perse, it's just that the setup for the Cassandra connector brings up this issue. To see the problem consider the following simplified code without the Cassandra connector.

{% highlight scala linenos %}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

sc.stop
val conf = sc.getConf
val sc = new SparkContext(conf)

val rdd = sc.parallelize(Array(0, 1), (1, 10), (2, 15), (3, 7))
val map = Map(0->3, 1->2, 2->1, 3->0)
val BVmap = sc.broadcast(map)
rdd.map( r => BVmap.value(r._1))

{% endhighlight %}

The above doesn't do anything particularly worthwhile, but it illustrates the problem. Because the `SparkContext` was recreated the code will fail in the shell, due to `sc` being not serialisable anymore.

# The Correct Way
<a name="solution">

The solution is extremely simple, but suprisingly difficult to find. Instead of calling `sc.stop` and then recreating the `conf` with the Cassandra host details added, just add the Cassandra host details to the configuration defaults in `$SPARK_HOME/conf/spark-defaults.conf`. Should you not have access to the default conf you can also provide the connection host in the call to `spark-shell`

`bin/spark-shell --packages datastax:spark-cassandra-connector:1.6.0-M2-s_2.10 --conf spark.cassandra.connection.host=127.0.0.1`

This not being included in the official Cassandra connector documentation is bizarre.

# misc

[key_value_stores]: http://vldb.org/pvldb/vol5/p1724_tilmannrabl_vldb2012.pdf "Solving Big Data Challenges for Enterprise Application Performance Management [Rabl et. al 2012] (PDF)"

[3] I don't like the term either but that's what we seem to have settled for.

- [Spark](http://spark.apache.org)
- [Cassandra](http://cassandra.apache.org)
- [Spark Cassandra Connector](https://github.com/datastax/spark-cassandra-connector)
