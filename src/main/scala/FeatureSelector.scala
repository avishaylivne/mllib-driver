package core.learn.trainers

import java.io._

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.regression._
import org.apache.spark.mllib.feature._
import org.slf4j.LoggerFactory

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

import scala.io.Source

/**
  * Created by avishaylivne on 1/17/16.
  */

object FeatureSelector {
  val usage = """
    Usage: featureSelector inputPath numFeatures outputPath selection_method_1 [selection_method_2] ... [selection_method_N]
        inputPath: Path to input file.
        numFeatures: Number of features to select.
        outputPath: Path to output file.
        selection_method_i: the name of the selection methods to use
|   """

  lazy protected val logger = LoggerFactory.getLogger("MLlib Feature Selector")

  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      println(usage)
      return
    }
    val conf = new SparkConf().setAppName(s"MLlib feature selector")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("WARN")
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val inputPath = args(0)
    val numFeatures = args(1).toInt
    val outputPath = args(2)
    val modelNames = args.slice(3, args.length)
    val inputDataFrame = sqlContext.read.format("libsvm").load(inputPath)
    val inputRDD = MLUtils.loadLibSVMFile(sc, inputPath)

    val pw = new PrintWriter(new File(outputPath))
    for (model <- modelNames) {
      val selectedFeatures: Array[Int] = model match {
        case "lasso" => lassoSelector(inputDataFrame, numFeatures)
        case "mim" => infoCriterionFeatures(inputRDD, "mim", numFeatures)
        case "mifs" => infoCriterionFeatures(inputRDD, "mifs", numFeatures)
        case "jmi" => infoCriterionFeatures(inputRDD, "jmi", numFeatures)
        case "mrmr" => infoCriterionFeatures(inputRDD, "mrmr", numFeatures)
        case "icap" => infoCriterionFeatures(inputRDD, "icap", numFeatures)
        case "if" => infoCriterionFeatures(inputRDD, "if", numFeatures)
      }
      pw.write(model + "\t")
      pw.println(selectedFeatures.mkString("\t"))
    }
    pw.close
  }

  def infoCriterionFeatures(data: RDD[LabeledPoint], method: String, numFeatures: Int, numPartitions: Int = 2) = {
    InfoThSelector.train(new InfoThCriterionFactory("mim"), data, numFeatures, numPartitions).selectedFeatures
  }

  def lassoSelector(data: DataFrame, numFeatures: Int): Array[Int] = {
    val regCoeff = 100
    val m = new LogisticRegression()
    m.setElasticNetParam(1.0)

    def linearDecrease(regCoeff: Double, delta: Double, lastCoeffs: Option[Vector]): Vector = {
      m.setRegParam(regCoeff)
      val coeffs = m.fit(data).coefficients
      logger.error(s"LASSO: linear search regCoeff=$regCoeff, delta=$delta, selected ${coeffs.numNonzeros} features out of ${coeffs.size}")
      if (coeffs.numNonzeros <= numFeatures * 0.9) {
        return linearDecrease(regCoeff - delta, delta, Some(coeffs))
      }
      if (numFeatures * 0.9 <= coeffs.numNonzeros && coeffs.numNonzeros <= numFeatures * 1.1) {
        return coeffs
      }
      return lastCoeffs match {
        case Some(c) => c
        case None => coeffs
      }
    }

    def exponentialDecrease(regCoeff: Double): Vector = {
      m.setRegParam(regCoeff)
      var coeffs = m.fit(data).coefficients
      logger.error(s"LASSO: exp search regCoeff=$regCoeff, selected ${coeffs.numNonzeros} features out of ${coeffs.size}")
      if (numFeatures * 0.9 <= coeffs.numNonzeros && coeffs.numNonzeros <= numFeatures * 1.1) {
        return coeffs
      }
      if (coeffs.numNonzeros < numFeatures) {
        return exponentialDecrease(regCoeff / 10)
      } else {
        logger.error(s"LASSO: relaxed too much, tuning regCoeff.")
        m.setRegParam(5 * regCoeff)
        coeffs = m.fit(data).coefficients
        logger.error(s"LASSO: with regCoeff=${5 * regCoeff}, selected ${coeffs.numNonzeros} features out of ${coeffs.size}")
        if (coeffs.numNonzeros < numFeatures) {
          return linearDecrease(4 * regCoeff, regCoeff, Some(coeffs))
        } else {
          return linearDecrease(9 * regCoeff, regCoeff, None)
        }
      }
    }

    var selectedFeatures = Array[(Int, Double)]()
    val coeffs = exponentialDecrease(100)
    coeffs.foreachActive((index, value) => selectedFeatures = selectedFeatures :+ (index, value))
    logger.error(s"LASSO: selected selected ${coeffs.numNonzeros} features out of ${coeffs.size} (asked for $numFeatures)")
    selectedFeatures.sortBy(e => -math.abs(e._2)).take(numFeatures).map(_._1)
  }
}

