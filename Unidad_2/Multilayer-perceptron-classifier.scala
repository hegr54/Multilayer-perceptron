//importacion de las Api's necesarias para la implementacion del algoritmo MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example for Multilayer Perceptron Classification.
 */
 ///creacion del inicio de seccion de spark.
 val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()
//lectura de nuestro dataset en un formato libsvm tambien carga de nuestro dataset
    val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
//creacion de nuestras variable de entrenamiento, con un 60% de entrenamiento y 40% de prueba de nuestros datos juneto con una semilla de profundidad de 1234L.
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)