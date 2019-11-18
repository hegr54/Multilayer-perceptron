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

   // specify layers for the neural network:
   // input layer of size 4 (features), two intermediate of size 5 and 4
   // and output of size 3 (classes)
   val layers = Array[Int](4, 5, 4, 3)

   // create the trainer and set its parameters
   val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

   // train the model
   val model = trainer.fit(train)

   // compute accuracy on the test set
   val result = model.transform(test)

// compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
    // $example off$

// scalastyle:on println
