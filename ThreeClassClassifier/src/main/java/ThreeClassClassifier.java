//Dependencies related to DL4J
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ThreeClassClassifier {

    private static final Logger log = LoggerFactory.getLogger(ThreeClassClassifier.class);
    private static long seed = 123;
    private static final Random rng = new Random (seed);
    private static List<String> labels;
    private static MultiLayerNetwork model = null;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static String modelFilename = new File(".").getAbsolutePath() + "/ThreeClassClassifier.zip";
    private static final int numClasses =3;

    //    input image will be 150x150, RGB - 3 channels image
    private static final int height = 150;
    private static final int width = 150;
    private static int channels = 3;

    //    Hyperparameters
    private static int epochs =25;
    private static int batchSize = 32;
    private static final double learningRate = 0.001;


    public static void main(String[] args) throws Exception {

        File train_dir= new File("C:\\Users\\choowilson\\Downloads\\rps-dataset\\ForDL4J\\rps");
        FileSplit trainData = new FileSplit(train_dir, NativeImageLoader.ALLOWED_FORMATS, rng);
        ImageRecordReader train_record= new ImageRecordReader(height,width,channels,labelMaker);
        train_record.initialize(trainData);
        RecordReaderDataSetIterator train_iterator = new RecordReaderDataSetIterator(train_record, batchSize,1,numClasses);

        File test_dir= new File("C:\\Users\\choowilson\\Downloads\\rps-dataset\\ForDL4J\\rps-test-set");
        FileSplit testData = new FileSplit(test_dir, NativeImageLoader.ALLOWED_FORMATS, rng);
        ImageRecordReader test_record= new ImageRecordReader(height,width,channels,labelMaker);
        test_record.initialize(testData);
        RecordReaderDataSetIterator test_iterator = new RecordReaderDataSetIterator(test_record, batchSize,1,numClasses);

        labels = train_iterator.getLabels();
        System.out.println("Labels for training: "+Arrays.toString(labels.toArray()));

        if (new File(modelFilename).exists())
        {
            Nd4j.getRandom().setSeed(seed);
            model = ModelSerializer.restoreMultiLayerNetwork(modelFilename);
        }
        else{
            Nd4j.getRandom().setSeed(seed);
            log.info("Loading images into DatasetIterator.");

            //Model architecture
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .updater(new RmsProp(learningRate))
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(0, new Convolution2D.Builder(3, 3)
                            .name("ConvInit")
                            .nIn(channels)
                            .nOut(64)
                            .activation(Activation.RELU)
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(2, 2)
                            .name("MaxPooling1")
                            .build())
                    .layer(2, new Convolution2D.Builder(3, 3)
                            .name("Conv2")
                            .nOut(64)
                            .activation(Activation.RELU)
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(2, 2)
                            .name("MaxPooling2")
                            .build())
                    .layer(4, new Convolution2D.Builder(3, 3)
                            .name("Conv3")
                            .nOut(128)
                            .activation(Activation.RELU)
                            .build())
                    .layer(5, new SubsamplingLayer.Builder(2, 2)
                            .name("MaxPooling3")
                            .build())
                    .layer(6, new Convolution2D.Builder(3, 3)
                            .name("Conv4")
                            .nOut(128)
                            .activation(Activation.RELU)
                            .build())
                    .layer(7, new SubsamplingLayer.Builder(2, 2)
                            .name("MaxPooling3")
                            .build())
                    .layer(8, new DenseLayer.Builder()
                            .name("FC1")
                            .nOut(512)
                            .activation(Activation.RELU)
                            .dropOut(0.5)
                            .build())
                    .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .name("output")
                            .nOut(3)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.convolutional(height, width, channels))
                    .build();

            model = new MultiLayerNetwork(conf);
            model.init();
            log.info("Model Summary:");
            log.info(model.summary());

            //Setup the GUI Dashboard for training
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(10), new StatsListener(storage));

            //Build Model
            log.info("Begin Model Training");
            for (int i = 1; i < epochs + 1; i++) {
                train_iterator.reset();
                while (train_iterator.hasNext()) {
                    model.fit(train_iterator.next());
                }
                log.info("Completed epoch {}", i);
            }
            //Save the model into a file
            ModelSerializer.writeModel(model, modelFilename, true);
            log.info("Model saved at {} - Done", modelFilename);
        }
        //Evaluate the accuracy of the classifier on unseen data
        Evaluation eval = model.evaluate(test_iterator);
        log.info(eval.stats());
        //Test with a single image;
        TestWithSingleImage();
    }

    private static void TestWithSingleImage() throws IOException{

        File my_image= new File("C:\\Users\\choowilson\\Desktop\\test\\rock\\rock.jpg");
        log.info("You are using this image file located at {}", my_image );
        NativeImageLoader loader = new NativeImageLoader(height,width, channels);
        INDArray image = loader.asMatrix(my_image);
        INDArray output = model.output(image);
        log.info("Labels: {}",Arrays.toString(labels.toArray()));
        log.info("Confidence Level: {}",output);
        log.info("Predicted class: {}",labels.toArray()[model.predict(image)[0]]);
    }
}
