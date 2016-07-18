import modules.*;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

/**
 * Created by sergio on 14/02/16.
 */
public class ExperimentMain {

    public static void main(String[] args){
        String algorithm = args[0];
        String size = args[1];
        String attrs = args[2];
        String split = args[3];
        String jobid = args[4];

        String outputDir = System.getProperty("user.home") + "/tesis/output/";
        String trainfile = "train/" + size;
        String testfile = "test/" + size;

        try {
            if (attrs.equals("no_pix")) {
                trainfile += "_no_pix_";
                testfile += "_no_pix_";
            }
            else{
                trainfile+="_";
                testfile+="_";
            }
            if (split.equals("25")) {
                trainfile += "25_train.arff";
                testfile += "75_test.arff";
            } else if (split.equals("50")) {
                trainfile += "50_train.arff";
                testfile += "50_test.arff";
            } else if (split.equals("75")) {
                trainfile += "75_train.arff";
                testfile += "25_test.arff";
            }

            Instances trainSet;
            Instances testSet;

            ConverterUtils.DataSource trainSource = new ConverterUtils.DataSource(trainfile);
            ConverterUtils.DataSource testSource = new ConverterUtils.DataSource(testfile);

            trainSet = trainSource.getDataSet();
            testSet = testSource.getDataSet();

            trainSet.setClassIndex(trainSet.numAttributes() - 1);
            testSet.setClassIndex(testSet.numAttributes() - 1);


            Module module = null;

            if (algorithm.equals("modlem-bagging")) {
                module = new BaggingModlemModule();
            } else if (algorithm.equals("modlem")) {
                module = new ModlemModule();
            } else if (algorithm.equals("random-modlem")) {
                module = new RandomModlemModule();
            } else if (algorithm.equals("random-forest")) {
                module = new RandomForestModule();
            } else if (algorithm.equals("modlem-boosting")) {
                module = new BoostingModlemModule();
            }

            long startTime = System.currentTimeMillis();
            Evaluation eval = module.executeExperiment(trainSet, testSet);
            long stopTime = System.currentTimeMillis();
            long runtime = stopTime - startTime;

            String csvString = algorithm + "," + size + "," + attrs + "," + split;
            csvString += "," + eval.pctCorrect() + "," + eval.errorRate() + "," + runtime;

            File theDir = new File(outputDir);

            if (!theDir.exists()) {
                theDir.mkdir();
            }

            PrintWriter writer = new PrintWriter(outputDir+jobid, "UTF-8");
            writer.println(csvString);
            writer.println(eval.toSummaryString());

            // Get the confusion matrix
            double[][] cmMatrix = eval.confusionMatrix();
            for(int row_i=0; row_i<cmMatrix.length; row_i++){
                for(int col_i=0; col_i<cmMatrix.length; col_i++){
                    writer.print(cmMatrix[row_i][col_i]);
                    writer.print("|");
                }
                writer.println();
            }

            writer.close();

            System.out.println(csvString);
        } catch (Exception e) {
            PrintWriter writer = null;
            try {
                writer = new PrintWriter(outputDir+jobid, "UTF-8");
            } catch (FileNotFoundException e1) {
                e1.printStackTrace();
            } catch (UnsupportedEncodingException e1) {
                e1.printStackTrace();
            }
            writer.println(e.getMessage());
            e.printStackTrace(writer);
            writer.close();
        }
    }
}
