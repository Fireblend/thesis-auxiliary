package modules;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.MODLEM;
import weka.core.Instances;

import java.util.Random;

/**
 * Created by sergio on 19/02/16.
 */
public class BaggingModlemModule extends Module {

    Classifier modlem;
    Bagging bagger;

    public BaggingModlemModule(){
        bagger = new Bagging();
        modlem = new MODLEM();

        bagger.setClassifier(modlem);
        bagger.setSeed(new Random().nextInt());
        bagger.setNumIterations(20);
        bagger.setBagSizePercent(100);
        bagger.setCalcOutOfBag(false);
    }

    @Override
    public Evaluation executeExperiment(Instances trainSet, Instances testSet) throws Exception {
        bagger.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(bagger, testSet);
        return eval;
    }
}
