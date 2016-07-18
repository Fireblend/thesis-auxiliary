package modules;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.MODLEM;
import weka.core.Instances;

import java.util.Random;

/**
 * Created by sergio on 19/02/16.
 */
public class BoostingModlemModule extends Module {

    Classifier modlem;
    AdaBoostM1 booster;

    public BoostingModlemModule(){
        booster = new AdaBoostM1();
        modlem = new MODLEM();

        booster.setClassifier(modlem);
        booster.setSeed(new Random().nextInt());
        booster.setNumIterations(25);
        booster.setUseResampling(false);
        booster.setWeightThreshold(100);
    }

    @Override
    public Evaluation executeExperiment(Instances trainSet, Instances testSet) throws Exception {
        booster.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(booster, testSet);
        return eval;
    }
}
