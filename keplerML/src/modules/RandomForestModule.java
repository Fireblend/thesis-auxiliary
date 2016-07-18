package modules;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.Random;

/**
 * Created by sergio on 18/02/16.
 */
public class RandomForestModule extends Module {
    RandomForest rf;

    public RandomForestModule(){
        rf = new RandomForest();
        rf.setNumTrees(100);
        rf.setSeed(new Random().nextInt());
    }

    @Override
    public Evaluation executeExperiment(Instances trainSet, Instances testSet) throws Exception {
        rf.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(rf, testSet);
        return eval;
    }
}
