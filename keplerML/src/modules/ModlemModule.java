package modules;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.MODLEM;
import weka.core.Instances;

/**
 * Created by sergio on 14/02/16.
 */
public class ModlemModule extends Module {

    Classifier modlem;

    public ModlemModule(){
        modlem = new MODLEM();
    }

    @Override
    public Evaluation executeExperiment(Instances trainSet, Instances testSet) throws Exception {
        modlem.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(modlem, testSet);
        return eval;
    }

}
