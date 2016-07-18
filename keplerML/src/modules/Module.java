package modules;

import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * Created by sergio on 14/02/16.
 */
public abstract class Module {

    public abstract Evaluation executeExperiment(Instances trainSet, Instances testSet) throws Exception;

}
