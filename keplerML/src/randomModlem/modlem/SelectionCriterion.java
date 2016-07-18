package randomModlem.modlem;

import static randomModlem.modlem.Constants.CONDITIONAL_ENTROPY;
import static randomModlem.modlem.Constants.LAPLACE_ESTIMATOR;

import java.io.Serializable;
import java.util.BitSet;

import weka.core.Utils;

/**
 * Class provides methods for counting the best condition for current rule
 * 
 * @author Szymon Wojciechowski
 */
public abstract class SelectionCriterion implements Serializable
{
	public static SelectionCriterion create(int type, int numOfClasses)
	{
		switch(type)
		{
		case CONDITIONAL_ENTROPY:
			return new ConditionalEntropy();
		case LAPLACE_ESTIMATOR:
			return new LaplaceEstimator(numOfClasses);
		default:
			throw new UnsupportedOperationException("Unsupported Selection Criterion");
		}
	}
	
	public double getWorstEvaluation()
	{
		return m_worstValue;
	}
	
	public abstract double best();
	public abstract boolean compare(RuleCondition first, RuleCondition second);
	public abstract RuleCondition evaluate(RuleCondition first, RuleCondition second, BitSet uncoveredSetExamples, BitSet p_positive);
	public abstract void evaluate(RuleCondition first, BitSet uncoveredSetExamples, BitSet p_positive);
	public abstract void setWorstEvaluation(BitSet uncoveredExamples, BitSet positive);
				
	protected BitSet prepareSelectorCoverage(RuleCondition first, BitSet uncoveredSetExamples)
	{
		BitSet coverage = (BitSet) first.getCoverage().clone();
		coverage.and(uncoveredSetExamples);
		return coverage;
	}
	
	protected double m_worstValue;
	private static final long serialVersionUID = -7678488440871390800L;
}

class ConditionalEntropy extends SelectionCriterion
{	
	protected ConditionalEntropy()
	{
		m_worstValue = Double.MAX_VALUE;
	}
	
	public double best()
	{
		return 0.0;
	}
	
	public boolean compare(RuleCondition first, RuleCondition second)
	{
		return first.getEvaluation() < second.getEvaluation();
	}
	
	public RuleCondition evaluate(RuleCondition first, RuleCondition second, BitSet uncoveredSetExamples, BitSet p_positive)
	{
		BitSet coverage1 = prepareSelectorCoverage(first, uncoveredSetExamples);
		BitSet coverage2 = prepareSelectorCoverage(second, uncoveredSetExamples);
			
		first.setEvaluation(calculate(coverage1, coverage2, p_positive));
		second.setEvaluation(first.getEvaluation());
		first.setNrOfPositives(m_positiveCardinalities[0]);
		second.setNrOfPositives(m_positiveCardinalities[1]);
		if(coverage1.cardinality() >= coverage2.cardinality())
		{
			return first;
		}
		return second;
	}

	public void evaluate(RuleCondition first, BitSet uncoveredSetExamples, BitSet p_positive)
	{		
		BitSet subSet1 = prepareSelectorCoverage(first, uncoveredSetExamples);

		BitSet subSet2 = new BitSet(first.getCoverage().size());
		subSet2.set(0, first.getCoverage().size(), true);
		subSet2.xor(subSet1);
		subSet2.and(uncoveredSetExamples);
		
		first.setEvaluation(calculate(subSet1, subSet2, p_positive));
		first.setNrOfPositives(m_positiveCardinalities[0]);
	}
	
	public void setWorstEvaluation(BitSet uncoveredExamples, BitSet positive)
	{
		double pos = positive.cardinality();
		double all = uncoveredExamples.cardinality();
		
		m_worstValue = getEntropy(pos/all, (all-pos)/all);
	}

	private double calculate(BitSet set1, BitSet set2, BitSet p_positive)
	{
		final int cardinality1 = set1.cardinality();
		final int cardinality2 = set2.cardinality();

		if (cardinality1 == 0 || cardinality2 == 0)
		{
			return m_worstValue;
		}

		final int totalCardinality = cardinality1 + cardinality2;
		
		set1.and(p_positive);
		m_positiveCardinalities[0] = set1.cardinality();
		double result = conditionalEntropy(totalCardinality, cardinality1, m_positiveCardinalities[0]);

		set2.and(p_positive);
		m_positiveCardinalities[1] = set2.cardinality();
		result += conditionalEntropy(totalCardinality, cardinality2, m_positiveCardinalities[1]);

		return result;
	}
	
	private double conditionalEntropy(double totalNumberInSet, double conditionCoverage, double positiveCoverage)
	{
		double negativeCoverage = conditionCoverage - positiveCoverage;
		return conditionCoverage / totalNumberInSet * getEntropy(positiveCoverage / conditionCoverage, negativeCoverage / conditionCoverage);
	}
	
	private double entropy(double probability)
	{
		if (probability == 0) return 0;
		return -probability * Utils.log2(probability);
	}
	
	private double getEntropy(double prob1, double prob2)
	{
		return entropy(prob1) + entropy(prob2);
	}

	private int m_positiveCardinalities[] = new int[2];
	private static final long serialVersionUID = 2917887624458436860L;
}

class LaplaceEstimator extends SelectionCriterion
{	
	protected LaplaceEstimator(int numOfClasses)
	{
		m_numOfClasses = numOfClasses;
		m_worstValue = 0;
	}
	
	public double best()
	{
		return 1.0;
	}

	public boolean compare(RuleCondition first, RuleCondition second)
	{
		return first.getEvaluation() > second.getEvaluation();
	}

	public RuleCondition evaluate(RuleCondition first, RuleCondition second, BitSet uncoveredSetExamples, BitSet p_positive)
	{		
		evaluate(first, uncoveredSetExamples, p_positive);
		evaluate(second, uncoveredSetExamples, p_positive);
		
		if(compare(first, second))
		{
			return first;
		}
		return second;
	}
	
	public void evaluate(RuleCondition first, BitSet uncoveredSetExamples, BitSet p_positive)
	{	
		BitSet subSet1 = prepareSelectorCoverage(first, uncoveredSetExamples);
		first.setEvaluation(calculate(subSet1, p_positive));	
		first.setNrOfPositives(m_positiveCardinality);
	}
	
	public void setWorstEvaluation(BitSet uncoveredExamples, BitSet positive)
	{
		double pos = positive.cardinality();
		double all = uncoveredExamples.cardinality();
		
		m_worstValue = estimate(pos, all);	
	}

	private double calculate(BitSet subSet1, BitSet p_positive)
	{
		int totalCardinality = subSet1.cardinality();
		subSet1.and(p_positive);
		m_positiveCardinality = subSet1.cardinality(); 
		
		if(m_positiveCardinality == 0)
		{
			return 0.0;
		}
		return estimate(m_positiveCardinality, totalCardinality);
	}
	
	private double estimate(double positive, double all)
	{
		return (positive + 1) / (all + m_numOfClasses);
	}

	private int m_numOfClasses;
	private int m_positiveCardinality;
	private static final long serialVersionUID = -9116783032920506658L;
}
