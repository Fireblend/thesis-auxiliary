package randomModlem.modlem.strategies;

import java.util.BitSet;
import java.util.List;
import java.util.PriorityQueue;

import randomModlem.modlem.RuleCondition;
import weka.core.Instance;
import weka.core.Instances;
import randomModlem.modlem.DataSetMapper;
import randomModlem.modlem.MLRule;
import randomModlem.modlem.NumericCondition;

/**
 * Class implementing classification strategy based on searching the nearest rule(s). Proposed in:
 * Stefanowski, Jerzy. Classification support based on the rough sets. �Foundation of
 * Computing and Decision Sciences� 1993. no 18, pages 371-380.
 * 
 * @author Szymon Wojciechowski
 */
public final class NearestRulesStrategy extends ClassificationStrategy
{	
	public NearestRulesStrategy(Instances dataset, List<MLRule> rules, int abstainingStatus, DataSetMapper dataSetMap)
	{
		super(dataset, rules, abstainingStatus, dataSetMap);
		
		m_minValues = new double[m_data.numAttributes()];
		m_maxValues = new double[m_data.numAttributes()];

		for (int i = 0; i < m_data.numAttributes(); ++i)
		{
			double[] values = m_data.attributeToDoubleArray(i);
			m_minValues[i] = Double.POSITIVE_INFINITY;
			m_maxValues[i] = Double.NEGATIVE_INFINITY;
			
			for(int j = 0 ; j < values.length ; ++j)
			{
				if(Double.isNaN(values[j]))
				{
					continue;
				}
				if(m_minValues[i] > values[j])
				{
					m_minValues[i] = values[j];
				}
				else if(m_maxValues[i] < values[j])
				{
					m_maxValues[i] = values[j];
				}
			}
		}
	}
		
	public void multiple(double[] distribution, BitSet coveringRules)
	{
		for( int i = coveringRules.nextSetBit(0); i >= 0; i = coveringRules.nextSetBit(i+1))
		{
			for (int j = 0; j < m_data.numClasses(); ++j)
			{
				distribution[j] += m_rules.get(i).getClassCoverage()[j];// * rules.get(i).size();
			}
		}	
	}
	
	public void partial(double[] distribution, Instance instance)
	{
		PriorityQueue<RuleStat> queue = new PriorityQueue<RuleStat>();
		for(MLRule rule : m_rules)
		{		
			if(rule.getMatchingNumber(instance) == 0)
			{
				continue;
			}
			
			int consequent = (int)rule.getConsequent();
			int coveredPositive = rule.getClassCoverage()[consequent];
			queue.add(new RuleStat(quality(instance, rule) * coveredPositive, consequent));
		}
		
		double last = Double.MAX_VALUE;
		int i = 1;
		while(queue.size() > 0)
		{
			RuleStat currentRuleStat = queue.poll();
			if (i > RULELIMIT && last != currentRuleStat.getKey())
			{
				break;
			}
						
			distribution[currentRuleStat.getConsequent()] += currentRuleStat.getKey();		
			last = currentRuleStat.getKey();
			++i;
		}	
	}

	private double quality(Instance instance, MLRule rule)
	{
		double Di = 0;

		for(RuleCondition condition : rule.getConditions())
		{		
			if (condition.covers(instance))
			{
				continue;
			}
		
			int attribute = condition.getAttribute();
			
			if (m_data.attribute(attribute).isNumeric())
			{
				NumericCondition numericCondition = (NumericCondition)condition;
				double conditionValues[] = numericCondition.getInterval().endpoints;
				if (instance.value(attribute) >= conditionValues[1])
				{
					Di += Math.pow((instance.value(attribute) - conditionValues[1]) / (m_maxValues[attribute] - conditionValues[1]), 2);
				}
				else
				{
					Di += Math.pow((conditionValues[0] - instance.value(attribute))/ (conditionValues[0] - m_minValues[attribute]), 2);
				}
			}
			else if (m_data.attribute(attribute).isNominal())
			{
				Di += 1;
			}
		}
		
		Di = Math.sqrt(Di) / rule.size();
		return (1 - Di);
	}
	
	private static final int RULELIMIT = 7;
	private double[] m_maxValues;
	private double[] m_minValues;
	private static final long serialVersionUID = -3139777531642287386L;
}

class RuleStat implements Comparable<RuleStat>
{	
	public RuleStat(double key, int consequent)
	{
		m_key = key;
		m_consequent = consequent;
	}
	public double getKey()
	{
		return m_key;
	}
	public int getConsequent()
	{
		return m_consequent;
	}

	public int compareTo(RuleStat o)
	{
		if(m_key > o.m_key) return -1;
		if(m_key < o.m_key) return 1;
		return 0;
	}
	
	private int m_consequent;
	private double m_key;
}