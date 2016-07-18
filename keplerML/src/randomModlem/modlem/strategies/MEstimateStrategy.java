package randomModlem.modlem.strategies;

import java.util.BitSet;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
import randomModlem.modlem.DataSetMapper;
import randomModlem.modlem.MLRule;

/**
 * Class implementing classification strategy based on m-estimate. Proposed in:
 * F�rnkranz, Johannes and Sulzmann, Jan-Nikolas. An Empirical Comparison of
 * Probability Estimation Techniques for Probabilistic Rules. In: Proceedings of Discovery
 * Science. 2009. pages 317-331.
 * 
 * @author Szymon Wojciechowski
 */
public final class MEstimateStrategy extends ClassificationStrategy
{	
	public MEstimateStrategy(Instances dataset, List<MLRule> rules, int abstainingStatus, DataSetMapper dataSetMap)
	{
		super(dataset, rules, abstainingStatus, dataSetMap);

		m_classCardinality = new int[m_data.numClasses()];
		for(int i = 0 ; i < m_data.numClasses() ; ++i)
		{
			m_classCardinality[i] = m_dataSetMap.getBitSet( m_data.classIndex(), i).cardinality();
		}
		
		m_ruleDistribution = new double[rules.size()];

		for(int i = 0 ; i < rules.size() ; ++i)
		{			
			m_ruleDistribution[i] = ruleEstimation((BitSet)rules.get(i).getCoverage().clone(), (int) rules.get(i).getConsequent());
		}
	}
			
	public void multiple(double[] distribution, BitSet coveringRules)
	{			
		for(int i = coveringRules.nextSetBit(0); i >= 0; i = coveringRules.nextSetBit(i+1))
		{ 
			int dec = (int)m_rules.get(i).getConsequent();
			distribution[dec] = Math.max(m_ruleDistribution[i], distribution[dec]);
		}	
	}
	
	public void partial(double[] distribution, Instance instance)
	{	
		for(MLRule rule : m_rules)
		{ 		
			if(rule.getMatchingNumber(instance) == 0)
			{
				continue;
			}
			
			int dec = (int)rule.getConsequent();
			double res = ruleEstimation(rule.getPartialRuleCoverage(instance), dec );
			distribution[dec] = Math.max(distribution[dec], res);
		} 		
	}
		
	private double ruleEstimation(final BitSet ruleCoverage, int consequent)
	{		
		double apriori = (double)m_classCardinality[consequent]/m_data.numInstances();
		
		int coverageSize = ruleCoverage.cardinality();
		ruleCoverage.and( m_dataSetMap.getBitSet(m_data.classIndex(), consequent) );
		int posCoverage = ruleCoverage.cardinality();

		return (posCoverage + M * apriori)/(coverageSize + M);
	}

	private static int M = 3;
	private int[] m_classCardinality;
	private double[] m_ruleDistribution;
	private static final long serialVersionUID = -4134902612280062158L;
}