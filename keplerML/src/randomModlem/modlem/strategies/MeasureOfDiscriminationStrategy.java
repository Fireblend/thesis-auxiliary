package randomModlem.modlem.strategies;

import java.util.BitSet;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
import randomModlem.modlem.DataSetMapper;
import randomModlem.modlem.MLRule;

/**
 * Class implementing classification strategy based on Aijun An's measure described in
 * An, Aijun. Learning Classification Rules from Data. �International Journal of
 * Computers and Mathematics with Applications� 2003, no 45, pages 737-748.
 * 
 * @author Szymon Wojciechowski
 */
public final class MeasureOfDiscriminationStrategy extends ClassificationStrategy
{
	public MeasureOfDiscriminationStrategy(Instances dataset, List<MLRule> rules, int abstainingStatus, DataSetMapper dataSetMap)
	{
		super(dataset, rules, abstainingStatus, dataSetMap);
		m_ruleRating = new double[rules.size()];
		
		for(int i = 0 ; i < rules.size() ; ++i)
		{
			m_ruleRating[i] = quality(rules.get(i));
		}
	}
	
	public void multiple(double[] distribution, BitSet coveringRules)
	{		
		for( int i = coveringRules.nextSetBit(0); i >= 0; i = coveringRules.nextSetBit(i+1))
		{ 
			distribution[(int)m_rules.get(i).getConsequent()] += m_ruleRating[i];
		}
		
		normalize(distribution);
	}
		
	public void partial(double[] distribution, Instance instance)
	{
		for (int i = 0; i < m_rules.size(); ++i)
		{	 					
			MLRule rule = m_rules.get(i);
			double coveredAtt = rule.getMatchingNumber(instance);
			if(coveredAtt == 0.0)
			{
				continue;
			}
			
			coveredAtt /= rule.size();
			
			distribution[(int)rule.getConsequent()] += m_ruleRating[i] * coveredAtt;
		}
		
		normalize(distribution);		
	}
	
	private void normalize(double[] distribution)
	{
		for(int i = 0 ; i < distribution.length ; ++i)
		{
			if(distribution[i] < 0)
			{
				distribution[i] = 0;
			}
		}
	}
		
	private double quality(MLRule rule)
	{
		BitSet classCoverage = m_dataSetMap.getBitSet(m_data.classIndex(), rule.getConsequent());
		BitSet ruleCoverage = (BitSet) rule.getCoverage().clone();
		double n = ruleCoverage.cardinality();
		ruleCoverage.and( classCoverage ) ;
		double m = ruleCoverage.cardinality();
		double M = classCoverage.cardinality();
		double N = m_data.numInstances();
		
		return Math.log10(((m + 0.5) * (N - n	- M + m + 0.5))	/ ((n - m + 0.5) * (M - m + 0.5)));
	}
	
	private double m_ruleRating[];
	private static final long serialVersionUID = 2486019578327707077L;
}
