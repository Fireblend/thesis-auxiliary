package randomModlem.modlem.strategies;

import java.util.BitSet;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
import randomModlem.modlem.DataSetMapper;
import randomModlem.modlem.MLRule;

/**
 * Class implementing classification strategy based on rule's strength and specificity. Described in:
 * Grzyma�a-Busse, Jerzy Witold and Zou, Xihong. Classification Strategies Using Certain
 * and Possible Rules. In: Polkowski, Lech and Skowron, Andrzej. Rough sets and current trends In
 * computing. Warszawa: Springer-Verlag, 1998. pages 37-44.
 * 
 * @author Szymon Wojciechowski
 */
public final class StrengthAndSpecificityStrategy extends ClassificationStrategy
{
	public StrengthAndSpecificityStrategy(Instances dataset, List<MLRule> rules, int abstainingStatus, DataSetMapper dataSetMap)
	{
		super(dataset, rules,  abstainingStatus, dataSetMap);
	}
		
	public void multiple(double[] distribution, BitSet coveringRules)
	{			
		for(int i = coveringRules.nextSetBit(0); i >= 0; i = coveringRules.nextSetBit(i+1))
		{ 				
			for (int j = 0; j < m_data.numClasses() ; ++j)
			{
				distribution[j] += m_rules.get(i).getClassCoverage()[j];// * rules.get(i).size();
			}
		}	
	}
	
	public void partial(double[] distribution, Instance instance)
	{
		for(MLRule rule : m_rules)
		{		
			double coveredAtt = rule.getMatchingNumber(instance);
			if(coveredAtt == 0.0)
			{
				continue;
			}

			coveredAtt /= rule.size();
			
			for (int j = 0; j < m_data.numClasses() ; ++j)
			{
				distribution[j] += rule.getClassCoverage()[j] * coveredAtt;
			}
		}		
	}
	private static final long serialVersionUID = -3769447486890357856L;
}
