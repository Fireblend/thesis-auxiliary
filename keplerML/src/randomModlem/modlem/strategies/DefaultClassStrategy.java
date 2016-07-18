package randomModlem.modlem.strategies;

import java.util.BitSet;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
import randomModlem.modlem.DataSetMapper;
import randomModlem.modlem.MLRule;

/**
 * Class implementing default class classification strategy.
 * 
 * @author Szymon Wojciechowski
 */
public final class DefaultClassStrategy extends ClassificationStrategy
{
	public DefaultClassStrategy(Instances dataset, List<MLRule> rules, int abstainingStatus, DataSetMapper dataSetMap)
	{
		super(dataset, rules, abstainingStatus, dataSetMap);
		m_classDistribution = new double[m_data.numClasses()];

		for(int i = 0 ; i < m_data.numClasses() ; ++i)
		{
			m_classDistribution[i] = m_dataSetMap.getBitSet(m_data.classIndex() ,i).cardinality();
		}
	}
		
	public void multiple(double[] distribution, BitSet coveringRules)
	{
		for( int i = coveringRules.nextSetBit(0); i >= 0; i = coveringRules.nextSetBit(i+1))
		{
			int consequent = (int)m_rules.get(i).getConsequent();
			distribution[ consequent ] += m_rules.get(i).getClassCoverage()[ consequent ];
		}
	}
	
	public void partial(double[] distribution, Instance instance)
	{
		System.arraycopy(this.m_classDistribution, 0, distribution, 0, distribution.length);
	}
	
	private double[] m_classDistribution;
	private static final long serialVersionUID = 6672187147238776823L;
}
