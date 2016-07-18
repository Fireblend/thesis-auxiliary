package randomModlem.modlem.strategies;

import java.util.BitSet;
import java.util.List;

import randomModlem.modlem.DataSetMapper;
import randomModlem.modlem.MLRule;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class implementing classification strategy based on chi-square test. It was proposed in:
 * Wojciechowski, Szymon. Abstaining in multiple rule classifiers - an analysis of
*  classification strategies for making a final decision. Master thesis dissertated
 * in Pozna� University of Technology.Pozna�, 2011.
 * 
 * @author Szymon Wojciechowski
 */
public final class ChiSquareStrategy extends ClassificationStrategy
{		
	public ChiSquareStrategy(Instances dataset, List<MLRule> rules, int abstainingStatus, DataSetMapper dataSetMap)
	{
		super(dataset, rules, abstainingStatus, dataSetMap);

		int att = m_data.classIndex();
		m_ruleRatings = new double[rules.size()];
		BitSet classSet = null;	
		
		for(int i = 0 ; i < rules.size() ; ++i)		
		{
			classSet = m_dataSetMap.getBitSet(att, rules.get(i).getConsequent());			
			m_ruleRatings[i] = chiSquare( rules.get(i).getCoverage(), classSet );
		}
	}
		
	public void multiple(double[] distribution, BitSet coveringRules)
	{			
		for(int i = coveringRules.nextSetBit(0); i >= 0 ; i = coveringRules.nextSetBit(i+1))
		{ 
			int ruleIndex = (int)m_rules.get(i).getConsequent();
			distribution[ruleIndex] = Math.max(distribution[ruleIndex], m_ruleRatings[i]);
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
			
			int ruleIndex = (int)rule.getConsequent();		
			double chiRes = chiSquare(rule.getPartialRuleCoverage(instance),
									  m_dataSetMap.getBitSet(m_data.classIndex(), ruleIndex) );
			distribution[ruleIndex] = Math.max(distribution[ruleIndex], chiRes);
		} 	
	}
	
	private double chiSquare(final BitSet ruleCoverage, final BitSet classCoverage)
	{
		double[][] O = new double[3][3];
		double[][] E = new double[2][2];
		
		BitSet temp = (BitSet)ruleCoverage.clone();
		temp.and(classCoverage);
		O[0][0] = temp.cardinality();
			
		temp = (BitSet)ruleCoverage.clone();
		BitSet differentClassInstances = new BitSet(m_data.numInstances());
		differentClassInstances.set(0, m_data.numInstances(), true);
		differentClassInstances.xor(classCoverage);
		temp.and(differentClassInstances);
		O[0][1] = temp.cardinality();
			
		temp.set(0, m_data.numInstances(), true);
		temp.xor(ruleCoverage);
		temp.and(differentClassInstances);
		O[1][1] = temp.cardinality();
		O[1][0] = m_data.numInstances() - O[0][0] - O[0][1] - O[1][1];
		
		O[2][0] = O[0][0] + O[1][0];
		O[2][1] = O[0][1] + O[1][1];
		O[0][2] = O[0][0] + O[0][1];
		O[1][2] = O[1][0] + O[1][1];
			
		for(int i = 0 ; i < 2 ; ++i)
		{
			for(int j = 0 ; j < 2 ; ++j)
			{
				E[i][j] = O[2][j] * O[i][2] / m_data.numInstances();
			}
		}
			
		double result = 0;
			
		for(int i = 0 ; i < 2 ; ++i)
		{
			for(int j = 0 ; j < 2 ; ++j)
			{
				result += Math.pow(O[i][j]-E[i][j],2)/E[i][j];
			}
		}
			
		return result;
	}
	
	private double[] m_ruleRatings;
	private static final long serialVersionUID = 6791293736053408793L;
}