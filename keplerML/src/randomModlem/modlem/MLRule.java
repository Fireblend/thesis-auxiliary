package randomModlem.modlem;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

import weka.classifiers.rules.Rule;
import weka.core.Instance;
import weka.core.Instances;

/**
 * MODLEM specific rule entity
 * 
 * @author Szymon Wojciechowski 
 */
public class MLRule extends Rule
{
	public MLRule(Instances dataset, int currentClassValue)
	{
		m_data = dataset;
		m_consequent = currentClassValue;
		m_conditions = new ArrayList<RuleCondition>();
		m_coverage = new BitSet(m_data.numInstances());
		m_coverage.set(0, m_data.numInstances(), true);
		m_classCoverage = new int[m_data.numClasses()];
	}

	public void calculateCoverage(DataSetMapper p_DataSetMapper)
	{	
		m_coverage.set(0, m_coverage.size(), true);
		for (RuleCondition condition : m_conditions)
		{
			m_coverage.and(condition.getCoverage());
		}
		
		for(int i = 0 ; i < m_data.numClasses() ; ++i)
		{		
			BitSet copy = (BitSet) p_DataSetMapper.getBitSet(m_data.classIndex(),i).clone();
			copy.and(m_coverage);
			m_classCoverage[i] = copy.cardinality();
		}
	}
	
	public boolean covers(Instance datum)
	{
		return getMatchingNumber(datum) == m_conditions.size();
	}
	
	public void dropRedundantConditions(BitSet positiveExamples)
	{	
		for (int i = 0; i < m_conditions.size() ; )
		{
			BitSet newCoverage = new BitSet(m_data.numInstances());
			newCoverage.set(0, newCoverage.size(), true);
			
			for (int j = 0; j < m_conditions.size(); ++j)
			{				
				if (i == j) continue;
				newCoverage.and(m_conditions.get(j).getCoverage());		
			}
			int cardinality = newCoverage.cardinality();
			newCoverage.and(positiveExamples);
			
			if (newCoverage.cardinality() == cardinality)
			{
				m_conditions.remove(i);
			}
			else
			{
				++i;
			}
		}
	}
	
	public void extend(RuleCondition p_condition)
	{
		m_conditions.add(p_condition);
	}
	
	public int[] getClassCoverage()
	{
		return m_classCoverage;
	}
	
	public List<RuleCondition> getConditions()
	{
		return m_conditions;
	}
	
	public double getConsequent()
	{
		return m_consequent;
	}
		
	public BitSet getCoverage()
	{
		return m_coverage;
	}
	
	public int getMatchingNumber(Instance instance)
	{
		int result = 0;
		for (RuleCondition condition : m_conditions)
		{
			if ( condition.covers( instance ) )
			{
				++result;
			}
		}
		return result;
	}

	public BitSet getPartialRuleCoverage(Instance instance)
	{		
		BitSet ruleCoverage = new BitSet(m_data.numInstances());		
		ruleCoverage.set(0, m_data.numInstances(), true); 
		
		for(RuleCondition condition : m_conditions)
		{
			if( condition.covers( instance ) )
			{
				ruleCoverage.and( condition.getCoverage() );
			}
		}
		
		return ruleCoverage;
	}
	
	public String getRevision()
	{
		throw new UnsupportedOperationException("getRevision is forbidden");
	}

	public void grow(Instances data) throws Exception
	{
		throw new UnsupportedOperationException("grow is forbidden");
	}

	public boolean hasAntds()
	{
		return m_conditions.size() != 0;
	}

	public void mergeConditions()
	{
		for(int j = 0 ; j < m_conditions.size(); ++j)
		{	
			RuleCondition old = m_conditions.get(j);
			for(int i = j + 1 ; i < m_conditions.size() ; )
			{
				RuleCondition selector = m_conditions.get(i);
				if(selector.getAttribute() == old.getAttribute())
				{
					m_conditions.set(j, old.merge(selector));			
					m_conditions.remove(i);	
				}
				else
				{
					++i;
				}
			}	
		}
	}
	
	public double size()
	{
		return m_conditions.size();
	}
	
	public String toString()
	{
		String end = " => (" + m_data.attribute(m_data.classIndex()).name() + " = "
					 + m_data.attribute(m_data.classIndex()).value(m_consequent) + ")";
		if (m_conditions.size() == 0)
		{
			return "{}" + end;
		}
		StringBuffer output = new StringBuffer();
		int i = 0;
		while(true)
		{
			output.append(m_conditions.get(i).toString(m_data));
			++i;
			if (i == m_conditions.size())
			{
				break;
			}
			output.append("&");	
		}
		output.append(end);

		return output.toString();
	}
	
	private int[] m_classCoverage;
	private List<RuleCondition> m_conditions;
	private int m_consequent;
	private BitSet m_coverage;
	private Instances m_data;
	private static final long serialVersionUID = 7839030997018897507L;
}