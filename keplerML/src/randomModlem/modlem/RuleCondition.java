package randomModlem.modlem;

/**
 * Entity which represents single relation on attribute
 * 
 * @author Szymon Wojciechowski
 */

import java.io.Serializable;
import java.util.BitSet;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public abstract class RuleCondition implements Serializable, Cloneable
{		
	public RuleCondition(int p_attribute, BitSet p_coverage)
	{
		m_attribute = p_attribute;
		m_coverage = p_coverage;
	}

	public RuleCondition clone()
	{
		RuleCondition object = null;
		try
		{
			object = (RuleCondition) super.clone();
		}
		catch (CloneNotSupportedException e)
		{
			throw new UnsupportedOperationException("RuleCondition's clone() failed");
		}
		
		object.m_attribute = m_attribute;
		object.m_coverage = (BitSet) m_coverage.clone();
		object.m_evaluation = m_evaluation;
		object.m_nrOfPositives = m_nrOfPositives;
		
		return object;
	}
		
	public boolean covers(Instance p_instance)
	{
		if(p_instance.isMissing(m_attribute))
		{
			return true;
		}
		
		double value = p_instance.value(m_attribute);
		return covers(value);
	}
		
	public abstract boolean covers(double value);
	
	public int getAttribute()
	{
		return m_attribute;
	}
	
	public BitSet getCoverage()
	{
		return m_coverage;
	}
	
	public double getEvaluation()
	{
		return m_evaluation;
	}

	public int getNrOfPositives()
	{
		return m_nrOfPositives;
	}	
		
	public abstract RuleCondition merge(RuleCondition p_selector);
	
	public void setEvaluation(double p_evaluation)
	{
		this.m_evaluation = p_evaluation;
	}
		
	public void setNrOfPositives(int p_nrOfPositives)
	{
		this.m_nrOfPositives = p_nrOfPositives;
	}
		
	public abstract String toString(Instances data);
				
	protected int m_attribute;
	private BitSet m_coverage;
	private double m_evaluation;
	private int m_nrOfPositives;

	private static final long serialVersionUID = -5040472529910807683L;
	
	public static RuleCondition createNumericCondition(int attribute, Interval interval, BitSet mask)
	{
		return new NumericCondition(attribute, interval, mask);
	}
	
	public static NominalCondition createNominalCondition(int attribute, List<Double> interval, BitSet mask)
	{
		return new NominalCondition(attribute, interval, mask);
	}
}
