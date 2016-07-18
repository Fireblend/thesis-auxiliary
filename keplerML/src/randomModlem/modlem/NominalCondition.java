package randomModlem.modlem;

import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.core.Attribute;
import weka.core.Instances;

public class NominalCondition extends RuleCondition
{
	private static final long serialVersionUID = 6152919093855713317L;
	public NominalCondition(int p_attribute, List<Double> values, BitSet p_coverage)
	{
		super(p_attribute, p_coverage);
		m_usage = false;
		m_values.addAll(values);
	}	
	
	public boolean covers(double value)
	{
		return contains(value);
	}

	public void extend(RuleCondition other)
	{
		NominalCondition o = (NominalCondition)other;
		m_values.addAll(o.m_values);
	}
	
	public boolean contains(RuleCondition other)
	{
		NominalCondition o = (NominalCondition)other;
		for(Double v : o.m_values)
			if(!contains(v))
				return false;
		return true;
	}
	
	public boolean isUsed()
	{
		return m_usage;
	}
	
	public void setUsed(boolean p_usage)
	{
		this.m_usage = p_usage;
	}
	
	protected boolean contains(double p_value)
	{
		return m_values.contains(p_value);
	}
	
	public RuleCondition merge(RuleCondition p_selector)
	{
		return p_selector;
	}
	
	public String toString(Instances data)
	{
		StringBuffer output = new StringBuffer();
		Attribute attribute = data.attribute(m_attribute);
		String prefix = "(" + attribute.name() + " in {";
	
		for (Double v : m_values)
			output.append(", " + attribute.value(v.intValue()));
		
		return prefix + output.substring(2) + "})";
	}
	
	public NominalCondition clone()
	{
		NominalCondition object = (NominalCondition) super.clone();		
		object.m_usage = m_usage;
		object.m_values = new HashSet<Double>();
		object.m_values.addAll(m_values);
		return object;
	}
	
	private Set<Double> m_values = new HashSet<Double>();
	private boolean m_usage;
}