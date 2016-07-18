package randomModlem.modlem;

import java.util.BitSet;

import weka.core.Instances;

public class NumericCondition extends RuleCondition
{
	private static final long serialVersionUID = -6372903475234577069L;
	public NumericCondition(int p_attribute, Interval interval, BitSet p_coverage)
	{
		super(p_attribute, p_coverage);
		this.interval = interval;
	}
	
	public Interval getInterval()
	{
		return interval;
	}
	
	public boolean covers(double value)
	{
		return interval.isIntersected(value);
	}
	
	public RuleCondition merge(RuleCondition p_selector)
	{
		NumericCondition selector = (NumericCondition)p_selector;	
		interval = interval.intersect(selector.interval);
		getCoverage().and(p_selector.getCoverage());
		return this;
	}
	
	public String toString(Instances data)
	{
		StringBuffer output = new StringBuffer();
		output.append("(" + data.attribute(m_attribute).name());
		output.append(" in " + interval);
		output.append(")");
		return output.toString();
	}
	private Interval interval;
}