package randomModlem.modlem;

import java.io.Serializable;
import java.util.Locale;

public class Interval implements Serializable
{
	private static final long serialVersionUID = 2245671966320979359L;
	public double[] endpoints = new double[2];
	private boolean[] sideClosure = new boolean[2];
	
	public Interval(double left, boolean isLeftSideClosed,
			        double right, boolean isRightSideClosed)
	{
		endpoints[0] = left;
		endpoints[1] = right;
		sideClosure[0] = isLeftSideClosed;
		sideClosure[1] = isRightSideClosed;
	}
	
	public boolean isIntersected(Interval other)
	{
		return intersect(other) != null;
	}
	
	public boolean isIntersected(double value)
	{
		return intersect(new Interval(value, true, value, true)) != null;
	}
	
	public Interval intersect(Interval other)
	{
		double left = Math.max(endpoints[0], other.endpoints[0]);
		boolean isLeftSideClosed = endpoints[0] == other.endpoints[0] ? sideClosure[0] && other.sideClosure[0] :
								   endpoints[0] == left ? sideClosure[0] : other.sideClosure[0];
		
		double right = Math.min(endpoints[1], other.endpoints[1]);	
		boolean isRightSideClosed = endpoints[1] == other.endpoints[1] ? sideClosure[1] && other.sideClosure[1] :
			   						endpoints[1] == right ? sideClosure[1] : other.sideClosure[1];
		
		if(right < left || left == right &&
				           (!isLeftSideClosed ||
				           !isRightSideClosed)) return null;
		
		return new Interval(left, isLeftSideClosed,
							right, isRightSideClosed);
	}
	
	public String toString()
	{
		return String.format(Locale.UK, "%c%.5f, %.5f%c",
							(sideClosure[0] ? '[' : '('),
							endpoints[0],
							endpoints[1],
							(sideClosure[1] ? ']' : ')'));
	}
	
	public boolean equals(Object object)
	{
		if(! (object instanceof Interval)) return false;
		Interval other = (Interval)object;
		
		return endpoints[0] == other.endpoints[0] &&
			   sideClosure[0] == other.sideClosure[0] &&
		       endpoints[1] == other.endpoints[1] &&
		       sideClosure[1] == other.sideClosure[1];
	}
			
	public static void main(String[] args)
	{
		Interval a = new Interval(0, false, 10, false);
		Interval b = new Interval(5, false, 15, false);
		assert_equals(a.intersect(b), new Interval(5, false, 10, false));
		assert_equals(b.intersect(a), new Interval(5, false, 10, false));
		
		Interval c = new Interval(15, false, 20, false);
		assert_null(a.intersect(c));
		assert_null(c.intersect(a));
		
		Interval d = new Interval(0, true, 10, false);
		assert_equals(a.intersect(d), a);
		assert_equals(d.intersect(a), a);
		
		Interval e = new Interval(0, true, 8, false);
		assert_equals(e.intersect(d), e);
		assert_equals(d.intersect(e), e);
		
		Interval f = new Interval(0, false, 10, true);
		assert_equals(a.intersect(f), a);
		assert_equals(f.intersect(a), a);
		
		Interval g = new Interval(8, false, 10, true);
		assert_equals(f.intersect(g), g);
		assert_equals(g.intersect(f), g);
		
		Interval h = new Interval(10, false, 20, false);
		assert_null(a.intersect(h));
		assert_null(h.intersect(a));
		
		Interval i = new Interval(10, true, 20, true);
		assert_null(a.intersect(i));
		assert_null(i.intersect(a));
		
		Interval j = new Interval(0, true, 10, false);
		Interval k = new Interval(0, true, 8, true);
		assert_equals(j.intersect(k), k);
		assert_equals(k.intersect(j), k);
		
		Interval m = new Interval(2, true, 10, true);
		assert_equals(m.intersect(f), m);
		assert_equals(f.intersect(m), m);
		
		Interval n = new Interval(Double.NEGATIVE_INFINITY, false, Double.POSITIVE_INFINITY, true);
		Interval o = new Interval(Double.NEGATIVE_INFINITY, true, Double.POSITIVE_INFINITY, false);
		assert_equals(n.intersect(o), new Interval(Double.NEGATIVE_INFINITY, false, Double.POSITIVE_INFINITY, false));
		assert_equals(o.intersect(n), new Interval(Double.NEGATIVE_INFINITY, false, Double.POSITIVE_INFINITY, false));
	}
	
	private static <T> void assert_equals(T left, T right)
	{
		if(! (left.equals(right)))
			throw new IllegalArgumentException(left + " does not equal to " + right);
	}
	
	private static <T> void assert_null(T left)
	{
		if(left != null)
			throw new IllegalArgumentException(left + " is not null");
	}
}