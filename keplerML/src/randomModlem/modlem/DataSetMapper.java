package randomModlem.modlem;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Iterator;
import java.util.TreeMap;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class provides bitsets for all attribute's values
 * 
 * @author Szymon Wojciechowski
 */
public class DataSetMapper implements Serializable
{	
	public DataSetMapper(Instances p_dataset)
	{
		m_numberOfInstances = p_dataset.numInstances();
		for (int a = 0; a < p_dataset.numAttributes(); ++a)
		{
			TreeMap<Double, BitSet> l_current = new TreeMap<Double, BitSet>();
			double[] attVals = p_dataset.attributeToDoubleArray(a);
			
			for(int i = 0 ; i < attVals.length ; ++i)
			{
				l_current.put(attVals[i], new BitSet(m_numberOfInstances));
			}
			
			m_valuesMap.add( l_current );
		}
		
		fillValuesMap(p_dataset);
	}
	
	public BitSet getBitSet(int attributeOrder, double valueOrderOrValue)
	{				
		return m_valuesMap.get(attributeOrder).get(valueOrderOrValue);
	}
	
	public BitSet getBitSetAndCheck(int attributeOrder, double valueOrderOrValue)
	{		
		if(!m_valuesMap.get(attributeOrder).containsKey(valueOrderOrValue) )
		{
			m_valuesMap.get(attributeOrder).put(valueOrderOrValue, new BitSet(m_numberOfInstances));
		}
		return getBitSet(attributeOrder, valueOrderOrValue);
	}
	
	public Iterator<Double> iterator(int attribute)
	{
		return m_valuesMap.get(attribute).keySet().iterator();
	}

	private void addNominal(Instances p_dataset, int attribute)
	{
		Instance current;
		ArrayList<Integer> l_missed = new ArrayList<Integer>();
		
		int[][] distribution = new int[p_dataset.numClasses()][p_dataset.attribute(attribute).numValues()];
					
		for (int iInstance = 0; iInstance < m_numberOfInstances ; ++iInstance)
		{
			current = p_dataset.get(iInstance);
			if(current.isMissing(attribute))
			{
				l_missed.add(iInstance);
				continue;
			}
			
			int currVal = (int)current.value(attribute);
			int currClass = (int)current.classValue();
			
			++distribution[currClass][currVal];
			getBitSet(attribute, currVal).set(iInstance);
		}

		int[] decisionsForUnknowns = new int[p_dataset.numClasses()];
		
		for(int iClass = 0 ; iClass < decisionsForUnknowns.length ; ++iClass)
		{
			decisionsForUnknowns[iClass] = Utils.maxIndex(distribution[iClass]);
		}	
		
		for(int i = 0 ; i < l_missed.size() ; ++i)
		{
			current = p_dataset.get( l_missed.get(i) );
			int decision = (int)current.classValue();
			current.setValue(attribute, decisionsForUnknowns[decision]);
			getBitSet(attribute, decisionsForUnknowns[decision]).set(l_missed.get(i));
		}
	}
	
	private void addNumeric(Instances p_dataset, int attribute)
	{
		Instance current;
		ArrayList<Integer> l_missed = new ArrayList<Integer>();
		int[] occurences = new int[p_dataset.numClasses()];
		double[] vals = new double[p_dataset.numClasses()];
			
		for (int iInstance = 0; iInstance < m_numberOfInstances ; ++iInstance)
		{
			current = p_dataset.get(iInstance);
			if(current.isMissing(attribute))
			{
				l_missed.add(iInstance);
				continue;
			}
				
			double currVal = current.value(attribute);
			int currClass = (int)current.classValue();
			getBitSet(attribute, currVal).set(iInstance);
			++occurences[currClass];
			vals[currClass] += currVal;
		}
			
		for(int i = 0 ; i < vals.length ; ++i)
		{
			if(occurences[i] != 0) vals[i] /= occurences[i];
		}
			
		for(int i = 0 ; i < l_missed.size() ; ++i)
		{
			current = p_dataset.get( l_missed.get(i) );
			int decision = (int)current.classValue();
			current.setValue(attribute, vals[decision]);
			getBitSetAndCheck(attribute, vals[decision]).set(l_missed.get(i));
		}
	}
	
	private void fillValuesMap(Instances p_dataset)
	{				
		Instance current;
		for (int a = 0; a < p_dataset.numAttributes(); ++a)
		{
			if(p_dataset.attribute(a).isNumeric())
			{
				addNumeric(p_dataset, a);
				continue;
			}
			else if(p_dataset.attribute(a).isNominal())
			{
				addNominal(p_dataset, a);
				continue;
			}
			
			for (int iInstance = 0; iInstance < m_numberOfInstances ; ++iInstance)
			{
				current = p_dataset.get(iInstance);
				if(current.isMissing(a))
				{
					continue;
				}
				getBitSet(a, current.value(a)).set(iInstance);
			}
		}
	}
		
	private int m_numberOfInstances;
	private ArrayList<TreeMap<Double, BitSet> > m_valuesMap = new ArrayList<TreeMap<Double, BitSet> >();
	private static final long serialVersionUID = -5485490654763951841L;
}