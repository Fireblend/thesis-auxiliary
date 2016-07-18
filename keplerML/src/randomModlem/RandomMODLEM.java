package randomModlem;

import static randomModlem.modlem.Constants.*;

import java.util.*;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import weka.classifiers.AbstractClassifier;
import randomModlem.modlem.strategies.ClassificationStrategy;
import randomModlem.modlem.DataSetMapper;
import randomModlem.modlem.Interval;
import randomModlem.modlem.MLRule;
import randomModlem.modlem.NumericCondition;
import randomModlem.modlem.SelectionCriterion;
import randomModlem.modlem.RuleCondition;
import randomModlem.modlem.NominalCondition;

/**
 * 
 * Class for generating a MODLEM rule set. Uses sequential covering.
 * 
 * @author Szymon Wojciechowski
 */
public class RandomMODLEM extends AbstractClassifier implements TechnicalInformationHandler
{

	Random r;

	public static void main(String[] args)
	{
		runClassifier(new RandomMODLEM(), args);
	}
	
	public void buildClassifier(Instances instances) throws Exception
	{
		r = new Random();

	    // can classifier handle the data?
	    getCapabilities().testWithFail(instances);
		
		// 	copy data
		m_data = new Instances(instances);
		m_data.deleteWithMissingClass();

		m_criterion = SelectionCriterion.create(m_conditionsMeasure, m_data.numClasses());

		List<MLRule> rules = new ArrayList<MLRule>();
		m_possibleConditions = new ArrayList<List<RuleCondition>>();
		m_dataSetMapper = new DataSetMapper(m_data);
		approximateClasses();
						
		MLRule rule;
		RuleCondition best;
		generateConditions();
		
		m_currentRuleCoverage = new BitSet(m_data.numInstances());
		m_currentRulePositiveCoverage = new BitSet(m_data.numInstances());
		BitSet coveredFromConcept = new BitSet(m_data.numInstances());	

		for (int consequent = 0; consequent < m_data.numClasses(); ++consequent)
		{		
			m_positives = m_dataSetMapper.getBitSetAndCheck(m_data.classIndex(), consequent);
			coveredFromConcept.xor(coveredFromConcept);

			while(coveredFromConcept.cardinality() < m_positives.cardinality())
			{
				m_currentRuleCoverage.set(0, m_data.numInstances(), true);
				m_currentRuleCoverage.andNot(coveredFromConcept);
				
				m_currentRulePositiveCoverage.xor(m_currentRulePositiveCoverage);
				m_currentRulePositiveCoverage.or(m_positives);
				m_currentRulePositiveCoverage.andNot(coveredFromConcept);	
				
				cleanUsage();
				rule = new MLRule(m_data, consequent);
				
				do
				{
					m_criterion.setWorstEvaluation(m_currentRuleCoverage, m_currentRulePositiveCoverage);
					best = bestCondition();

					if (best.getEvaluation() == m_criterion.getWorstEvaluation())
					{
						break;
					}

					rule.extend(best.clone());

					m_currentRuleCoverage.and(best.getCoverage());
					m_currentRulePositiveCoverage.and(best.getCoverage());
					markAsUsed(best);
				}
				while(m_currentRuleCoverage.cardinality() > m_currentRulePositiveCoverage.cardinality());

				rule.dropRedundantConditions(m_positives);
				
				if (rule.size() == 0)
				{
					break;
				}
				
				rule.mergeConditions();
				rule.calculateCoverage(m_dataSetMapper);
				rules.add(rule);
				BitSet currentRulePositiveCoverage = (BitSet) m_positives.clone();
				currentRulePositiveCoverage.and(rule.getCoverage());	
				coveredFromConcept.or(currentRulePositiveCoverage);
			}

			removeRedundantRules(rules, consequent);
		}
				
		m_classificationStrategy = ClassificationStrategy.createClassificationStrategy(m_classificationStrategyLabel, m_data, rules, m_matchingType, m_dataSetMapper);
		generateOutput(rules);
		
		m_data = null;
		m_possibleConditions = null;
		m_dataSetMapper = null;
	}
	
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String classificationStrategyTipText()
	{
		return "The classification strategy.";
	}

	/**
	 * Classifies the given test instance. The instance has to belong to a
	 * dataset when it's being classified. Note that a classifier MUST implement
	 * either this or distributionForInstance().
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return the predicted most likely class for the instance or
	 *         Instance.missingValue() if no prediction is made
	 */
	public double classifyInstance(Instance instance)
	{
		return m_classificationStrategy.classifyInstance(instance);
	}
	
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String conditionsMeasureTipText()
	{
		return "The measure to find the best condition.";
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return      the capabilities of this classifier
	 */
	public Capabilities getCapabilities()
	{
		Capabilities result = super.getCapabilities();
		result.disableAll();
		
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		
		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		
	    // instances
	    result.setMinimumNumberInstances(1);
			   			 
		return result;
	}

	/**
	 * Gets the classification strategy
	 * 
	 * @return the classification strategy
	 */
	public SelectedTag getClassificationStrategy()
	{
		return new SelectedTag(m_classificationStrategyLabel, CLASSIFICATION_STRATEGIES);
	}

	/**
	 * Gets the conditions measure.
	 * 
	 * @return the conditions measure.
	 */
	public SelectedTag getConditionsMeasure()
	{
		return new SelectedTag(m_conditionsMeasure, CONDITIONS_MEASURES);
	}

	/**
	 * Gets the matching type used to cope with ambiguous or unclassified objects.
	 * 
	 * @return the matching type
	 */
	public SelectedTag getMatchingType()
	{
		return new SelectedTag(m_matchingType, MATCHING_TYPES);
	}

	/**
	* Gets the current settings of the Classifier.
	*
	* @return an array of strings suitable for passing to setOptions
	*/
	public String[] getOptions()
	{
		String[] superOptions = super.getOptions();
		String[] options = new String[superOptions.length + 8];

		int current = 0;

		options[current++] = "-RT";
		options[current++] = "" + m_rulesType;

		options[current++] = "-CM";
		options[current++] = "" + m_conditionsMeasure;
				
		options[current++] = "-CS";
		options[current++] = "" + m_classificationStrategyLabel;
		
		options[current++] = "-AS";
		options[current++] = "" + m_matchingType;

		System.arraycopy(superOptions, 0, options, current, superOptions.length);
		
		return options;
	}
	
	/**
	 * Gets the rules type
	 * 
	 * @return the rules type
	 */
	public SelectedTag getRulesType()
	{
		return new SelectedTag(m_rulesType, RULES_TYPES);
	}
	
	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation()
	{
		TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
	    result.setValue(Field.AUTHOR, "Jerzy Stefanowski");
	    result.setValue(Field.TITLE, "The rough set based rule induction technique for classification problems");
	    result.setValue(Field.BOOKTITLE, "6th European Congress on Intelligent Techniques and Soft Computing");
	    result.setValue(Field.YEAR, "1998");
	    result.setValue(Field.PAGES, "109-113");
	    return result;
	}
			
	/**
	* Returns a string describing classifier
	* @return a description suitable for
	* displaying in the explorer/experimenter gui
	*/
	public String globalInfo()
	{
		return "Class for building and using a MODLEM algorithm to induce rule set for classification.\n\n"
				+ getTechnicalInformation().toString() + "\n\n";
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions()
	{
		Vector<Option> newVector = new Vector<Option>(4);
		
		newVector.addElement(new Option(
				"\tRules type. Possible values are: class approxmiation | lower approxmiation | upper approxmiation.\n" +
				"\t(default is l)",
				"RT", 1, "-RT <c | l | u>"));

		newVector.addElement(new Option(
				"\tConditions measure. Possible values are: conditional entropy | Laplace estimator.\n" +
				"\t(default is l)",
				"CM", 1, "-CM <e | l>"));

		newVector.addElement(new Option(
				"\tClassification strategy. Possible values are: nearest rules | strength and specificity | measure of discrimination | m-estimate | chi square | default values.\n" +
				"\t(default is m)",
				"CS", 1, "-CS <n | s | d | m | c | l>"));
		
		newVector.addElement(new Option(
				"\tMatching type. Possible values are: full matching | full and multiple matching | abstain due to uncertainty.\n" +
				"\t(default is f)",
				"AS", 1, "-AS <f | m | a>"));

		return newVector.elements();
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String matchingTypeTipText()
	{
		return "The matching type. Full matching allows to classify " +
			   "objects uncovered or covered ambiguously by rules.";
	}
	
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String rulesTypeTipText()
	{
		return "The type of generated rules.\n"
				+ "Class approximation is natural type of rules while certain "
				+ "rules are generated from lower approximation and possible rules "
				+ "are generated from upper approximation in terms of Rough-Set Theory.\n";
	}

	/**
	 * Sets the classification strategy
	 * 
	 * @param classificationStrategy the classification strategy
	 */
	public void setClassificationStrategy(SelectedTag classificationStrategy)
	{
		if (classificationStrategy.getTags() == CLASSIFICATION_STRATEGIES)
		{
			m_classificationStrategyLabel = classificationStrategy.getSelectedTag().getID();
		}
	}
	
	/**
	 * Sets the conditions measure
	 * 
	 * @param conditionsMeasure the conditions measure
	 */
	public void setConditionsMeasure(SelectedTag conditionsMeasure)
	{
		if (conditionsMeasure.getTags() == CONDITIONS_MEASURES)
		{
			m_conditionsMeasure = conditionsMeasure.getSelectedTag().getID();
		}
	}

	/**
	 * Parses a given list of options.
	 * 
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception
	{
		// set rules type
		String value = Utils.getOption("RT", options);

		if (value.toLowerCase().equals("c"))
		{
			setRulesType(new SelectedTag(NORMAL_APPROXIMATION, RULES_TYPES));
		}
		else if (value.toLowerCase().equals("l"))
		{
			setRulesType(new SelectedTag(LOWER_APPROXIMATION, RULES_TYPES));
		}
		else if(value.toLowerCase().equals("u"))
		{
			setRulesType(new SelectedTag(UPPER_APPROXIMATION, RULES_TYPES));
		}
		else
		{
			throw new IllegalArgumentException("Invalid rules approximation type");
		}

		// set conditions measure
		value = Utils.getOption("CM", options);
		if (value.toLowerCase().equals("e"))
		{
			setConditionsMeasure(new SelectedTag(CONDITIONAL_ENTROPY, CONDITIONS_MEASURES));
		}
		else if (value.toLowerCase().equals("l"))
		{
			setConditionsMeasure(new SelectedTag(LAPLACE_ESTIMATOR, CONDITIONS_MEASURES));
		}
		else
		{
			throw new IllegalArgumentException("Invalid conditions measure");
		}
				
		// set classification strategy
		value = Utils.getOption("CS", options);
		if (value.toLowerCase().equals("n"))
		{
			setClassificationStrategy(new SelectedTag(NEAREST_RULES, CLASSIFICATION_STRATEGIES));
		}
		else if (value.toLowerCase().equals("s"))
		{
			setClassificationStrategy(new SelectedTag(STRENGTH_AND_SPECIFICITY, CLASSIFICATION_STRATEGIES));
		}
		else if (value.toLowerCase().equals("d"))
		{
			setClassificationStrategy(new SelectedTag(MEASURE_OF_DISCRIMINATION, CLASSIFICATION_STRATEGIES));
		}
		else if (value.toLowerCase().equals("m"))
		{
			setClassificationStrategy(new SelectedTag(M_ESTIMATE, CLASSIFICATION_STRATEGIES));
		}
		else if (value.toLowerCase().equals("c"))
		{
			setClassificationStrategy(new SelectedTag(CHI_SQUARE, CLASSIFICATION_STRATEGIES));
		}
		else if (value.toLowerCase().equals("l"))
		{
			setClassificationStrategy(new SelectedTag(DEFAULT_CLASS, CLASSIFICATION_STRATEGIES));
		}
		else
		{
			throw new IllegalArgumentException("Invalid classification strategy");
		}
		
		value = Utils.getOption("AS", options);
		if (value.toLowerCase().equals("f"))
		{
			setMatchingType(new SelectedTag(FULL_MATCHING, MATCHING_TYPES));
		}
		else if (value.toLowerCase().equals("m"))
		{
			setMatchingType(new SelectedTag(PARTIAL_MATCHING_OFF, MATCHING_TYPES));
		}
		else if (value.toLowerCase().equals("a"))
		{
			setMatchingType(new SelectedTag(MULTI_MATCHING_OFF, MATCHING_TYPES));
		}
		else
		{
			throw new IllegalArgumentException("Invalid matching type");
		}
		
		super.setOptions(options);
	}

	/**
	 * Sets the matching type
	 * 
	 * @param matchingType the matching type
	 */
	public void setMatchingType(SelectedTag matchingType)
	{
		if (matchingType.getTags() == MATCHING_TYPES)
		{
			m_matchingType = matchingType.getSelectedTag().getID();
		}
	}

	/**
	 * Sets the rules type
	 * 
	 * @param rulesType the rules type
	 */
	public void setRulesType(SelectedTag rulesType)
	{
		if (rulesType.getTags() == RULES_TYPES)
		{
			m_rulesType = rulesType.getSelectedTag().getID();
		}
	}
	
	public String toString()
	{
		return m_string.toString();
	}

	private void approximateClasses()
	{
		if(m_rulesType == NORMAL_APPROXIMATION) return;
		
		double consequent;
		int classIndex = m_data.classIndex();
			
		for (int i = 0; i < m_data.numInstances(); ++i)
		{
			consequent = m_data.instance(i).classValue();
			BitSet currCoverage = m_dataSetMapper.getBitSet(classIndex, consequent);
			
			for (int j = i + 1; j < m_data.numInstances(); ++j)
			{
				if (currCoverage.get(j) || !compare(i,j))
				{
					continue;
				}

				if (m_rulesType == UPPER_APPROXIMATION)
				{
					currCoverage.set(j, true);
					m_dataSetMapper.getBitSet(classIndex, m_data.instance(j).classValue()).set(i, true);	
				}
				else if (m_rulesType == LOWER_APPROXIMATION)
				{
					currCoverage.set(i, false);
					m_dataSetMapper.getBitSet(classIndex, m_data.instance(j).classValue()).set(j, false);
				}

			}
		}
	}

	private RuleCondition bestCondition()
	{
		RuleCondition best = RuleCondition.createNominalCondition(-1, new ArrayList<Double>(), new BitSet());
		best.setEvaluation(m_criterion.getWorstEvaluation());
		best.setNrOfPositives(0);

		int numFeatures = r.nextInt(m_possibleConditions.size());
		if(numFeatures == 0){
			numFeatures = 1;
		}

		List<List<RuleCondition>> copy = new ArrayList<>(m_possibleConditions);
		Collections.shuffle(copy);
		copy = copy.subList(0, numFeatures);
		
		for (int attributeIndex = 0; attributeIndex < copy.size(); ++attributeIndex)
		{
			if (copy.get(attributeIndex).size() == 0) continue;

			if (m_data.attribute(attributeIndex).isNumeric())
			{
				best = bestNumerical(attributeIndex, best);
			}
			else if (m_data.attribute(attributeIndex).isNominal())
			{
				best = bestNominal(attributeIndex, best);
			}
		}
		return best;
	}
		
	private RuleCondition bestNominal(int attribute, RuleCondition best)
	{
		NominalCondition candidate = RuleCondition.createNominalCondition(attribute, new ArrayList<Double>(), new BitSet(m_data.numInstances()));
		candidate.setEvaluation(m_criterion.getWorstEvaluation());
		
		NominalCondition current;
		
		while(true)
		{
			current = (NominalCondition) candidate.clone();
			
			for(RuleCondition e : m_possibleConditions.get(attribute))
			{
				NominalCondition element = (NominalCondition)e;
				if (element.isUsed())
				{
					continue;
				}
				if (candidate.contains(element))
				{
					continue;
				}
				NominalCondition merged = (NominalCondition) candidate.clone();
				merged.extend(element);
				merged.getCoverage().or(element.getCoverage());
				
				m_criterion.evaluate(merged, m_currentRuleCoverage, m_currentRulePositiveCoverage);
				
				if (m_criterion.compare(merged, current))
				{
					current = merged;
				}
			}

			if (!m_criterion.compare(current, candidate))
			{
				break;
			}
			candidate = current;
		}
							
		return selectBetter(candidate, best);
	}
				
	private RuleCondition bestNumerical(int attribute, RuleCondition best)
	{
		NumericCondition current1;
		NumericCondition current2;
		
		for (int i = 0; i < m_possibleConditions.get(attribute).size(); i += 2)
		{
			current1 = (NumericCondition)m_possibleConditions.get(attribute).get(i);
			current2 = (NumericCondition)m_possibleConditions.get(attribute).get(i + 1);

			boolean isLesserConditionIntersected = m_availableIntervalsForNumerical[attribute].isIntersected(current1.getInterval());
			boolean isGreaterEqualConditionIntersected = m_availableIntervalsForNumerical[attribute].isIntersected(current2.getInterval());
						
			if(!(isLesserConditionIntersected && isGreaterEqualConditionIntersected))
			{
				continue;
			}
			
			RuleCondition candidate = m_criterion.evaluate(current1, current2, m_currentRuleCoverage, m_currentRulePositiveCoverage);
			best = selectBetter(candidate, best);
			if (best.getEvaluation() == m_criterion.best())
			{
				break;
			}
		}
		return best;
	}
	  
	private void cleanUsage()
	{
		for (int a = 0; a < m_possibleConditions.size(); ++a)
		{
			if(m_data.attribute(a).isNominal())
			{
				for(RuleCondition c : m_possibleConditions.get(a))
				{
					NominalCondition condition = (NominalCondition)c;
					condition.setUsed(!condition.getCoverage().intersects(m_currentRulePositiveCoverage));
				}
			}
		}
		
		m_availableIntervalsForNumerical = new Interval[m_data.numAttributes()];
		for(int i = 0 ; i < m_availableIntervalsForNumerical.length ; ++i)
			m_availableIntervalsForNumerical[i] = new Interval(Double.NEGATIVE_INFINITY, true, Double.POSITIVE_INFINITY, true);
			
	}
	
	private boolean compare(int left, int right)
	{
		for (int attribute = 0; attribute < m_data.numAttributes(); ++attribute)
		{
			if (attribute == m_data.classIndex())
			{
				continue;
			}
			
			if (m_data.instance(left).value(attribute) !=
				m_data.instance(right).value(attribute))
			{
				return false;
			}
		}
		return true;
	}
			
	private void generateConditions()
	{
		for (int attribute = 0; attribute < m_data.numAttributes(); ++attribute)
		{
			if (attribute == m_data.classIndex())
			{
				continue;
			}
			
			if (m_data.attribute(attribute).isNumeric())
			{
				generateNumerical(attribute);
			}
			else if (m_data.attribute(attribute).isNominal())
			{
				generateNominal(attribute);
			}
		}
	}
	
	private void generateNominal(int attribute)
	{
		List<RuleCondition> conditions = new ArrayList<RuleCondition>();
		RuleCondition condition;
		for (double i = 0; i < m_data.attribute(attribute).numValues(); ++i)
		{
			condition = RuleCondition.createNominalCondition(attribute, Arrays.asList(i), (BitSet)m_dataSetMapper.getBitSetAndCheck(attribute, i).clone());
			
			if (condition.getCoverage().cardinality() == 0)
			{
				continue;
			}
			conditions.add(condition);
		}
		m_possibleConditions.add(conditions);
	}
			
	private void generateNumerical(int attribute)
	{
		double[] values = m_data.attributeToDoubleArray(attribute);
		int[] sortedIndexes = Utils.sort(values);
		
		List<NumericalClassBorder> points = new ArrayList<NumericalClassBorder>();
		NumericalClassBorder temp = new NumericalClassBorder(values[sortedIndexes[0]]);
		temp.add(m_data.instance(sortedIndexes[0]).classValue());
		points.add(temp);
						
		for(int i = 1 ; i < m_data.numInstances() ; ++i)
		{
			int index = sortedIndexes[i];
			double value = values[index];
			double consequent = m_data.instance(index).classValue();
			
			if(temp.m_value != value)
			{
				temp = new NumericalClassBorder(value);
				points.add(temp);
			}
			
			temp.add(consequent);
		}
				
		BitSet mask = new BitSet(m_data.numInstances());
		BitSet negMask = new BitSet(m_data.numInstances());
		
		Iterator<Double> it = m_dataSetMapper.iterator(attribute);
		List<RuleCondition> conditions = new ArrayList<RuleCondition>();
		
		for(int i = 1 ; i < points.size() ; ++i)
		{
			double lessKey = Double.MAX_VALUE;
			double currentKey = points.get(i).m_value;
			double previousKey = points.get(i-1).m_value;
			
			while(it.hasNext())
			{
				lessKey = it.next();
				if(!(lessKey < currentKey))
				{
					break;
				}
				
				mask.or( m_dataSetMapper.getBitSet(attribute, lessKey) );
			}
			
			if(points.get(i-1).m_hasAmbiguousConsequent ||
			   points.get(i).m_hasAmbiguousConsequent ||
			   points.get(i-1).m_consequent != points.get(i).m_consequent)
			{
				double border = (previousKey + currentKey) / 2;
				negMask.set(0, m_data.numInstances(), true);
				negMask.xor(mask);

				RuleCondition condition1 = RuleCondition.createNumericCondition(attribute, new Interval(Double.NEGATIVE_INFINITY, true, border, false), (BitSet)mask.clone());
				RuleCondition condition2 = RuleCondition.createNumericCondition(attribute, new Interval(border, true, Double.POSITIVE_INFINITY, true), (BitSet)negMask.clone());
				
				conditions.add(condition1);
				conditions.add(condition2);
			}
			mask.or( m_dataSetMapper.getBitSet(attribute, lessKey) );
		}
		m_possibleConditions.add(conditions);	
	}
		
	private void generateOutput(List<MLRule> rules)
	{	
		int i = 1;
		for (MLRule rule : rules)
		{
			int covered = rule.getCoverage().cardinality();
			int positiveCoverage = rule.getClassCoverage()[(int)rule.getConsequent()];
			double coveredPositives = (double) positiveCoverage / m_dataSetMapper.getBitSet(m_data.classIndex(), rule.getConsequent()).cardinality();
			m_string.append("Rule " + i + ". " + rule + "   ("
						  + positiveCoverage + "/"+ covered + ", "
						  + Utils.doubleToString(coveredPositives * 100, 2) + "%)\n");
			++i;
		}
		m_string.append("\nNumber of Rules: " + rules.size() + "\n");
	}
		
	private void markAsUsed(RuleCondition p_best)
	{
		int attribute = p_best.getAttribute();

		if (m_data.attribute(attribute).isNumeric())
		{			
			NumericCondition condition = (NumericCondition)p_best;
			m_availableIntervalsForNumerical[attribute] = m_availableIntervalsForNumerical[attribute].intersect(condition.getInterval());
		}
		
		for (attribute = 0; attribute < m_possibleConditions.size(); ++attribute)
		{
			if(m_data.attribute(attribute).isNominal())
			{
				for(RuleCondition c : m_possibleConditions.get(attribute))
				{		
					NominalCondition condition = (NominalCondition)c;
					if (!condition.isUsed() &&
						! condition.getCoverage().intersects(m_currentRulePositiveCoverage))
					{
						condition.setUsed(true);
					}
				}
			}
		}
	}
	
	private void removeRedundantRules(List<MLRule> rules, int p_consequent)
	{
		BitSet temporaryCoverage;	
		for (int i = 0; i < rules.size(); )
		{
			if (rules.get(i).getConsequent() == p_consequent)
			{
				temporaryCoverage = new BitSet(m_data.numInstances());
				for (int j = 0; j < rules.size(); ++j)
				{
					if (rules.get(j).getConsequent() != p_consequent || i == j)
					{
						continue;
					}
					temporaryCoverage.or(rules.get(j).getCoverage());
				}
				
				temporaryCoverage.and(m_positives);
				
				if (m_positives.cardinality() == temporaryCoverage.cardinality())
				{
					rules.remove(i);
					continue;
				}
			}
			++i;
		}
	}
	
	private RuleCondition selectBetter(RuleCondition candidate, RuleCondition best)
	{
		if (candidate.getEvaluation() == best.getEvaluation())
		{
			if(candidate.getNrOfPositives() > best.getNrOfPositives())
			{
				best = candidate;
			}
		}
		else if (m_criterion.compare(candidate, best))
		{
			best = candidate;
		}
		
		return best;
	}

	/** The classification strategies */
	private static final Tag[] CLASSIFICATION_STRATEGIES = {
			new Tag(STRENGTH_AND_SPECIFICITY, "Strength and Specificity"),
			new Tag(NEAREST_RULES, "Nearest Rules"),
			new Tag(MEASURE_OF_DISCRIMINATION, "Measure of Discrimination"),
			new Tag(DEFAULT_CLASS, "Default Class"), 
			new Tag(CHI_SQUARE, "Chi-Square test"),
			new Tag(M_ESTIMATE, "m-Estimate")};

	/** The conditions measures */
	private static final Tag[] CONDITIONS_MEASURES = {
			new Tag(CONDITIONAL_ENTROPY, "Conditional entropy"),
			new Tag(LAPLACE_ESTIMATOR, "Laplace estimator") };

	/** The matching types */
	private static final Tag[] MATCHING_TYPES = {
			new Tag(FULL_MATCHING, "Full matching"),
			new Tag(PARTIAL_MATCHING_OFF, "Allow only to multiple match"),
			new Tag(MULTI_MATCHING_OFF, "Abstaining") };
	
	/** The rules types */
	private static final Tag[] RULES_TYPES = {
			new Tag(NORMAL_APPROXIMATION, "class approximation"),
			new Tag(LOWER_APPROXIMATION, "lower approximation - certain rules"),
			new Tag(UPPER_APPROXIMATION, "upper approximation - possible rules") };
			
	/** classification strategy */
	public ClassificationStrategy m_classificationStrategy;

	/** classification strategy */
	private int m_classificationStrategyLabel = M_ESTIMATE;

	/** conditions measure */
	private int m_conditionsMeasure = LAPLACE_ESTIMATOR;
		
	/** criterion in which the conditions are added to rule */
	private SelectionCriterion m_criterion;
	
	/** all examples covered by currently inducing rule */
	private BitSet m_currentRuleCoverage;
	
	/** all examples from concept covered by currently inducing rule */
	private BitSet m_currentRulePositiveCoverage;
	
	/** data */
	private Instances m_data;
		
	/** attributes and class bitset mapper */
	private DataSetMapper m_dataSetMapper;

	/** Matching type */
	private int m_matchingType = FULL_MATCHING;
	
	/** positive examples from concept */
	private BitSet m_positives;
	
	/** all possible rule conditions */
	private List< List<RuleCondition> > m_possibleConditions;

	/** Classifier's rules type */
	private int m_rulesType = LOWER_APPROXIMATION;
		
	/** classifier's output */
	private StringBuffer m_string = new StringBuffer();
	
	private Interval[] m_availableIntervalsForNumerical;
	
	private static final long serialVersionUID = -1972228786208948653L;
			
	class NumericalClassBorder
	{		
		public NumericalClassBorder(double p_value)
		{
			m_value = p_value;
			m_hasAmbiguousConsequent = false;
			m_consequent = EMPTY;
		}
		
		public void add(double p_consequent)
		{
			if(m_consequent != EMPTY)
			{
				if(p_consequent == m_consequent) return;
				
				m_hasAmbiguousConsequent = true;
				return;
			}
			
			m_consequent = p_consequent;
		}
		
		private final static int EMPTY = Integer.MIN_VALUE;
		public double m_consequent;
		public boolean m_hasAmbiguousConsequent;
		public double m_value;
	}
}