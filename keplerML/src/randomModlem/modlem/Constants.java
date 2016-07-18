package randomModlem.modlem;

/**
 * MODLEM classifier constants
 * 
 * @author Szymon Wojciechowski
 */
public class Constants
{
	/** Rules types */
	public static final int NORMAL_APPROXIMATION = 0;
	public static final int LOWER_APPROXIMATION = 1;
	public static final int UPPER_APPROXIMATION = 2;
	
	/** Conditions measures */
	public static final int CONDITIONAL_ENTROPY = 0;
	public static final int LAPLACE_ESTIMATOR = 1;

	/** Classification strategies */
	public static final int STRENGTH_AND_SPECIFICITY = 0;
	public static final int NEAREST_RULES = 1;
	public static final int MEASURE_OF_DISCRIMINATION = 2;
	public static final int DEFAULT_CLASS = 3;
	public static final int CHI_SQUARE = 6;
	public static final int M_ESTIMATE = 8;
	
	/** Matching types */
	public static final int FULL_MATCHING = 0;
	public static final int PARTIAL_MATCHING_OFF = 1;
	public static final int MULTI_MATCHING_OFF = 2;
}
