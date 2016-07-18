import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by sergio on 16/02/16.
 */
public class SimpleRunGen {

    public static int sizes_count = 2;
    public static int attr_count = 2;
    public static int splits_count = 3;
    public static int algo_count = 5;

    public static int reps = 5;

    static String root = "/home/sergio/DATASETS/";

    public static String[] algos = {"modlem", "modlem-bagging", "modlem-boosting", "random-forest", "random-modlem"};
    public static String[] sizes = {"all", "half"};
    public static String[] attrs = {"all", "no_pix"};
    public static String[] splits = {"25", "50", "75"};


    public static void main(String[] args){

        int jobCount = 1000;

        try {
            PrintWriter writer = new PrintWriter(root + "runExperiment.sh", "UTF-8");

            for(int a = 0; a < reps; a++) {
                List<String> list = new ArrayList<String>();

                for (int h = 0; h < algo_count; h++) {
                    for (int i = 0; i < sizes_count; i++) {
                        for (int j = 0; j < attr_count; j++) {
                            for (int k = 0; k < splits_count; k++) {
                                String str = "./kepler \""+
                                        algos[h]+"\" \""+sizes[i]+"\" \""+attrs[j]+"\" "+splits[k];
                                list.add(str);
                            }
                        }
                    }
                }

                Collections.shuffle(list);
                for(int x = 0; x < list.size(); x++){
                    writer.println(list.get(x)+" "+jobCount + " >> " + "OUTPUT.TXT");
                    jobCount++;
                }
            }


            writer.close();
        }
        catch(Exception e){

        }

    }

}
