import java.io.*;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by sergio on 16/02/16.
 */
public class Splitter {
    static Random random = new Random();
    static String[] files_all = {"all", "all_no_pix"};
    static String[] files_half = {"half", "half_no_pix"};
    static String root = "/home/sergio/DATASETS/";

    public static void main(String[] args){
        for(int i =0; i< files_all.length; i++) {
            String file = files_all[i];
            split(5717/3, 75, 1906/3, 25, file);
            split(3811/3, 50, 3811/3, 50, file);
            split(1906/3, 25, 5717/3, 75, file);
        }
        for(int i =0; i< files_half.length; i++) {
            String file = files_half[i];
            split(2857/3, 75, 953/3, 25, file);
            split(1905/3, 50, 1905/3, 50, file);
            split(953/3, 25, 2857/3, 75, file);
        }
    }
    //Big samples = 7623  5717/1906  ====  3811/3811
    //Low samples = 3810  2857/953   ====  1905/1905

    public static void split(int train, int trainp, int test, int testp, String filename){
        try {
            PrintWriter trainWriter = new PrintWriter(root+"train/"+filename+"_"+trainp+"_train.arff", "UTF-8");
            PrintWriter testWriter = new PrintWriter(root+"test/"+filename+"_"+testp+"_test.arff", "UTF-8");

            InputStream fis = new FileInputStream(root+filename+".arff");
            InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
            BufferedReader br = new BufferedReader(isr);

            String line;
            String PC = "PC";
            String AFP = "AFP";
            String NTP = "NTP";

            HashMap<String, Integer> trainBins = new HashMap<>();
            trainBins.put(PC, train);
            trainBins.put(AFP, train);
            trainBins.put(NTP, train);
            HashMap<String, Integer> testBins = new HashMap<>();
            testBins.put(PC, test);
            testBins.put(AFP, test);
            testBins.put(NTP, test);


            while ((line = br.readLine()) != null) {
                if(line.endsWith(PC) || line.endsWith(AFP) || line.endsWith(NTP)){
                    if(line.endsWith(PC)){
                        if(trainBins.get(PC) != 0 && testBins.get(PC) != 0){
                            if(random.nextInt(1) == 1){
                                trainBins.put(PC, trainBins.get(PC)-1);
                                trainWriter.println(line);
                            }
                            else {
                                testBins.put(PC, testBins.get(PC)-1);
                                testWriter.println(line);
                            }
                        }
                        else if(trainBins.get(PC) != 0){
                            trainBins.put(PC, trainBins.get(PC)-1);
                            trainWriter.println(line);
                        }
                        else if(testBins.get(PC) != 0){
                            testBins.put(PC, testBins.get(PC)-1);
                            testWriter.println(line);
                        }
                    }

                    if(line.endsWith(AFP)){
                        if(trainBins.get(AFP) != 0 && testBins.get(AFP) != 0){
                            if(random.nextInt(1) == 1){
                                trainBins.put(AFP, trainBins.get(AFP)-1);
                                trainWriter.println(line);
                            }
                            else {
                                testBins.put(AFP, testBins.get(AFP)-1);
                                testWriter.println(line);
                            }
                        }
                        else if(trainBins.get(AFP) != 0){
                            trainBins.put(AFP, trainBins.get(AFP)-1);
                            trainWriter.println(line);
                        }
                        else if(testBins.get(AFP) != 0){
                            testBins.put(AFP, testBins.get(AFP)-1);
                            testWriter.println(line);
                        }
                    }

                    if(line.endsWith(NTP)){
                        if(trainBins.get(NTP) != 0 && testBins.get(NTP) != 0){
                            if(random.nextInt(1) == 1){
                                trainBins.put(NTP, trainBins.get(NTP)-1);
                                trainWriter.println(line);
                            }
                            else {
                                testBins.put(NTP, testBins.get(NTP)-1);
                                testWriter.println(line);
                            }
                        }
                        else if(trainBins.get(NTP) != 0){
                            trainBins.put(NTP, trainBins.get(NTP)-1);
                            trainWriter.println(line);
                        }
                        else if(testBins.get(NTP) != 0){
                            testBins.put(NTP, testBins.get(NTP)-1);
                            testWriter.println(line);
                        }
                    }

                }
                else {
                    trainWriter.println(line);
                    testWriter.println(line);
                }
            }

            trainWriter.close();
            testWriter.close();

        } catch (Exception e) {

        }
    }
}


