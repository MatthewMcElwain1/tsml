package ml_6002b_coursework;
import java.util.ArrayList;
import static weka.core.Utils.log2;

/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {

    public static double measureInformationGain(int[][] data){
        //setting up array lists to store entropy and case numbers
        ArrayList<Double> attribute_value_entropy = new ArrayList<Double>();
        ArrayList<Integer> attribute_value_cases = new ArrayList<Integer>();

        int total_cases = 0;

        //iterate through the number of attributes at the split
        for (int[] Attribute : data) {
            //setup in loop cases variable and array list for value probabilities
            int cases = 0;


            //iterate through the array to find number of cases
            for (int value : Attribute) {
                cases += value;
            }

            //add to total number of cases out of loop
            total_cases += cases;

            //calculate the local probability for each value and then the entropy
            double entropy = 0.0;
            double prob;
            for (int value : Attribute) {
                prob = (double) value/cases;
                if (prob != 0.0){
                    entropy += (prob*log2(prob));
                }

            }

            //add the local number of cases and entropy (negative if not 0) to their array lists outside of loop
            attribute_value_cases.add(cases);
            if (entropy == 0.0){
                attribute_value_entropy.add(entropy);
            }
            else{
                attribute_value_entropy.add(-entropy);
            }
        }

        //calculating the entropy of the root node
        ArrayList<Integer> root_cases = new ArrayList<Integer>();
        for (int i = 0; i < data[0].length; i++){
            root_cases.add(i,0);
        }

        int temp1;
        for (int[] Attribute : data) {
            for (int i = 0; i < Attribute.length; i++){
                temp1 = root_cases.get(i);
                temp1 += Attribute[i];
                root_cases.remove(i);
                root_cases.add(i,temp1);
            }
        }

        double root_entropy = 0;
        for (Integer root_case : root_cases) {
            root_entropy += (double) root_case/total_cases*log2((double) root_case/total_cases);
        }
        root_entropy = -root_entropy;

        //for each attribute value, calculate the weighted entropy and take it away from the root entropy
        double node_value;

        for (int i = 0; i < attribute_value_entropy.size(); i++){
            node_value = ((double)attribute_value_cases.get(i)/total_cases)*attribute_value_entropy.get(i);
            root_entropy -= node_value;
        }

        //return the information gain rounded to two decimal places
        return (double) Math.round(root_entropy*100)/100;
    }

    public static double measureInformationGainRatio(int[][] data){

        double ig = measureInformationGain(data);
        int total_cases = 0;
        double split_info = 0;

        for (int[] attribute : data) {
            for (int value : attribute) {
                total_cases += value;
            }
        }

        for (int[] attribute : data) {
            int local_cases = 0;
            for (int value : attribute) {
                local_cases += value;
            }
            split_info += ((double)local_cases/total_cases)*log2((double)local_cases/total_cases);
        }

        return (double) Math.round(ig/-split_info*100)/100;
    }

    public static double measureGini(int[][] data){
        int total_cases = 0;

        for (int[] attribute : data) {
            for (int value : attribute) {
                total_cases += value;
            }
        }

        ArrayList<Double> probabilities = new ArrayList<Double>();

        for (int[] attribute : data) {
            int local_cases = 0;

            for (int value : attribute) {
               local_cases += value;
            }

            probabilities.add(Math.pow((double) local_cases/total_cases, 2));
        }

        double gini = 1;
        for (Double probability : probabilities) {
            gini -= probability;
        }

        return gini;

    }

    public static double measureChiSquared(int[][] data){
        double test = 1;
        return test;

    }
    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    public static void main(String[] args) {
        int[][] data = {{1,3},{0,2}};

        System.out.println(measureInformationGain(data));

    }

}
