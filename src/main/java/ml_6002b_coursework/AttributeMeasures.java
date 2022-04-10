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
            ArrayList<Double> attribute_value_probs = new ArrayList<Double>();

            //iterate through the array to find number of cases
            for (int value : Attribute) {
                cases += value;
            }

            //add to total number of cases out of loop
            total_cases += cases;

            //calculate the local probability for each value
            for (int value : Attribute) {
                double probability = (double) value/cases;
                attribute_value_probs.add(probability);
            }

            //calculate the entropy of each value
            double entropy = 0.0;
            for (Double prob : attribute_value_probs) {
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

        //set up variables for final calculation, set entropy of root node to 1 as a baseline
        double ig = 1;
        double node_value;

        //for each attribute value, calculate the weighted entropy and take it away from the root entropy
        for (int i = 0; i < attribute_value_entropy.size(); i++){
            node_value = ((double)attribute_value_cases.get(i)/total_cases)*attribute_value_entropy.get(i);
            ig -= node_value;
        }

        //return the information gain rounded to two decimal places
        return (double) Math.round(ig*100)/100;

    }

    public static double measureInformationGainRatio(int[][] data){
        double test = 1;
        return test;

    }

    public static double measureGini(int[][] data){
        double test = 1;
        return test;

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
        int[][] data = {{4,0},{1,5}};

        System.out.println(measureInformationGain(data));


    }

}
