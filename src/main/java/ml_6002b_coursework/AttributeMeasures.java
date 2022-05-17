package ml_6002b_coursework;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

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
        ArrayList<Double> weighted_gini_measures = new ArrayList<Double>();


        for (int[] attribute : data) {
            int local_cases = 0;

            for (int value : attribute) {
               local_cases += value;
            }

            for (int value : attribute) {
                probabilities.add(Math.pow((double) value/local_cases, 2));
            }

            double gini = 1;
            for (Double probability : probabilities) {
                gini -= probability;
            }
            probabilities.clear();
            weighted_gini_measures.add(((double)local_cases/total_cases)*gini);
        }

        //calculating the gini of the root node
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

        double root_gini = 1;
        for (Integer root_case : root_cases) {
            root_gini -= Math.pow((double)root_case/total_cases, 2);
        }


        //-----------------------------------------------

        for (Double weighted_gini_measure : weighted_gini_measures) {
            root_gini -= weighted_gini_measure;
        }

        return root_gini;
    }

    public static double measureChiSquared(int[][] data){
        //calculating total cases
        int total_cases = 0;

        for (int[] attribute : data) {
            for (int value : attribute) {
                total_cases += value;
            }
        }

        ArrayList<Integer> root_cases = new ArrayList<Integer>();
        for (int i = 0; i < data[0].length; i++){
            root_cases.add(i,0);
        }

        //calculating root cases and global probability
        int temp1;
        for (int[] Attribute : data) {
            for (int i = 0; i < Attribute.length; i++){
                temp1 = root_cases.get(i);
                temp1 += Attribute[i];
                root_cases.remove(i);
                root_cases.add(i,temp1);
            }
        }

        ArrayList<Double> class_probs = new ArrayList<Double>();
        for (Integer root_case : root_cases) {
            class_probs.add((double)root_case/total_cases);
        }

        //calculating expected values
        ArrayList<Double> expected_values = new ArrayList<Double>();
        for (int[] Attribute : data) {
            int attribute_value_total = 0;
            for (int value : Attribute) {
                attribute_value_total += value;
            }

            for (int i = 0; i < Attribute.length; i++){
                expected_values.add(attribute_value_total*class_probs.get(i));
            }
        }

     //--------------------------------------------------------------

        ArrayList<Double> node_values = new ArrayList<Double>();
        int i = 0;
        for (int[] Attribute : data) {
            for (int value : Attribute) {
                node_values.add((Math.pow((double)value-expected_values.get(i), 2))/expected_values.get(i));
                i++;
            }
        }

        double chi_squared = 0;
        for (Double node_value : node_values) {
            chi_squared += node_value;
        }

        return (double)Math.round(chi_squared*100)/100;

    }

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */

    public static void main(String[] args) throws Exception {
        int[][] data = {{0,5},{1,0}};

        BufferedReader reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/Whiskey.arff"));
        Instances test = new Instances(reader);

        BufferedReader test1 = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff"));
        Instances test2 = new Instances(test1);


        AttributeSplitMeasure SplitMeasure = new GiniAttributeSplitMeasure();



        //System.out.println(measureInformationGain(data));

    }

}
