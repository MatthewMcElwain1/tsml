package ml_6002b_coursework;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import static ml_6002b_coursework.AttributeMeasures.measureGini;


public class GiniAttributeSplitMeasure extends AttributeSplitMeasure {


    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        Instances[] split_data;
        if (att.type() == 0){
            split_data = splitDataOnNumeric(data, att);
        }
        else{
            split_data = splitData(data, att);
        }
        // first split the data by attribute value
        // setup attr split array
        int[][] attr_split_array = new int[split_data.length][];
        // loop through the split data
        for (int x = 0; x < split_data.length; x++){
            // set the class attribute
            split_data[x].setClassIndex(split_data[x].numAttributes()-1);
            // setup class distribution array
            int[] class_dist = new int[split_data[x].numClasses()];
            // loop through the instances x (how many classes) number of times
            for (double i =0.0; i < split_data[x].numClasses(); i++){
                for (Instance instance:split_data[x]){
                    // if the class value of the instance is the same as the current class loop add one to the correct number in the array
                    if (instance.classValue() == i){
                        class_dist[(int) i] += 1;
                    }
                }
            }
            // adding the class dist array to the main split array
            attr_split_array[x] = class_dist;

        }
        // returning final number
        return measureGini(attr_split_array);
    }


    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/Whiskey.arff"));
        Instances data = new Instances(reader);

        AttributeSplitMeasure SplitMeasure = new GiniAttributeSplitMeasure();
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "GINI", "Peaty", SplitMeasure.computeAttributeQuality(data, data.attribute("Peaty")));
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "GINI", "Woody", SplitMeasure.computeAttributeQuality(data, data.attribute("Woody")));
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "GINI", "Sweet", SplitMeasure.computeAttributeQuality(data, data.attribute("Sweet")));

        BufferedReader reader1 = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/test_data/optdigits.arff"));
        Instances optDigits = new Instances(reader1);


        System.out.println(SplitMeasure.computeAttributeQuality(optDigits, optDigits.attribute(15)));

    }

}
