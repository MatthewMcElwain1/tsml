package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

import static ml_6002b_coursework.AttributeMeasures.measureInformationGain;
import static ml_6002b_coursework.AttributeMeasures.measureInformationGainRatio;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

    @Override
    public double computeAttributeQuality(Instances data, Attribute att, Boolean useGain) throws Exception {
        // first split the data by attribute value
        Instances[] split_data = splitData(data, att);
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
        // determining whether to use gain ratio or not and then returning final number
        if (useGain){
            return measureInformationGain(attr_split_array);
        }
        else{
            return measureInformationGainRatio(attr_split_array);
        }
    }

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        return 0;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) {

    }

}
