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
        // first split the data by attribute
        Instances[] split_data = splitData(data, att);
        int[][] attr_split_array = new int[split_data.length][];

        for (int x = 0; x < split_data.length; x++){
                split_data[x].setClassIndex(split_data[x].numAttributes()-1);
                int[] class_dist = new int[split_data[x].numClasses()];

                for (double i =0.0; i < split_data[x].numClasses(); i++){
                    for (Instance instance:split_data[x]){
                        if (instance.classValue() == i){
                            class_dist[(int) i] += 1;
                        }
                    }
                }
            attr_split_array[x] = class_dist;
            }

        if (useGain){
            return measureInformationGainRatio(attr_split_array);
        }
        else{
            return measureInformationGain(attr_split_array);
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
