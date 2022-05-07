package ml_6002b_coursework;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import static ml_6002b_coursework.AttributeMeasures.measureInformationGain;
import static ml_6002b_coursework.AttributeMeasures.measureInformationGainRatio;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

    private final Boolean useGain;

    public IGAttributeSplitMeasure(Boolean useGain) {
        this.useGain = useGain;
    }

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
        // determining whether to use gain ratio or not and then returning final number
        if (useGain){
            return measureInformationGain(attr_split_array);
        }
        else{
            return measureInformationGainRatio(attr_split_array);
        }
    }


    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/Whiskey.arff"));
        Instances data = new Instances(reader);

        AttributeSplitMeasure SplitMeasure = new IGAttributeSplitMeasure(true);
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "IG", "Peaty", SplitMeasure.computeAttributeQuality(data, data.attribute("Peaty")));
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "IG", "Woody", SplitMeasure.computeAttributeQuality(data, data.attribute("Woody")));
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "IG", "Sweet", SplitMeasure.computeAttributeQuality(data, data.attribute("Sweet")));

        System.out.println("");

        AttributeSplitMeasure SplitMeasure2 = new IGAttributeSplitMeasure(false);
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "IGR", "Peaty", SplitMeasure2.computeAttributeQuality(data, data.attribute("Peaty")));
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "IGR", "Woody", SplitMeasure2.computeAttributeQuality(data, data.attribute("Woody")));
        System.out.printf("measure %s for attribute %s splitting diagnosis = %f\n", "IGR", "Sweet", SplitMeasure2.computeAttributeQuality(data, data.attribute("Sweet")));


    }

}
