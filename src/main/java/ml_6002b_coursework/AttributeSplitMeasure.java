package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Enumeration;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure {

    private double median;

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    public double getMedian(){
        return this.median;
    }

    public Instances[] compute_best_split(Instances data, Attribute att){
        Instances[] splitDataNumeric = new Instances[2];
        // https://medium.com/geekculture/handling-continuous-attributes-in-decision-trees-bbc044986621
        // working out best split value
        double[] test = data.attributeToDoubleArray(att.index());
        double median;
        Arrays.sort(test);
        if (test.length%2==1){
            median = test[(int) (test.length / 2 + 0.5)];
            for (int i = 0; i < 2; i++) {
                splitDataNumeric[i] = new Instances(data, (int) (test.length / 2 + 0.5));
            }
        }
        else{
            median = test[test.length/2];
            for (int i = 0; i < 2; i++) {
                splitDataNumeric[i] = new Instances(data, test.length/2);
            }
        }
        this.median = median;
        return splitDataNumeric;
    }

    public Instances[] splitDataOnNumeric(Instances data, Attribute att){
       Instances[] splitDataNumeric = compute_best_split(data, att);
        // performing the split
        for (Instance instance:data){
            if (instance.value(att) >= median){
                splitDataNumeric[0].add(instance);
            }
            else{
                splitDataNumeric[1].add(instance);
            }
        }
        return splitDataNumeric;
    }
    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att)].add(inst);
        }

        for (Instances split : splitData) {
            split.compactify();
        }

        return splitData;
    }

}
