package ml_6002b_coursework;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

/**
 * A basic decision tree classifier for use in machine learning coursework (6002B).
 */
public class CourseworkTree extends AbstractClassifier {

    /** Measure to use when selecting an attribute to split the data with. */
    private AttributeSplitMeasure attSplitMeasure;

    /** Maxiumum depth for the tree. */
    private int maxDepth = Integer.MAX_VALUE;

    /** The root node of the tree. */
    private TreeNode root;

    /**
     * Sets the attribute split measure for the classifier.
     *
     * @param attSplitMeasure the split measure
     */
    public void setAttSplitMeasure(AttributeSplitMeasure attSplitMeasure) {
        this.attSplitMeasure = attSplitMeasure;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String     tmpStr;

        tmpStr = Utils.getOption('S', options);

        switch (tmpStr) {
            case "gain":
                setAttSplitMeasure(new IGAttributeSplitMeasure(true));
                break;
            case "ratio":
                setAttSplitMeasure(new IGAttributeSplitMeasure(false));
                break;
            case "gini":
                setAttSplitMeasure(new GiniAttributeSplitMeasure());
                break;
            case "chi":
                setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
            default:
                setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
                break;
        }

        tmpStr = Utils.getOption('D', options);

        switch (tmpStr) {
            case "1":
                setMaxDepth(1);
                break;
            case "2":
                setMaxDepth(2);
                break;
            case "4":
                setMaxDepth(4);
                break;
            case "8":
                setMaxDepth(8);
                break;
            case "16":
                setMaxDepth(16);
                break;
            case "32":
                setMaxDepth(32);
                break;
            case "64":
                setMaxDepth(64);
                break;
            default:
                setMaxDepth(Integer.MAX_VALUE);
                break;
        }


    }

    /**
     * Sets the max depth for the classifier.
     *
     * @param maxDepth the max depth
     */
    public void setMaxDepth(int maxDepth){
        this.maxDepth = maxDepth;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //instances
        result.setMinimumNumberInstances(2);

        return result;
    }

    /**
     * Builds a decision tree classifier.
     *
     * @param data the training data
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last index.");
        }

        root = new TreeNode();
        root.buildTree(data, 0);
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    /**
     * Computes class distribution for instance using the decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    /**
     * Class representing a single node in the tree.
     */
    private class TreeNode {

        /** Attribute used for splitting, if null the node is a leaf. */
        Attribute bestSplit = null;

        /** Best gain from the splitting measure if the node is not a leaf. */
        double bestGain = 0;

        /** Depth of the node in the tree. */
        int depth;

        /** The node's children if it is not a leaf. */
        TreeNode[] children;

        /** The class distribution if the node is a leaf. */
        double[] leafDistribution;

        /**
         * Recursive function for building the tree.
         * Builds a single tree node, finding the best attribute to split on using a splitting measure.
         * Splits the best attribute into multiple child tree node's if they can be made, else creates a leaf node.
         *
         * @param data Instances to build the tree node with
         * @param depth the depth of the node in the tree
         */
        void buildTree(Instances data, int depth) throws Exception {
            this.depth = depth;

            // Loop through each attribute, finding the best one.
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                double gain;

                gain = attSplitMeasure.computeAttributeQuality(data, data.attribute(i));

                if (gain > bestGain) {
                    bestSplit = data.attribute(i);
                    bestGain = gain;
                }
            }

            // If we found an attribute to split on, create child nodes.
            if (bestSplit != null) {
                Instances[] split;
                if (bestSplit.type() == 0){
                   split = attSplitMeasure.splitDataOnNumeric(data, bestSplit);
                }
                else{
                   split = attSplitMeasure.splitData(data, bestSplit);
                }

                children = new TreeNode[split.length];

                // Create a child for each value in the selected attribute, and determine whether it is a leaf or not.
                for (int i = 0; i < children.length; i++){
                    children[i] = new TreeNode();

                    boolean leaf = split[i].numDistinctValues(data.classIndex()) == 1 || depth + 1 == maxDepth;

                    if (split[i].isEmpty()) {
                        children[i].buildLeaf(data, depth + 1);
                    } else if (leaf) {
                        children[i].buildLeaf(split[i], depth + 1);
                    } else {
                        children[i].buildTree(split[i], depth + 1);
                    }
                }
            // Else turn this node into a leaf node.
            } else {
                leafDistribution = classDistribution(data);
            }
        }

        /**
         * Builds a leaf node for the tree, setting the depth and recording the class distribution of the remaining
         * instances.
         *
         * @param data remaining Instances to build the leafs class distribution
         * @param depth the depth of the node in the tree
         */
        void buildLeaf(Instances data, int depth) {
            this.depth = depth;
            leafDistribution = classDistribution(data);
        }

        /**
         * Recursive function traversing node's of the tree until a leaf is found. Returns the leafs class distribution.
         *
         * @return the class distribution of the first leaf node
         */
        double[] distributionForInstance(Instance inst) {
            // If the node is a leaf return the distribution, else select the next node based on the best attributes
            // value.
            if (bestSplit == null) {
                return leafDistribution;
            } else {
                if (bestSplit.isNumeric()){
                    if(inst.value(bestSplit) < attSplitMeasure.getMedian()){
                        return children[1].distributionForInstance(inst);
                    }
                    else{
                        return children[0].distributionForInstance(inst);
                    }

                }
                return children[(int) inst.value(bestSplit)].distributionForInstance(inst);
            }
        }

        /**
         * Returns the normalised version of the input array with values summing to 1.
         *
         * @return the class distribution as an array
         */
        double[] classDistribution(Instances data) {
            double[] distribution = new double[data.numClasses()];
            for (Instance inst : data) {
                distribution[(int) inst.classValue()]++;
            }

            double sum = 0;
            for (double d : distribution){
                sum += d;
            }

            if (sum != 0){
                for (int i = 0; i < distribution.length; i++) {
                    distribution[i] = distribution[i] / sum;
                }
            }

            return distribution;
        }

        /**
         * Summarises the tree node into a String.
         *
         * @return the summarised node as a String
         */
        @Override
        public String toString() {
            String str;
            if (bestSplit == null){
                str = "Leaf," + Arrays.toString(leafDistribution) + "," + depth;
            } else {
                str = bestSplit.name() + "," + bestGain + "," + depth;
            }
            return str;
        }
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier main
     */
    public static void main(String[] args) throws Exception {
        // --------------------optdigits problems-------------------------------
        BufferedReader reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/test_data/optdigits.arff"));
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes()-1);


        // https://stackoverflow.com/questions/28123954/how-do-i-divide-a-dataset-into-training-and-test-sets-using-weka
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();

        String[] options = new String[6];
        Random rand = new Random();
        int seed = rand.nextInt(5000)+1;
        options[0] = "-N";
        options[1] = Integer.toString(5);
        options[2] = "-S";
        options[3] = Integer.toString(seed);
        options[4] = "-F";
        options[5] = Integer.toString(1);

        filter.setOptions(options);
        filter.setInputFormat(data);
        filter.setInvertSelection(false);

        Instances test_split = Filter.useFilter(data, filter);
        filter.setInvertSelection(true);
        Instances train_split = Filter.useFilter(data, filter);

        CourseworkTree tree = new CourseworkTree();


        // -------------------IG optdigits problem-------------------------------
        tree.setAttSplitMeasure(new IGAttributeSplitMeasure(true));
        tree.buildClassifier(train_split);


        int correct = 0;
        int total = 0;
        for (Instance instance:test_split){
            double prediction = tree.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }

        double accuracy = (double) correct/total;
        System.out.printf("DT using measure %s on optdigits problem has test accuracy = %f\n", "IG", accuracy);

        // ------------------Chi squared optdigits problem------------------------
        tree.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
        tree.buildClassifier(train_split);

        correct = 0;
        total = 0;
        for (Instance instance:test_split){
            double prediction = tree.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }

        accuracy = (double) correct/total;
        System.out.printf("DT using measure %s on optdigits problem has test accuracy = %f\n", "Chi", accuracy);

        // ----------------Gini index optdigits problem---------------------------
        tree.setAttSplitMeasure(new GiniAttributeSplitMeasure());
        tree.buildClassifier(train_split);

        correct = 0;
        total = 0;
        for (Instance instance:test_split){
            double prediction = tree.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }

        accuracy = (double) correct/total;
        System.out.printf("DT using measure %s on optdigits problem has test accuracy = %f\n", "Gini", accuracy);


        // --------------------China town problems-----------------------------
        reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff"));
        data = new Instances(reader);
        data.setClassIndex(data.numAttributes()-1);

        // https://stackoverflow.com/questions/28123954/how-do-i-divide-a-dataset-into-training-and-test-sets-using-weka
        filter = new StratifiedRemoveFolds();

        options = new String[6];

        options[0] = "-N";
        options[1] = Integer.toString(5);
        options[2] = "-S";
        options[3] = Integer.toString(seed);
        options[4] = "-F";
        options[5] = Integer.toString(1);

        filter.setOptions(options);
        filter.setInputFormat(data);
        filter.setInvertSelection(false);

        test_split = Filter.useFilter(data, filter);
        filter.setInvertSelection(true);
        train_split = Filter.useFilter(data, filter);

        tree = new CourseworkTree();


        // -------------------IG China town problem-------------------------------
        tree.setAttSplitMeasure(new IGAttributeSplitMeasure(true));
        tree.buildClassifier(train_split);

        correct = 0;
        total = 0;
        for (Instance instance:test_split){
            double prediction = tree.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }

        accuracy = (double) correct/total;
        System.out.printf("DT using measure %s on chinatown problem has test accuracy = %f\n", "IG", accuracy);

        // ------------------Chi squared china town problem------------------------
        tree.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
        tree.buildClassifier(train_split);

        correct = 0;
        total = 0;
        for (Instance instance:test_split){
            double prediction = tree.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }

        accuracy = (double) correct/total;
        System.out.printf("DT using measure %s on chinatown problem has test accuracy = %f\n", "Chi", accuracy);

        // ----------------Gini index china town problem---------------------------
        tree.setAttSplitMeasure(new GiniAttributeSplitMeasure());
        tree.buildClassifier(train_split);

        correct = 0;
        total = 0;
        for (Instance instance:test_split){
            double prediction = tree.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }

        accuracy = (double) correct/total;
        System.out.printf("DT using measure %s on chinatown problem has test accuracy = %f\n", "Gini", accuracy);

    }
}