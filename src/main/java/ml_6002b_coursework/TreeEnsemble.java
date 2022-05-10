package ml_6002b_coursework;

import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SingleIndex;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.Remove;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;


public class TreeEnsemble extends AbstractClassifier {

    private int numTrees;

    private boolean averageDistributions;

    private double subset_percentage;

    ArrayList<CourseworkTree> classifiers = new ArrayList<CourseworkTree>();

    int[][] att_selections;

    Instances input_format;

    public TreeEnsemble(){
        this.numTrees = 50;
        this.subset_percentage = 0.5;
        this.averageDistributions = false;
    }

    public void setNumTrees(int numTrees){
        this.numTrees = numTrees;
    }

    public void setAverageDistributions(boolean averageDistributions){
        this.averageDistributions = averageDistributions;
    }

    public void setSubset_percentage(double subset_percentage){
        this.subset_percentage = subset_percentage;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // creating random subsets and removing random attributes
        Random rand = new Random();

        input_format = data;
        att_selections = new int[numTrees][(int) (data.numAttributes()*subset_percentage)+1];

        int att_index;

        int[] atts_to_keep = new int[(int) (data.numAttributes()*subset_percentage)+1];

        Remove removeFilter = new Remove();

        Instances[] folds = new Instances[numTrees];

        ArrayList<Integer> att_list = new ArrayList<>();

        for (int i = 0; i < numTrees; i++){
            for (int y =0; y < data.numAttributes()-1; y++){
                att_list.add(y);
            }

            for (int x = 0; x < (int)data.numAttributes()*subset_percentage; x++){
                att_index = rand.nextInt(att_list.size()-1);
                atts_to_keep[x] = att_list.get(att_index);
                att_list.remove(att_index);
            }

            atts_to_keep[atts_to_keep.length-1] = data.numAttributes()-1;
            removeFilter.setAttributeIndicesArray(atts_to_keep);
            removeFilter.setInvertSelection(true);
            removeFilter.setInputFormat(data);
            folds[i] = Filter.useFilter(data, removeFilter);
            att_list.clear();
            att_selections[i] = atts_to_keep;
            atts_to_keep = new int[(int) (data.numAttributes()*subset_percentage)+1];
        }

        for (Instances fold:folds){
            CourseworkTree tree = new CourseworkTree();
            switch(rand.nextInt(3)+1){
                case 1:
                    tree.setAttSplitMeasure(new IGAttributeSplitMeasure(true));
                    break;
                case 2:
                    tree.setAttSplitMeasure(new IGAttributeSplitMeasure(false));
                    break;
                case 3:
                    tree.setAttSplitMeasure(new GiniAttributeSplitMeasure());
                    break;
                default:
                    tree.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
                    break;
            }
            tree.buildClassifier(fold);
            classifiers.add(tree);
        }
    }

    public double classifyInstance(Instance instance) throws Exception {

        if (averageDistributions){
            double[] probabilities = distributionForInstance(instance);
            int highest_probability_class = 0;
            double probability_temp_holder = 0;

            for (int x=0; x < probabilities.length; x++){
                if (probabilities[x] > probability_temp_holder){
                    probability_temp_holder = probabilities[x];
                    highest_probability_class = x;
                }
            }
            return highest_probability_class;

        }
        else{
        Hashtable<Double, Integer> votes = new Hashtable<Double, Integer>();
        instance.setDataset(input_format);
        double prediction;
        for (int i =0; i < classifiers.size(); i++){
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(att_selections[i]);
            removeFilter.setInvertSelection(true);
            removeFilter.setInputFormat(input_format);
            removeFilter.input(instance);
            Instance temp = removeFilter.output();

            prediction = classifiers.get(i).classifyInstance(temp);
            try{
                votes.put(prediction, votes.get(prediction)+1);
            }catch(Exception exception){
                votes.put(prediction, 1);
            }
        }

        Set<Double> classes = votes.keySet();
        int majority_vote = 0;
        double class_prediction = 0.0;
        for(Double class_attr:classes){
            if (votes.get(class_attr) >= majority_vote){
                class_prediction = class_attr;
                majority_vote = votes.get(class_attr);
            }
        }

        return class_prediction;
    }
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        double[][] test = new double[classifiers.size()][];
        instance.setDataset(input_format);

        for (int i = 0; i < classifiers.size(); i++){
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(att_selections[i]);
            removeFilter.setInvertSelection(true);
            removeFilter.setInputFormat(input_format);
            removeFilter.input(instance);
            Instance temp = removeFilter.output();


            test[i] = classifiers.get(i).distributionForInstance(temp);
        }

        double[] probabilities = new double[instance.numClasses()];

        for (double[] doubles : test) {
            for (int y = 0; y < doubles.length; y++) {
                probabilities[y] += (doubles[y] / numTrees);
            }
        }
        return probabilities;
    }

    public static void main(String[] args) throws Exception {
        // ----------------ensemble optdigits problem---------------
        BufferedReader reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/test_data/optdigits.arff"));
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes()-1);

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

        TreeEnsemble ensemble = new TreeEnsemble();
        ensemble.buildClassifier(train_split);
        ensemble.setAverageDistributions(true);

        int correct = 0;
        int total = 0;

        for (int i=0; i < 5; i++){
            System.out.println(Arrays.toString(ensemble.distributionForInstance(test_split.get(i))));
        }
        for (Instance instance:test_split){
            double prediction = ensemble.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }

        double accuracy = (double) correct/total;
        System.out.printf("Ensemble on optdigits problem has test accuracy = %f\n", accuracy);

        // ----------------ensemble chinatown problem------------

        reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff"));
        data = new Instances(reader);
        data.setClassIndex(data.numAttributes()-1);

        filter = new StratifiedRemoveFolds();

        options = new String[6];
        rand = new Random();
        seed = rand.nextInt(5000)+1;
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

        ensemble = new TreeEnsemble();
        ensemble.buildClassifier(train_split);
        ensemble.setAverageDistributions(true);

        correct = 0;
        total = 0;

        for (int i=0; i < 5; i++){
            System.out.println(Arrays.toString(ensemble.distributionForInstance(test_split.get(i))));
        }
        for (Instance instance:test_split){
            double prediction = ensemble.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }

        accuracy = (double) correct/total;
        System.out.printf("Ensemble on chinatown problem has test accuracy = %f\n", accuracy);

    }
}
