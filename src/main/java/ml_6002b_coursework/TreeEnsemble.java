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

    private final int numTrees;

    private final double subset_percentage;

    ArrayList<CourseworkTree> classifiers = new ArrayList<CourseworkTree>();

    int[][] att_selections;

    Instances input_format;

    public TreeEnsemble(int numTrees, double subset_percentage){
        this.subset_percentage = subset_percentage;
        this.numTrees = numTrees;
    }

    public TreeEnsemble(){
        this.numTrees = 50;
        this.subset_percentage = 0.5;
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
        Hashtable<Double, Integer> votes = new Hashtable<Double, Integer>();


        instance.setDataset(input_format);
        double prediction;
        for (int i =0; i < classifiers.size(); i++){
//            Instance temp = instance.;
//            // System.out.println(att_selections.length);
//            for (int x = 0, y=1; x < att_selections.length; x++,y++){
//                temp.deleteAttributeAt(att_selections[i][x]);
//                System.out.println(temp.numAttributes());
//            }

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

        System.out.println(votes);
        return class_prediction;
    }

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/test_data/optdigits.arff"));
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes()-1);

        TreeEnsemble ensemble = new TreeEnsemble();
        ensemble.buildClassifier(data);
        System.out.println(ensemble.classifyInstance(data.get(0)));
    }
}
