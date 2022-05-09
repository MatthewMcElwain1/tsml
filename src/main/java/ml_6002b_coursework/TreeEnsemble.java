package ml_6002b_coursework;

import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.Remove;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


public class TreeEnsemble extends AbstractClassifier {

    private int numTrees;

    private double subset_percentage;

    CourseworkTree[] classifiers;

    int[][] att_selections;

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

        att_selections = new int[numTrees][(int) (data.numAttributes()*0.5)+1];

        int att_index;

        int[] atts_to_keep = new int[(int) (data.numAttributes()*0.5)+1];

        Remove removeFilter = new Remove();

        Instances[] folds = new Instances[numTrees];

        ArrayList<Integer> att_list = new ArrayList<>();

        for (int i = 0; i < numTrees; i++){
            for (int y =0; y < data.numAttributes()-1; y++){
                att_list.add(y);
            }

            for (int x = 0; x < (int)data.numAttributes()*0.5; x++){
                att_index = rand.nextInt(att_list.size()-1);
                atts_to_keep[x] = att_list.get(att_index);
                att_list.remove(att_index);
            }

            removeFilter.setAttributeIndicesArray(atts_to_keep);
            removeFilter.setInvertSelection(true);
            removeFilter.setInputFormat(data);
            folds[i] = Filter.useFilter(data, removeFilter);
            att_list.clear();
            att_selections[i] = atts_to_keep;
            atts_to_keep = new int[(int) (data.numAttributes()*0.5)+1];
        }
        






    }

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/test_data/optdigits.arff"));
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes()-1);

        TreeEnsemble ensemble = new TreeEnsemble();
        ensemble.buildClassifier(data);
    }
}
