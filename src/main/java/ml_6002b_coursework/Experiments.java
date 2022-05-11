package ml_6002b_coursework;

import core.contracts.Dataset;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class Experiments {

    public static void experiment_1() throws Exception {

        // -------------getting datasets into Instances format----------------------------------

        String[] discrete_dataset_names = DatasetLists.nominalAttributeProblems;
        String[] continuous_dataset_names = DatasetLists.continuousAttributeProblems;

        Instances[] discrete_datasets = new Instances[DatasetLists.nominalAttributeProblems.length];
        Instances[] continuous_datasets = new Instances[DatasetLists.continuousAttributeProblems.length];

        String discrete_file_location = "src/main/java/ml_6002b_coursework/UCI Discrete";
        String continous_file_location = "src/main/java/ml_6002b_coursework/UCI Continuous";

        String file_location;
        BufferedReader reader;
        Instances data;

        for (int i = 0; i < discrete_dataset_names.length; i++){
            file_location = String.format("%s/%s/%s.arff", discrete_file_location, discrete_dataset_names[i], discrete_dataset_names[i]);
            reader = new BufferedReader(new FileReader(file_location));
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);
            discrete_datasets[i] = data;
        }

        for (int i = 0; i < continuous_dataset_names.length; i++){
            file_location = String.format("%s/%s/%s.arff", continous_file_location, continuous_dataset_names[i], continuous_dataset_names[i]);
            reader = new BufferedReader(new FileReader(file_location));
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);
            continuous_datasets[i] = data;
        }

        // --------------making train test splits-------------------------------

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

        Instances[] discrete_train_splits = new Instances[discrete_datasets.length];
        Instances[] discrete_test_splits = new Instances[discrete_datasets.length];

        Instances[] continuous_train_splits = new Instances[continuous_datasets.length];
        Instances[] continuous_test_splits = new Instances[continuous_datasets.length];


        for (int i = 0; i < discrete_datasets.length; i++){
            filter.setInputFormat(discrete_datasets[i]);
            filter.setInvertSelection(false);
            discrete_test_splits[i] = Filter.useFilter(discrete_datasets[i], filter);
            filter.setInvertSelection(true);
            discrete_train_splits[i] = Filter.useFilter(discrete_datasets[i], filter);
        }

        for (int i = 0; i < continuous_datasets.length; i++){
            filter.setInputFormat(continuous_datasets[i]);
            filter.setInvertSelection(false);
            continuous_test_splits[i] = Filter.useFilter(continuous_datasets[i], filter);
            filter.setInvertSelection(true);
            continuous_train_splits[i] = Filter.useFilter(continuous_datasets[i], filter);
        }

        System.out.println(discrete_train_splits.length);
        System.out.println(discrete_test_splits.length);

        System.out.println(continuous_train_splits.length);
        System.out.println(continuous_test_splits.length);


    }

    public static void main(String[] args) throws Exception {
        Experiments.experiment_1();
    }
}
