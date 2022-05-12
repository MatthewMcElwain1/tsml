package ml_6002b_coursework;

import core.contracts.Dataset;
import org.checkerframework.checker.units.qual.C;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
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

       // --------setting up testing variables-----------

        CourseworkTree decision_tree;
        int correct;
        int total;
        double prediction;
        double accuracy;
        double[][] accuracies;
        double averaged_accuracy;

        // --------decision tree on discrete-----------

        accuracies = new double[4][discrete_datasets.length];
        ArrayList<Double> discrete_split_measure_accuracies = new ArrayList<Double>();
        decision_tree  = new CourseworkTree();

        for (int y = 0; y < 4; y++) {

            switch (y){
                case(0):
                    decision_tree.setAttSplitMeasure(new IGAttributeSplitMeasure(true));
                case(1):
                    decision_tree.setAttSplitMeasure(new IGAttributeSplitMeasure(false));
                case(2):
                    decision_tree.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
                default:
                    decision_tree.setAttSplitMeasure(new GiniAttributeSplitMeasure());
            }

            correct = 0;
            total = 0;

            for (int i = 0; i < discrete_train_splits.length; i++) {
                decision_tree.buildClassifier(discrete_train_splits[i]);

                for (int x = 0; x < discrete_test_splits[i].numInstances(); x++) {
                    prediction = decision_tree.classifyInstance(discrete_test_splits[i].get(x));
                    if (discrete_test_splits[i].get(x).classValue() == prediction) {
                        correct += 1;
                    }
                    total++;
                }
                accuracy = (double) correct / total;
                accuracies[y][i] = accuracy;
            }
        }

        for (double[] measure_accuracy:accuracies){
            averaged_accuracy = 0;

            for (double dataset_accuracy:measure_accuracy){
                averaged_accuracy += dataset_accuracy;
            }
            averaged_accuracy = averaged_accuracy/measure_accuracy.length;
            discrete_split_measure_accuracies.add(averaged_accuracy);
        }

        for (int i = 0; i < discrete_split_measure_accuracies.size(); i++){
            if (i == 0){
                System.out.printf("IG averaged accuracy on discrete= %f\n", discrete_split_measure_accuracies.get(i));
            }
            else if (i == 1){
                System.out.printf("IGR averaged accuracy on discrete = %f\n", discrete_split_measure_accuracies.get(i));
            }
            else if (i == 2){
                System.out.printf("Chi averaged accuracy on discrete = %f\n", discrete_split_measure_accuracies.get(i));
            }
            else{
                System.out.printf("Gini averaged accuracy on discrete = %f\n", discrete_split_measure_accuracies.get(i));
            }
        }

        // ------decision tree on continuous---------------------

        accuracies = new double[4][continuous_datasets.length];
        ArrayList<Double> continuous_split_measure_accuracies = new ArrayList<Double>();
        decision_tree  = new CourseworkTree();

        for (int y = 0; y < 4; y++) {

            switch (y){
                case(0):
                    decision_tree.setAttSplitMeasure(new IGAttributeSplitMeasure(true));
                case(1):
                    decision_tree.setAttSplitMeasure(new IGAttributeSplitMeasure(false));
                case(2):
                    decision_tree.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
                default:
                    decision_tree.setAttSplitMeasure(new GiniAttributeSplitMeasure());
            }

            correct = 0;
            total = 0;

            for (int i = 0; i < continuous_train_splits.length; i++) {
                decision_tree.buildClassifier(continuous_train_splits[i]);

                for (int x = 0; x < continuous_test_splits[i].numInstances(); x++) {
                    prediction = decision_tree.classifyInstance(continuous_test_splits[i].get(x));
                    if (continuous_test_splits[i].get(x).classValue() == prediction) {
                        correct += 1;
                    }
                    total++;
                }
                accuracy = (double) correct / total;
                accuracies[y][i] = accuracy;
            }
        }

        for (double[] measure_accuracy:accuracies){
            averaged_accuracy = 0;

            for (double dataset_accuracy:measure_accuracy){
                averaged_accuracy += dataset_accuracy;
            }
            averaged_accuracy = averaged_accuracy/measure_accuracy.length;
            continuous_split_measure_accuracies.add(averaged_accuracy);
        }

        for (int i = 0; i < continuous_split_measure_accuracies.size(); i++){
                if (i == 0){
                    System.out.printf("IG averaged accuracy on continuous = %f\n", continuous_split_measure_accuracies.get(i));
                }
                else if (i == 1){
                    System.out.printf("IGR averaged accuracy on continuous = %f\n", continuous_split_measure_accuracies.get(i));
                }
                else if (i == 2){
                    System.out.printf("Chi averaged accuracy on continuous = %f\n", continuous_split_measure_accuracies.get(i));
                }
                else{
                    System.out.printf("Gini averaged accuracy on continuous = %f\n", continuous_split_measure_accuracies.get(i));
                }
            }

        // ------------------weka ID3 classifier-------------------
        // --------used only on discrete as id3 doesnt work with continuous-----------------
        Id3 id3_tree = new Id3();

        correct = 0;
        total = 0;
        accuracy = 0;

        for (int i = 0; i < discrete_train_splits.length; i++) {
            id3_tree.buildClassifier(discrete_train_splits[i]);

            for (int x = 0; x < discrete_test_splits[i].numInstances(); x++) {
                prediction = id3_tree.classifyInstance(discrete_test_splits[i].get(x));
                if (discrete_test_splits[i].get(x).classValue() == prediction) {
                    correct += 1;
                }
                total++;
            }
            accuracy += ((double) correct / total)/discrete_test_splits.length;
        }

        System.out.printf("ID3 averaged accuracy on discrete = %f\n", accuracy);

        // ---------------weka j48 classifier----------------

        J48 j48_tree = new J48();

        correct = 0;
        total = 0;
        accuracy = 0;

        for (int i = 0; i < discrete_train_splits.length; i++) {
            j48_tree.buildClassifier(discrete_train_splits[i]);

            for (int x = 0; x < discrete_test_splits[i].numInstances(); x++) {
                prediction = j48_tree.classifyInstance(discrete_test_splits[i].get(x));
                if (discrete_test_splits[i].get(x).classValue() == prediction) {
                    correct += 1;
                }
                total++;
            }
            accuracy += ((double) correct / total)/discrete_test_splits.length;
        }

        System.out.printf("J48 averaged accuracy on discrete = %f\n", accuracy);

        correct = 0;
        total = 0;
        accuracy = 0;

        for (int i = 0; i < continuous_train_splits.length; i++) {
            j48_tree.buildClassifier(continuous_train_splits[i]);

            for (int x = 0; x < continuous_test_splits[i].numInstances(); x++) {
                prediction = j48_tree.classifyInstance(continuous_test_splits[i].get(x));
                if (continuous_test_splits[i].get(x).classValue() == prediction) {
                    correct += 1;
                }
                total++;
            }
            accuracy += ((double) correct / total)/continuous_test_splits.length;
        }

        System.out.printf("J48 averaged accuracy on continuous = %f\n", accuracy);



        }







    public static void main(String[] args) throws Exception {
        Experiments.experiment_1();
    }
}
