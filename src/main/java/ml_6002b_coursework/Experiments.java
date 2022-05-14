package ml_6002b_coursework;

import evaluation.tuning.ParameterResults;
import evaluation.tuning.ParameterSpace;
import evaluation.tuning.Tuner;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class Experiments {

    public static Instances[] prepare_discrete() throws IOException {
        // -------------getting datasets into Instances format----------------------------------

        String[] discrete_dataset_names = DatasetLists.nominalAttributeProblems;
        Instances[] discrete_datasets = new Instances[DatasetLists.nominalAttributeProblems.length];
        String discrete_file_location = "src/main/java/ml_6002b_coursework/UCI Discrete";

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

        return discrete_datasets;
    }

    public static Instances[] prepare_continuous() throws IOException {
        // -------------getting datasets into Instances format----------------------------------

        String[] continuous_dataset_names = DatasetLists.continuousAttributeProblems;
        Instances[] continuous_datasets = new Instances[DatasetLists.continuousAttributeProblems.length];
        String continous_file_location = "src/main/java/ml_6002b_coursework/UCI Continuous";

        String file_location;
        BufferedReader reader;
        Instances data;

        for (int i = 0; i < continuous_dataset_names.length; i++){
            file_location = String.format("%s/%s/%s.arff", continous_file_location, continuous_dataset_names[i], continuous_dataset_names[i]);
            reader = new BufferedReader(new FileReader(file_location));
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);
            continuous_datasets[i] = data;
        }

        return continuous_datasets;

    }

    public static Instances[][] prepare_testTrain_split() throws Exception {
        // --------------making train test splits-------------------------------

        Instances[] discrete_datasets = prepare_discrete();
        Instances[] continuous_datasets = prepare_continuous();

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

        Instances[][] test_train_splits = new Instances[4][];
        test_train_splits[0] = discrete_train_splits;
        test_train_splits[1] = discrete_test_splits;
        test_train_splits[2] = continuous_train_splits;
        test_train_splits[3] = continuous_test_splits;
        return test_train_splits;
    }

    public static void experiment_1() throws Exception {
       // --------setting up testing variables-----------

        CourseworkTree decision_tree;
        int correct;
        int total;
        double prediction;
        double accuracy;
        double[][] accuracies;
        double averaged_accuracy;


        Instances[][] test_train_splits = prepare_testTrain_split();

        Instances[] discrete_train_splits = test_train_splits[0];
        Instances[] discrete_test_splits = test_train_splits[1];
        Instances[] continuous_train_splits = test_train_splits[2];
        Instances[] continuous_test_splits = test_train_splits[3];

        // --------decision tree on discrete-----------

        accuracies = new double[4][discrete_train_splits.length];
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

        accuracies = new double[4][continuous_train_splits.length];
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


        public static void experiment_3() throws Exception {

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

            // ---------------- setting up variables for experiment-------------

            int correct;
            int total;
            double prediction;
            double accuracy;


            // --------------- coursework ensemble---------------

//            TreeEnsemble ensemble = new TreeEnsemble();
//
//            correct = 0;
//            total = 0;
//            accuracy = 0;
//
//            for (int i = 0; i < discrete_train_splits.length; i++) {
//                ensemble.buildClassifier(discrete_train_splits[i]);
//
//                for (int x = 0; x < discrete_test_splits[i].numInstances(); x++) {
//
//                    prediction = ensemble.classifyInstance(discrete_test_splits[i].get(x));
//
//                    if (discrete_test_splits[i].get(x).classValue() == prediction) {
//                        correct += 1;
//                    }
//                    total++;
//                }
//                accuracy += ((double) correct / total)/discrete_test_splits.length;
//            }
//
//            System.out.printf("Coursework ensemble averaged accuracy on discrete = %f\n", accuracy);
//
//            correct = 0;
//            total = 0;
//            accuracy = 0;
//
//            for (int i = 0; i < continuous_train_splits.length; i++) {
//                ensemble.buildClassifier(continuous_train_splits[i]);
//
//                for (int x = 0; x < continuous_test_splits[i].numInstances(); x++) {
//                    prediction = ensemble.classifyInstance(continuous_test_splits[i].get(x));
//                    if (continuous_test_splits[i].get(x).classValue() == prediction) {
//                        correct += 1;
//                    }
//                    total++;
//                }
//                accuracy += ((double) correct / total)/continuous_test_splits.length;
//            }
//
//            System.out.printf("Coursework ensemble averaged accuracy on continuous = %f\n", accuracy);


            CourseworkTree tree = new CourseworkTree();
            Tuner tuner = new Tuner();
            ParameterSpace params = new ParameterSpace();

            String[] splitMeasure_values = new String[4];
            splitMeasure_values[0] = "gain";
            splitMeasure_values[1] = "ratio";
            splitMeasure_values[2] = "gini";
            splitMeasure_values[3] = "chi";

            String[] maxDepth_values = new String[8];
            maxDepth_values[0] = "1";
            maxDepth_values[1] = "2";
            maxDepth_values[2] = "4";
            maxDepth_values[3] = "8";
            maxDepth_values[4] = "16";
            maxDepth_values[5] = "32";
            maxDepth_values[6] = "64";
            maxDepth_values[7] = "unlimited";

            params.addParameter("S", splitMeasure_values);
            params.addParameter("D", maxDepth_values);


            ParameterResults results = tuner.tune(tree, discrete_train_splits[2], params);
            System.out.println(results.paras);
//            options = new String[2];
//            options[0] = "S";
//            options[1] = "gain";
//
//
//            tree.setOptions(options);





//            for (Instances instances:discrete_train_splits){
//                tree.buildClassifier(instances);
//
//                for (int i = 0; i < instances.numAttributes(); i++){
//                    params.addParameter(String.format("attibute_%d", i), instances);
//                }
//                tuner.tune(tree, instances, params);
//
//            }
//
//            tuner.tune(tree, discrete_train_splits[0], params);

//
//            for (int i = 0; i < discrete_train_splits[0].numAttributes(); i++){
//                params.addParameter(String.format("attibute_%d", i), discrete_train_splits[0]);
//            }
//            tuner.tune(tree, discrete_train_splits[0], params);
//
//
//            tuner.tune(tree, discrete_train_splits[0], params);


        }






    public static void main(String[] args) throws Exception {
        Experiments.experiment_1();
        //Experiments.experiment_3();
    }
}
