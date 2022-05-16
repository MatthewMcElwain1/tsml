package ml_6002b_coursework;
import evaluation.tuning.ParameterResults;
import evaluation.tuning.ParameterSpace;
import evaluation.tuning.Tuner;
import experiments.ExperimentalArguments;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.tuned.TunedClassifier;
import org.apache.commons.math3.geometry.euclidean.threed.Rotation;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.*;
import weka.core.Instances;
import weka.core.matrix.LinearRegression;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.*;
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


        FileWriter results_file = new FileWriter("src/main/java/ml_6002b_coursework/experiment_results/experiment1.txt", true);
        results_file.write("cheese\n");
        results_file.write("crumpets\n");
        results_file.write("ya nan\n");
        results_file.close();



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
            long time = System.currentTimeMillis();;

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

            switch(y){
                case(0):
                    System.out.printf("IG time taken on discrete = %f\n", (double)(System.currentTimeMillis()-time)/1000);
                    break;
                case(1):
                    System.out.printf("Ratio time taken on discrete = %f\n", (double)(System.currentTimeMillis()-time)/1000);
                    break;
                case(2):
                    System.out.printf("Gini time taken on discrete = %f\n", (double)(System.currentTimeMillis()-time)/1000);
                    break;
                default:
                    System.out.printf("Chi time taken on discrete = %f\n", (double)(System.currentTimeMillis()-time)/1000);
                    break;
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
            long time = System.currentTimeMillis();

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

            switch(y){
                case(0):
                    System.out.printf("IG time taken on continuous = %f\n", (double)(System.currentTimeMillis()-time)/1000);
                    break;
                case(1):
                    System.out.printf("Ratio time taken on continuous = %f\n", (double)(System.currentTimeMillis()-time)/1000);
                    break;
                case(2):
                    System.out.printf("Gini time taken on continuous = %f\n", (double)(System.currentTimeMillis()-time)/1000);
                    break;
                default:
                    System.out.printf("Chi time taken on continuous = %f\n", (double)(System.currentTimeMillis()-time)/1000);
                    break;
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
        long time = System.currentTimeMillis();

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

        System.out.printf("ID3 time taken on discrete = %f\n", (double)(System.currentTimeMillis()-time)/1000);
        System.out.printf("ID3 averaged accuracy on discrete = %f\n", accuracy);

        // ---------------weka j48 classifier----------------

        J48 j48_tree = new J48();

        correct = 0;
        total = 0;
        accuracy = 0;
        time = System.currentTimeMillis();

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

        System.out.printf("J48 time taken on discrete = %f\n", (double)(System.currentTimeMillis()-time)/1000);
        System.out.printf("J48 averaged accuracy on discrete = %f\n", accuracy);

        correct = 0;
        total = 0;
        accuracy = 0;
        time = System.currentTimeMillis();

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

        System.out.printf("J48 time taken on continuous = %f\n", (double)(System.currentTimeMillis()-time)/1000);
        System.out.printf("J48 averaged accuracy on continuous = %f\n", accuracy);




        }


        public static void experiment_2() throws Exception {
            // ---------------- setting up variables for experiment-------------

            int correct;
            int total;
            double prediction;
            double accuracy;

            Instances[][] test_train_splits = prepare_testTrain_split();

            Instances[] discrete_train_splits = test_train_splits[0];
            Instances[] discrete_test_splits = test_train_splits[1];
            Instances[] continuous_train_splits = test_train_splits[2];
            Instances[] continuous_test_splits = test_train_splits[3];


            // --------------- coursework ensemble---------------

            TreeEnsemble ensemble = new TreeEnsemble();

            correct = 0;
            total = 0;
            accuracy = 0;

            for (int i = 0; i < discrete_train_splits.length; i++) {
                ensemble.buildClassifier(discrete_train_splits[i]);

                for (int x = 0; x < discrete_test_splits[i].numInstances(); x++) {

                    prediction = ensemble.classifyInstance(discrete_test_splits[i].get(x));

                    if (discrete_test_splits[i].get(x).classValue() == prediction) {
                        correct += 1;
                    }
                    total++;
                }
                accuracy += ((double) correct / total)/discrete_test_splits.length;
            }

            System.out.printf("Coursework ensemble averaged accuracy on discrete = %f\n", accuracy);

            correct = 0;
            total = 0;
            accuracy = 0;

            for (int i = 0; i < continuous_train_splits.length; i++) {
                ensemble.buildClassifier(continuous_train_splits[i]);

                for (int x = 0; x < continuous_test_splits[i].numInstances(); x++) {
                    prediction = ensemble.classifyInstance(continuous_test_splits[i].get(x));
                    if (continuous_test_splits[i].get(x).classValue() == prediction) {
                        correct += 1;
                    }
                    total++;
                }
                accuracy += ((double) correct / total)/continuous_test_splits.length;
            }

            System.out.printf("Coursework ensemble averaged accuracy on continuous = %f\n", accuracy);


            CourseworkTree tree = new CourseworkTree();
            Tuner tuner = new Tuner();
            ParameterSpace params = new ParameterSpace();
            ParameterResults results;

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

            correct = 0;
            total = 0;
            accuracy = 0;

            String[] options = new String[4];
            options[0] = "S";
            options[2] = "D";

            for (int i = 0; i < discrete_train_splits.length; i++) {
                results = tuner.tune(tree, discrete_train_splits[i], params);
                options[1] = results.paras.getParameterValue("S");
                options[3] = results.paras.getParameterValue("D");
                tree.setOptions(options);
                tree.buildClassifier(discrete_train_splits[i]);

                for (int x = 0; x < discrete_test_splits[i].numInstances(); x++) {
                    prediction = tree.classifyInstance(discrete_test_splits[i].get(x));
                    if (discrete_test_splits[i].get(x).classValue() == prediction) {
                        correct += 1;
                    }
                    total++;
                }
                accuracy += ((double) correct / total)/discrete_test_splits.length;
            }

            System.out.printf("Tuned tree averaged accuracy on discrete = %f\n", accuracy);

            correct = 0;
            total = 0;
            accuracy = 0;

            for (int i = 0; i < continuous_train_splits.length; i++) {
                results = tuner.tune(tree, continuous_train_splits[i], params);
                options[1] = results.paras.getParameterValue("S");
                options[3] = results.paras.getParameterValue("D");
                tree.setOptions(options);
                tree.buildClassifier(continuous_train_splits[i]);

                for (int x = 0; x < continuous_test_splits[i].numInstances(); x++) {
                    prediction = tree.classifyInstance(continuous_test_splits[i].get(x));
                    if (continuous_test_splits[i].get(x).classValue() == prediction) {
                        correct += 1;
                    }
                    total++;
                }
                accuracy += ((double) correct / total)/continuous_test_splits.length;
            }

            System.out.printf("Tuned tree averaged accuracy on continuous = %f\n", accuracy);

        }

        public static void run_experiment_4_normal() throws IOException {
            BufferedReader reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/case_study/EOGVerticalSignal/EOGVerticalSignal_TRAIN.arff"));
            Instances train = new Instances(reader);
            reader = new BufferedReader(new FileReader("src/main/java/ml_6002b_coursework/case_study/EOGVerticalSignal/EOGVerticalSignal_TEST.arff"));
            Instances test = new Instances(reader);



//            experiments.ExperimentalArguments expSettings = new ExperimentalArguments();
//            Classifier[] cls = new Classifier[1];
//
//            String[] names = {"Rotf"};
//
//            cls[0] = new RotationForest();
//
//            expSettings.dataReadLocation = "src/main/java/ml_6002b_coursework/case_study/EOGVerticalSignal/";
//            expSettings.resultsWriteLocation = "src/main/java/ml_6002b_coursework/experiment_results/experiment_4/normal";
//            expSettings.forceEvaluation = false;
//            expSettings.numberOfThreads = 7;
//
//            DatasetLoading.setProportionKeptForTraining(0.8);
//            for (int i = 0; i < cls.length; i++){
//                expSettings.classifier = cls[i];
//                expSettings.estimatorName = names[i];
//                for (int x = 0; x < 5; x++) {
//                    expSettings.foldId = x;
//                    expSettings.run();
//
//                }
//            }
        }

        public static void run_experiment_3_discrete(){
            experiments.ExperimentalArguments expSettings = new ExperimentalArguments();
            Classifier[] cls = new Classifier[7];

            String[] names = {"CWE", "Randf", "Rotf", "J48", "LADTree", "DStump", "NBayes"};


            TreeEnsemble CWE = new TreeEnsemble();
            RandomForest Randf = new RandomForest();
            RotationForest Rotf = new RotationForest();
            J48 J48 = new J48();
            LADTree LADTree = new LADTree();
            DecisionStump DStump = new DecisionStump();
            NaiveBayes NBayes = new NaiveBayes();

            cls[0] = DStump;
            cls[1] = NBayes;
            cls[2] = J48;
            cls[3] = LADTree;
            cls[4] = CWE;
            cls[5] = Randf;
            cls[6] = Rotf;

            expSettings.dataReadLocation = "src/main/java/ml_6002b_coursework/UCI Discrete";
            expSettings.resultsWriteLocation = "src/main/java/ml_6002b_coursework/experiment_results/experiment_3/discrete_results/";
            expSettings.forceEvaluation = false;
            expSettings.numberOfThreads = 7;

            DatasetLoading.setProportionKeptForTraining(0.8);
            for (int i = 0; i < cls.length; i++){
                expSettings.classifier = cls[i];
                expSettings.estimatorName = names[i];
                for (int x = 0; x < 5; x++) {
                    for (String str : DatasetLists.nominalAttributeProblems) {
                        expSettings.datasetName = str;
                        expSettings.foldId = x;
                        expSettings.run();
                    }
                }
            }
        }

        public static void run_experiment_2_discrete_halfAttr(){
            experiments.ExperimentalArguments expSettings = new ExperimentalArguments();
            Classifier[] cls = new Classifier[2];

            String[] names = {"Tuned_CWT", "CWE"};

            TunedClassifier Tuned_CWT = new TunedClassifier();
            TreeEnsemble CWE = new TreeEnsemble();

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
            Tuned_CWT.setClassifier(new CourseworkTree());
            Tuned_CWT.setParameterSpace(params);

            cls[0] = Tuned_CWT;
            cls[1] = CWE;
            expSettings.dataReadLocation = "src/main/java/ml_6002b_coursework/UCI Discrete";
            expSettings.resultsWriteLocation = "src/main/java/ml_6002b_coursework/experiment_results/experiment_2/0.5_attributes/discrete_results/";
            expSettings.forceEvaluation = false;
            expSettings.numberOfThreads = 7;

            DatasetLoading.setProportionKeptForTraining(0.8);
            for (int i = 0; i < cls.length; i++){
                expSettings.classifier = cls[i];
                expSettings.estimatorName = names[i];
                for (int x = 0; x < 5; x++) {
                    for (String str : DatasetLists.nominalAttributeProblems) {
                        expSettings.datasetName = str;
                        expSettings.foldId = x;
                        expSettings.run();
                    }
                }
            }
        }

    public static void run_experiment_2_discrete_fullAttr(){
        experiments.ExperimentalArguments expSettings = new ExperimentalArguments();
        Classifier[] cls = new Classifier[2];

        String[] names = {"Tuned_CWT", "CWE"};

        TunedClassifier Tuned_CWT = new TunedClassifier();
        TreeEnsemble CWE = new TreeEnsemble();
        CWE.setSubset_percentage(1.0);

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
        Tuned_CWT.setClassifier(new CourseworkTree());
        Tuned_CWT.setParameterSpace(params);

        cls[0] = CWE;
        cls[1] = Tuned_CWT;
        expSettings.dataReadLocation = "src/main/java/ml_6002b_coursework/UCI Discrete";
        expSettings.resultsWriteLocation = "src/main/java/ml_6002b_coursework/experiment_results/experiment_2/1.0_attributes/discrete_results/";
        expSettings.forceEvaluation = false;
        expSettings.numberOfThreads = 7;

        DatasetLoading.setProportionKeptForTraining(0.8);
        for (int i = 0; i < cls.length; i++){
            expSettings.classifier = cls[i];
            expSettings.estimatorName = names[i];
            for (int x = 0; x < 5; x++) {
                for (String str : DatasetLists.nominalAttributeProblems) {
                    expSettings.datasetName = str;
                    expSettings.foldId = x;
                    expSettings.run();
                }
            }
        }
    }

    public static void run_experiment_2_continuous_fullAttr(){
        experiments.ExperimentalArguments expSettings = new ExperimentalArguments();
        Classifier[] cls = new Classifier[2];

        String[] names = {"Tuned_CWT", "CWE"};

        TunedClassifier Tuned_CWT = new TunedClassifier();
        TreeEnsemble CWE = new TreeEnsemble();
        CWE.setSubset_percentage(1.0);

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
        Tuned_CWT.setClassifier(new CourseworkTree());
        Tuned_CWT.setParameterSpace(params);

        cls[0] = CWE;
        cls[1] = Tuned_CWT;
        expSettings.dataReadLocation = "src/main/java/ml_6002b_coursework/UCI Continuous";
        expSettings.resultsWriteLocation = "src/main/java/ml_6002b_coursework/experiment_results/experiment_2/1.0_attributes/continuous_results/";
        expSettings.forceEvaluation = false;
        expSettings.numberOfThreads = 7;

        DatasetLoading.setProportionKeptForTraining(0.8);
        for (int i = 0; i < cls.length; i++){
            expSettings.classifier = cls[i];
            expSettings.estimatorName = names[i];
            for (int x = 0; x < 5; x++) {
                for (String str : DatasetLists.continuousAttributeProblems) {
                    expSettings.datasetName = str;
                    expSettings.foldId = x;
                    expSettings.run();
                }
            }
        }
    }

    public static void run_experiment_2_continuous_halfAttr(){
        experiments.ExperimentalArguments expSettings = new ExperimentalArguments();
        Classifier[] cls = new Classifier[2];

        String[] names = {"Tuned_CWT", "CWE"};

        TunedClassifier Tuned_CWT = new TunedClassifier();
        TreeEnsemble CWE = new TreeEnsemble();

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
        Tuned_CWT.setClassifier(new CourseworkTree());
        Tuned_CWT.setParameterSpace(params);

        cls[0] = CWE;
        cls[1] = Tuned_CWT;
        expSettings.dataReadLocation = "src/main/java/ml_6002b_coursework/UCI Continuous";
        expSettings.resultsWriteLocation = "src/main/java/ml_6002b_coursework/experiment_results/experiment_2/0.5_attributes/continuous_results/";
        expSettings.forceEvaluation = false;
        expSettings.numberOfThreads = 7;

        DatasetLoading.setProportionKeptForTraining(0.8);
        for (int i = 0; i < cls.length; i++){
            expSettings.classifier = cls[i];
            expSettings.estimatorName = names[i];
            for (int x = 0; x < 5; x++) {
                for (String str : DatasetLists.continuousAttributeProblems) {
                    expSettings.datasetName = str;
                    expSettings.foldId = x;
                    expSettings.run();
                }
            }
        }
    }

        public static void run_experiment_1_continuous(){
            experiments.ExperimentalArguments expSettings = new ExperimentalArguments();
            Classifier[] cls = new Classifier[5];

            String[] names = {"CWT_ig", "CWT_ratio", "CWT_gini", "CWT_chi", "J48"};

            CourseworkTree CWT_ig = new CourseworkTree();
            CWT_ig.setAttSplitMeasure(new IGAttributeSplitMeasure(true));

            CourseworkTree CWT_ratio = new CourseworkTree();
            CWT_ratio.setAttSplitMeasure(new IGAttributeSplitMeasure(false));

            CourseworkTree CWT_gini = new CourseworkTree();
            CWT_gini.setAttSplitMeasure(new GiniAttributeSplitMeasure());

            CourseworkTree CWT_chi = new CourseworkTree();
            CWT_chi.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());

            J48 J48 = new J48();

            cls[0] = CWT_ig;
            cls[1] = CWT_ratio;
            cls[2] = CWT_gini;
            cls[3] = CWT_chi;
            cls[4] = J48;

            expSettings.dataReadLocation = "src/main/java/ml_6002b_coursework/UCI Continuous";
            expSettings.resultsWriteLocation = "src/main/java/ml_6002b_coursework/experiment_results/experiment_1/continuous_results/";
            expSettings.forceEvaluation = false;
            expSettings.numberOfThreads = 7;

            DatasetLoading.setProportionKeptForTraining(0.8);
            for (int i = 0; i < cls.length; i++){
                expSettings.classifier = cls[i];
                expSettings.estimatorName = names[i];
                for (int x = 0; x < 5; x++) {
                    for (String str : DatasetLists.continuousAttributeProblems) {
                        expSettings.datasetName = str;
                        expSettings.foldId = x;
                        expSettings.run();
                    }
                }
        }
    }

        public static void run_experiment_1_discrete(){
            experiments.ExperimentalArguments expSettings = new ExperimentalArguments();
            Classifier[] cls = new Classifier[6];

            String[] names = {"CWT_ig", "CWT_ratio", "CWT_gini", "CWT_chi", "ID3", "J48"};

            CourseworkTree CWT_ig = new CourseworkTree();
            CWT_ig.setAttSplitMeasure(new IGAttributeSplitMeasure(true));

            CourseworkTree CWT_ratio = new CourseworkTree();
            CWT_ratio.setAttSplitMeasure(new IGAttributeSplitMeasure(false));

            CourseworkTree CWT_gini = new CourseworkTree();
            CWT_gini.setAttSplitMeasure(new GiniAttributeSplitMeasure());

            CourseworkTree CWT_chi = new CourseworkTree();
            CWT_chi.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());

            Id3 ID3 = new Id3();
            J48 J48 = new J48();

            cls[0] = CWT_ig;
            cls[1] = CWT_ratio;
            cls[2] = CWT_gini;
            cls[3] = CWT_chi;
            cls[4] = ID3;
            cls[5] = J48;

            expSettings.dataReadLocation = "src/main/java/ml_6002b_coursework/UCI Discrete";
            expSettings.resultsWriteLocation = "src/main/java/ml_6002b_coursework/experiment_results/experiment_1/discrete_results/";
            expSettings.forceEvaluation = false;
            expSettings.numberOfThreads = 7;

            DatasetLoading.setProportionKeptForTraining(0.8);
            for (int i = 0; i < cls.length; i++){
                expSettings.classifier = cls[i];
                expSettings.estimatorName = names[i];
                for (int x = 0; x < 5; x++) {
                    for (String str : DatasetLists.nominalAttributeProblems) {
                        expSettings.datasetName = str;
                        expSettings.foldId = x;
                        expSettings.run();
                    }
                }
            }





        }




    public static void main(String[] args) throws Exception {
        // built in tsml experiments use resampling to split the data
        Experiments.run_experiment_3_discrete();

    }
}
