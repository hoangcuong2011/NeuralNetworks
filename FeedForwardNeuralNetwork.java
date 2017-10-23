
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cnnsimplementation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import mnistreader.MNISTImageFile;
import mnistreader.MNISTLabelFile;

/**
 *
 * @author hoangcuong2011
 */
public class FeedForwardNeuralNetwork {


    public void CalculateSignalErrorsCrossEntropy(double hs[][][], double ys[][], double[][][] wxhs, int hidden_units[],
            int trainingSamples, int total_layers, double[][][] signalErrors, 
            double[][][] wxhs_dropout, double dropout_rate) {
        for (int layer = total_layers - 1; layer >= 1; layer--) {
            if (layer == total_layers - 1) {
                signalErrors[layer] = new double[trainingSamples][hidden_units[layer]];
                double[][] hs_layer = hs[layer];
                for (int trial = 0; trial < trainingSamples; trial++) {
                    for (int j = 0; j < hidden_units[layer]; j++) {
                        signalErrors[layer][trial][j] = hs_layer[trial][j]-ys[trial][j];
                    }
                }
            } else {
                signalErrors[layer] = new double[trainingSamples][hidden_units[layer]];
                double[][] hs_layer = hs[layer];
                double[][] wxhs_layer = wxhs[layer];

                for (int trial = 0; trial < trainingSamples; trial++) {
                    for (int j = 0; j < hidden_units[layer]; j++) {
                        double sum = 0.0;
                        for (int l = 0; l < hidden_units[layer + 1]; l++) {
                            sum += signalErrors[layer + 1][trial][l]
                                    * wxhs_layer[l][j]*wxhs_dropout[layer][l][j]*dropout_rate;
                        }
                        signalErrors[layer][trial][j] = sum * hs_layer[trial][j] * (1.0 - hs_layer[trial][j]);
                    }
                }
            }
        }
    }

    public void updateWeights(double hs[][][], double[][][] wxhs, double[][] biases, int hidden_units[],
            int trainingSamples, int total_layers, double[][][] signalErrors, double wxhs_update[][][],
            double[][] biases_update,
            double constant, double[][][] wxhs_dropout, double dropout_rate, double[][][] cache, double lambda) {
        for (int layer = total_layers - 2; layer >= 0; layer--) {
            double[][] wxhs_layer = wxhs[layer];
            double hs_layer[][] = hs[layer];
            double wxhs_update_layer[][] = wxhs_update[layer];
            for (int i = 0; i < hidden_units[layer]; i++) {
                for (int j = 0; j < hidden_units[layer + 1]; j++) {
                    double d = 0.0;
                    for (int trial = 0; trial < trainingSamples; trial++) {
                        d += signalErrors[layer + 1][trial][j] * hs_layer[trial][i];
                    }
                    cache[layer][i][j] = cache[layer][i][j]*lambda+constant * d;
                    
                    if (wxhs_dropout[layer][j][i] > 0) {
                        wxhs_update_layer[j][i] = wxhs_layer[j][i] - cache[layer][i][j];
                    } else {
                        wxhs_update_layer[j][i] = wxhs_layer[j][i];
                    }
                }
            }
        }
    }

    public double gradientChecking(int samples, double xs[][], double ys[][], int hidden_units[],
            int activationtype, int computationalTYPE, double[][][] hs) {
        int total_layers = hidden_units.length;
        double d = 0;
        for (int trial = 0; trial < samples; trial++) {
            int max = -1;
            double dmax = 0.0;
            for (int i = 0; i < hidden_units[total_layers - 1]; i++) {
                if (ys[trial][i] > dmax) {
                    dmax = hs[total_layers - 1][trial][i];
                    max = i;
                }
            }
            d -= Math.log(hs[total_layers - 1][trial][max]);
        }
        return d;
    }

    public double getCrossEntropy(int samples, double xs[][], double ys[][], int hidden_units[], double[][][] hs) throws IOException {
        int total_layers = hidden_units.length;

        double d = 0;

        for (int trial = 0; trial < samples; trial++) {
            int max = -1;
            double dmax = 0.0;
            for (int i = 0; i < hidden_units[total_layers - 1]; i++) {
                if (ys[trial][i] > dmax) {
                    dmax = hs[total_layers - 1][trial][i];
                    max = i;
                }
            }
            d -= Math.log(hs[total_layers - 1][trial][max]);
        }

        return d;

    }
    
    public double getClassificationError(int samples, double xs[][], double ys[][], int hidden_units[], double[][][] hs) throws IOException {
        int total_layers = hidden_units.length;


        double d1 = 0;
        double d2 = 0;
        for (int trial = 0; trial < samples; trial++) {
            int max = -1;
            double dmax = 0;
            for (int i = 0; i < hidden_units[total_layers - 1]; i++) {
                if (hs[total_layers - 1][trial][i] > dmax) {
                    dmax = hs[total_layers - 1][trial][i];
                    max = i;
                }
            }
            try {
                if (ys[trial][max] != 1.0) {
                    d1++;
                }
                if (ys[trial][max] == 1.0) {
                    d2++;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return d2/(d1+d2)*100;
    }

    public double[][][] computeForward(int total_layers, int trainingSamples, int hidden_units[],
            double xs_shuffle[][], double[][][] wxhs, double[][] bias,
            boolean dropout, double[][][] wxhs_dropout, double dropout_rate) {
        double hs[][][] = new double[total_layers][][];
        
        for (int k = 0; k < total_layers; k++) {
            if (k == 0) {
                hs[0] = new double[trainingSamples][hidden_units[0]];
                for (int trial = 0; trial < trainingSamples; trial++) {
                    for (int j = 0; j < hidden_units[0]; j++) {
                        hs[0][trial][j] = xs_shuffle[trial][j];
                    }
                }
            } else {
                hs[k] = new double[trainingSamples][hidden_units[k]];
            }
        }

        for (int k = 1; k < total_layers; k++) {
            double[][] hs_layer_k = hs[k];
            if (k < (total_layers - 1)) {
                for (int trial = 0; trial < trainingSamples; trial++) {
                    for (int i = 0; i < hidden_units[k]; i++) {
                        if(dropout==false) {
                            hs_layer_k[trial][i] = NonLinearFunctions.sigmoid(
                                hs[k - 1][trial], wxhs[k - 1][i]);
                        } else {
                            hs_layer_k[trial][i] = NonLinearFunctions.sigmoid(
                                hs[k - 1][trial], 
                                NonLinearFunctions.matrixmul(wxhs[k - 1][i], wxhs_dropout[k - 1][i], dropout_rate));
                        }
                    }
                }
            } else {//The last layer
                for (int trial = 0; trial < trainingSamples; trial++) {
                    for (int i = 0; i < hidden_units[k]; i++) {
                        for (int j = 0; j < wxhs[k - 1][i].length; j++) {
                            if(dropout==false) {
                                hs_layer_k[trial][i] += hs[k - 1][trial][j] * wxhs[k - 1][i][j];
                            } else {
                                hs_layer_k[trial][i] += hs[k - 1][trial][j] * wxhs[k - 1][i][j]
                                        * wxhs_dropout[k - 1][i][j] * dropout_rate;
                            }
                        }
                    }
                    Utilities.computedSoftMaxwithTricks(hs_layer_k[trial]);
                }
            }
        }
        return hs;
    }

    public double generateNumber(double rangeMin, double rangeMax) {
        Random r = new Random();
        double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
        return randomValue;
    }
    Random random = new Random();   
    public boolean getRandomBoolean(float p) {
        return random.nextFloat() < p;
    }

    public void backpropagationNLayers(double xs_training[][], double ys_training[][], 
            int totaltrainingSamples,
            double xs_dev[][], double ys_dev[][], int totaldevSamples, 
            double xs_test[][], double ys_test[][], int totaltestSamples, 
            int hidden_units[], boolean dropout) throws IOException {
        

        Random r = new Random();
        int total_layers = hidden_units.length;

        double lambda = 0.01;
        double dropout_keep_rate = 0.5;
        
        double cutoff = dropout_keep_rate;

        double wxhs[][][] = new double[total_layers - 1][][];
        double wxhs_dropout[][][] = new double[total_layers - 1][][];
        /*a 3-D array: this represents the weights for connecting all neurals in all layers
        D1: number of layers (ALL layers - 1)
        D2: number of neurons in layer + 1
        D3: number of neurons of the layer current
         */
        Utilities.initializationWeights(wxhs, total_layers, hidden_units);
        Utilities.initializationWeights(wxhs_dropout, total_layers, hidden_units);

        double biases[][] = new double[total_layers - 1][];
        if (1 == 1) {
            for (int k = 0; k < total_layers - 1; k++) {
                biases[k] = new double[hidden_units[k + 1]];
                for (int i = 0; i < biases[k].length; i++) {
                    biases[k][i] = r.nextDouble() - 0.5;
                }
            }
        }
        int total_iterations = 30000;
        int minibatch = 20;
        
        double[][][] cache = new double[total_layers][][];
        for (int layer = total_layers - 2; layer >= 0; layer--) {
            cache[layer] = new double[hidden_units[layer]][hidden_units[layer + 1]];
        }
        
        double gamma = 0.5;
        
        
        boolean[][][] dropoutMatrices = new boolean[total_layers][total_iterations][];
        for (int i = 0; i < total_layers; i++) {
        
            //int pseudo_cutoffs = (int) (hidden_units[i] * cutoff);
            dropoutMatrices[i] = new boolean[total_iterations][hidden_units[i]];
            
            for(int j = 0; j < total_iterations; j++) {
                for(int k = 0; k < hidden_units[i]; k++ ) {
                    dropoutMatrices[i][j][k] = true;
                    if(i==0 || i==(total_layers-1)) {
                        dropoutMatrices[i][j][k] = false; //I dont' do dropout for inputs and outputs
                    }
                }
            }
            if(i==0 || i==(total_layers-1)) {
                continue;
            }
            Heuristic program = new Heuristic(hidden_units[i],
                    total_iterations,
                    (int) (hidden_units[i] * dropout_keep_rate));
            int[][] outputs = program.select();
            for (int k = 0; k < outputs.length; k++) {
                for (int j = 0; j < outputs[k].length; j++) {
                    //System.out.print(" " + outputs[k][j]);
                }
                //System.out.println();
            }
            for(int j = 0; j < total_iterations; j++) {
                for(int k = 0; k < outputs[j].length; k++ ) {
                    dropoutMatrices[i][j][outputs[j][k]] = false;//ok I don't do dropout
                }
            }
        }
        
        for (int iteration = 0; iteration < total_iterations; iteration++) {
            
            //sampling with replacement
            double xs_shuffle[][] = new double[minibatch][];
            double ys_shuffle[][] = new double[minibatch][];
            for (int i = 0; i < minibatch; i++) {
                int id = r.nextInt(totaltrainingSamples);
                xs_shuffle[i] = (xs_training[id]);
                ys_shuffle[i] = (ys_training[id]);
            }
            
            double constant = lambda / minibatch;

            boolean[][] dropoutMatrix = new boolean[total_layers][];
            
            for (int k = 0; k < total_layers; k++) {
                dropoutMatrix[k] = dropoutMatrices[k][iteration];
            }
            for (int k = 1; k < total_layers; k++) {
                for (int i = 0; i < hidden_units[k]; i++) {
                    for (int j = 0; j < wxhs[k - 1][i].length; j++) {
                        if (dropoutMatrix[k - 1][j] == true || dropoutMatrix[k][i] == true) {
                            wxhs_dropout[k - 1][i][j] = 0.0;
                            if(dropout==false) {
                                wxhs_dropout[k - 1][i][j] = 1.0;
                            }
                        } else {
                            wxhs_dropout[k - 1][i][j] = 1.0;
                        }
                    }
                }
            }
            
            
            if ((iteration + 1) % 1000 == 0) {

                
                
                Utilities.writeWeights(wxhs, iteration);                
                System.out.println("Iteration: "+(iteration+1));
                System.out.println("- For training data");
                double hs[][][] = computeForward(total_layers, minibatch, hidden_units, xs_shuffle, wxhs, 
                        biases, false, wxhs_dropout, dropout_keep_rate);
                System.out.println("loss function (cross entropy): "+getCrossEntropy(minibatch, xs_shuffle, ys_shuffle, hidden_units, hs));
                System.out.println("classification accuracy: "+getClassificationError(minibatch, xs_shuffle, ys_shuffle, hidden_units, hs));
                
                //---
                System.out.println("- For dev");
                hs = computeForward(total_layers, totaldevSamples, hidden_units, xs_dev, wxhs, biases, false, 
                        wxhs_dropout, dropout_keep_rate);
                System.out.println("Classification accuracy: "+getClassificationError(totaldevSamples, xs_dev, ys_dev, hidden_units, hs));
                
                System.out.println("- For test");
                hs = computeForward(total_layers, totaltestSamples, hidden_units, xs_test, wxhs, biases, false, 
                        wxhs_dropout, dropout_keep_rate);
                System.out.println("Classification accuracy: "+getClassificationError(totaltestSamples, xs_test, ys_test, hidden_units, hs));
                
            }
            
                            
            double hs[][][] = 
                    computeForward(total_layers, minibatch, hidden_units, xs_shuffle, wxhs, biases,
                            true, wxhs_dropout, dropout_keep_rate);
            /*a 3-D array: this represents all layers: inputs, hidden layers and outputs
        D1: number of layers (ALL layers)
        D2: number of training samples
        D3: number of neurons of the layer
             */

            double wxhs_update[][][] = Utilities.initializeEmptyArray(wxhs);
            double biases_update[][] = Utilities.initializeEmptyArray(biases);
            double signalErrors[][][] = new double[total_layers][][];

            CalculateSignalErrorsCrossEntropy(hs, ys_shuffle, wxhs, 
                    hidden_units, minibatch, 
                    total_layers, signalErrors, wxhs_dropout, dropout_keep_rate);

            updateWeights(hs, wxhs, biases, hidden_units, minibatch, total_layers, signalErrors,
                    wxhs_update, biases_update, constant, wxhs_dropout, dropout_keep_rate, cache, gamma);
            wxhs = Utilities.DeepCopy(wxhs_update);
            biases = Utilities.DeepCopy(biases_update);

        }
    }

    public void readData(String filename, int totalSize, double[][] myInputs, double[][] myOutputs) throws FileNotFoundException, IOException {
        BufferedReader buf = new BufferedReader(new FileReader(filename));
        String s = "";
        
        int count = 0;
        while ((s = buf.readLine()) != null) {
            String arr[] = s.trim().split(" ");
            double inputs[] = new double[arr.length - 1];
            double outputs[] = new double[10];
            for (int i = 0; i < arr.length - 1; i++) {
                inputs[i] = Integer.parseInt(arr[i]);
            }
            myInputs[count] = inputs;
            outputs[Integer.parseInt(arr[arr.length - 1])] = 1;
            myOutputs[count] = outputs;
            count++;
            if (count >= totalSize) {
                break;
            }
        }
        buf.close();
    }
    public static void main(String argss[]) throws IOException {
        FeedForwardNeuralNetwork program = new FeedForwardNeuralNetwork();
        int trainingIDs = 50000;
        int testIDs = 10000;
        int devIDs = 10000;
        double[][] TrainingXs = new double[trainingIDs][];
        double[][] TrainingYs = new double[trainingIDs][];
        double[][] TestXs = new double[testIDs][];
        double[][] TestYs = new double[testIDs][];
        double[][] DevXs = new double[devIDs][];
        double[][] DevYs = new double[devIDs][];

        program.readData("MNISTTraining", trainingIDs, TrainingXs, TrainingYs);
        program.readData("MNISTTest", testIDs, TestXs, TestYs);
        program.readData("MNISTDev", devIDs, DevXs, DevYs);

        int hidden_units[] = new int[3];
        hidden_units[0] = TrainingXs[0].length;
        hidden_units[1] = 100;
        hidden_units[2] = TrainingYs[0].length;//fixed

        program.backpropagationNLayers(TrainingXs, TrainingYs, trainingIDs, DevXs, DevYs, devIDs, 
                TestXs, TestYs, testIDs, hidden_units, true);

    }
}
