/**
 * Bowen Yin
 * 9/10/2019
 * This program contains features to train and run a neural network with one hidden layer.
 * It reads data from a configuration file, training data file, and a weights file if needed.
 * Runtime options are set in the configuration file.
 * Other options, such as network size, are set by the user during runtime.
 * The network can be trained using provided values to determine weights.
 * The code implements back propagation.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
using namespace std;

double MIN_RAND;
double MAX_RAND;
int ITER_EPOCH;
double MIN_ERR_CHANGE;
double LEARN_FACTOR;
double LEARN_MULTIPLIER;
double MAX_LEARN;
int MAX_ITERATIONS;
double ERR_THRESHOLD;
bool SKIP_ROLLBACK = false;
string WEIGHTS_FILE_OUT;

int inputNodes;
int outputNodes;
int hiddenLayers;
int trainingSets;
vector<int> hiddenLayerNodes;
vector<int> layerSizes;
vector<vector<double>> a;
vector<vector<vector<double>>> w;
vector<vector<vector<double>>> prevW;
vector<vector<double>> inputs;
vector<vector<double>> outputs;
vector<vector<double>> targets;
vector<vector<double>> theta;
vector<vector<double>> psi;
vector<vector<double>> omega;
double error;
double prevError;
double learningFactor;

/**
 * Threshold function that is executed on every activation calculation.
 * Performs the sigmoid function on the given value.
 */
double outputFunc(double x)
{
   return 1.0 / (1.0 + exp(-x));
}

/**
 * Returns the derivative of the output function on the given value.
 */
double dFunc(double x)
{
   return outputFunc(x) * (1.0 - outputFunc(x));
}

/**
 * Randomizes all the weights using the configured dimensions.
 * Uses a uniform distribution between 0 and the configured maximum value to generate values.
 */
void randomizeWeights()
{
   random_device rd;
   mt19937 mt(rd());
   uniform_real_distribution<double> rand(MIN_RAND, MAX_RAND);
   //normal_distribution<double> rand(0.0, 1.0);
   
   for (int k = 0; k < inputNodes; k++)
      for (int j = 0; j < hiddenLayerNodes[0]; j++)
         w[0][k][j] = rand(mt);
   
   for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++)
      for (int i = 0; i < outputNodes; i++)
         w[hiddenLayers][j][i] = rand(mt);
   return;
} // void randomizeWeights

/**
 * Calculates the partial derivative for a weight between an input and a hidden layer.
 */
double calculateInputDerivative(int k, int j, int trainingSet)
{
   // sum of activations times weights for input layer
   double sum1 = 0.0;
   for (int K = 0; K < inputNodes; K++)
      sum1 += a[0][K] * w[0][K][j];
   
   // second summation for calculating the partial
   double sum2 = 0.0;
   for (int I = 0; I < outputNodes; I++)
      sum2 += (targets[trainingSet][I] - outputs[trainingSet][I]) * dFunc(outputs[trainingSet][I]) * w[hiddenLayers][j][I];
   
   double partial = -a[0][k] * dFunc(sum1) * sum2;
   return partial;
} // double calculateInputDerivative

/**
 * Calculates the partial derivative for a weight between a hidden layer and an output.
 */
double calculateOutputDerivative(int j, int i, int trainingSet)
{
   double partial = -(targets[trainingSet][i] - outputs[trainingSet][i]) * dFunc(outputs[trainingSet][i]) * a[hiddenLayers][j];
   return partial;
}

/**
 * Calculates the total error of the network, using target values from the training data.
 */
double calculateError()
{
   double error = 0.0;
   for (int i = 0; i < outputNodes; i++)
      error += omega[hiddenLayers+1][i] * omega[hiddenLayers+1][i];
   return error / 2.0;
}

/**
 * Calculates the output values in a preconfigured network.
 * Loops through each of the layers and fill in the activations.
 */
void calculateOutput(int trainingSet)
{
   for (int k = 0; k < inputNodes; k++) // fill in the input layer with training values
      a[0][k] = inputs[trainingSet][k];
   
   for (int n = 0; n < hiddenLayers; n++)           // compute each of the hidden layers
      for (int j = 0; j < hiddenLayerNodes[n]; j++) // loop through each node in the layer
      {
         //cout << "DEBUG: a[" << n+1 << "][" << j << "] =";
         double sum = 0.0;
         for (int k = 0; k < inputNodes; k++) // loop through previous layer
         {
            //cout << " a[" << n << "][" << k << "] * w[" << n << "][" << k << "][" << j << "] +";
            sum += a[n][k] * w[n][k][j];
         }
         //cout << endl;
         a[n+1][j] = outputFunc(sum);    // perform activation function on the dot product
      }
   
   for (int i = 0; i < outputNodes; i++) // compute the final layer of outputs
   {
      double sum = 0.0;
      for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++) // loop through last hidden layer
         sum += a[hiddenLayers][j] * w[hiddenLayers][j][i];
      a[hiddenLayers+1][i] = outputFunc(sum);
   }
} // void calculateOutput

/**
 * 
 */
void runNetwork(int trainingSet) {
   for (auto &v: theta)
      fill(v.begin(), v.end(), 0.0);
   for (auto &v: omega)
      fill(v.begin(), v.end(), 0.0);
   
   /*for (int i = 0; i < outputNodes; i++) {
      for (auto &v: theta)
         fill(v.begin(), v.end(), 0.0);
      for (int j = 0; j < hiddenLayerNodes[0]; j++) {
         for (int k = 0; k < inputNodes; k++) {
            theta[hiddenLayers][j] += a[0][k]*w[0][k][j];
         }
         a[hiddenLayers][j] = outputFunc(theta[hiddenLayers][j]);
         theta[hiddenLayers+1][i] += a[hiddenLayers][j]*w[hiddenLayers][j][i];
      }
      a[hiddenLayers+1][i] = outputFunc(theta[hiddenLayers+1][i]);
      omega[hiddenLayers+1][i] = targets[trainingSet][i]-a[hiddenLayers+1][i];
   }*/
   /*for (int i = 0; i < outputNodes; i++) {
      for (int j = 0; j < hiddenLayerNodes[0]; j++) {
         theta[hiddenLayers+1][i] += a[hiddenLayers][j]*w[hiddenLayers][j][i];
      }
      a[hiddenLayers+1][i] = outputFunc(theta[hiddenLayers+1][i]);
      omega[hiddenLayers+1][i] = targets[trainingSet][i]-a[hiddenLayers+1][i];
      psi[hiddenLayers+1][i] = omega[hiddenLayers+1][i]*dFunc(theta[hiddenLayers+1][i]);
   }*/
   for (int alpha = 1; alpha <= hiddenLayers+1; alpha++) {
      for (int beta = 0; beta < layerSizes[alpha]; beta++) {
         for (int gamma = 0; gamma < layerSizes[alpha-1]; gamma++) {
            theta[alpha][beta] += a[alpha-1][gamma]*w[alpha-1][gamma][beta];
         }
         a[alpha][beta] = outputFunc(theta[alpha][beta]);
      }
   }
}

/**
 * 
 */
void trainNetwork() {
   for (int t = 0; t < trainingSets; t++) {
      for (auto &v: psi)
         fill(v.begin(), v.end(), 0.0);
      for (int k = 0; k < inputNodes; k++) // fill in the input layer with training values
         a[0][k] = inputs[t][k];
      runNetwork(t);
      for (int i = 0; i < outputNodes; i++)
         outputs[t][i] = a[hiddenLayers+1][i];
      
      /*for (int j = 0; j < hiddenLayerNodes[0]; j++) {
         for (int i = 0; i < outputNodes; i++) {
            psi[hiddenLayers+1][i] = omega[hiddenLayers+1][i]*dFunc(theta[hiddenLayers+1][i]);
            w[hiddenLayers][j][i] += learningFactor*a[hiddenLayers][j]*psi[hiddenLayers+1][i];
            omega[hiddenLayers][j] += psi[hiddenLayers+1][i]*w[hiddenLayers][j][i];
         }
         psi[hiddenLayers][j] = omega[hiddenLayers][j]*dFunc(theta[hiddenLayers][j]);
      }*/
      /*for (int i = 0; i < outputNodes; i++) {
         psi[hiddenLayers+1][i] = omega[hiddenLayers+1][i]*dFunc(theta[hiddenLayers+1][i]);
         for (int j = 0; j < hiddenLayerNodes[0]; j++) {
            w[hiddenLayers][j][i] += learningFactor*a[hiddenLayers][j]*psi[hiddenLayers+1][i];
            omega[hiddenLayers][j] += psi[hiddenLayers+1][i]*w[hiddenLayers][j][i];
         }
      }
      for (int j = 0; j < hiddenLayerNodes[0]; j++) {
         psi[hiddenLayers][j] = omega[hiddenLayers][j]*dFunc(theta[hiddenLayers][j]);
         for (int k = 0; k < inputNodes; k++) {
            w[0][k][j] += learningFactor*a[0][k]*psi[hiddenLayers][j];
         }
      }*/
      
      for (int i = 0; i < outputNodes; i++) {
         omega[hiddenLayers+1][i] = targets[t][i]-a[hiddenLayers+1][i];
         psi[hiddenLayers+1][i] = omega[hiddenLayers+1][i]*dFunc(theta[hiddenLayers+1][i]);
         for (int j = 0; j < layerSizes[hiddenLayers]; j++) {
            w[hiddenLayers][j][i] += learningFactor*a[hiddenLayers][j]*psi[hiddenLayers+1][i];
         }
      }
      /*for (int alpha = hiddenLayers; alpha > 0; alpha--) {
         for (int beta = 0; beta < layerSizes[alpha+1]; beta++) {
            psi[alpha+1][beta] = omega[alpha+1][beta]*dFunc(theta[alpha+1][beta]);
            for (int gamma = 0; gamma < layerSizes[alpha]; gamma++) {
               w[alpha][gamma][beta] += learningFactor*a[alpha][gamma]*psi[alpha+1][beta];
               omega[alpha][gamma] += psi[alpha+1][beta]*w[alpha][gamma][beta];
            }
         }
      }*/
      for (int alpha = hiddenLayers; alpha > 0; alpha--) {
         for (int beta = 0; beta < layerSizes[alpha]; beta++) {
            for (int right = 0; right < layerSizes[alpha+1]; right++) {
               omega[alpha][beta] += psi[alpha+1][right]*w[alpha][beta][right];
            }
            psi[alpha][beta] = omega[alpha][beta]*dFunc(theta[alpha][beta]);
            for (int left = 0; left < layerSizes[alpha-1]; left++) {
               w[alpha-1][left][beta] += learningFactor*a[alpha-1][left]*psi[alpha][beta];
            }
         }
      }
   }
}

/**
 * Adjusts all the weights in the network by calculating partial derivates and adjusting,
 * based on the learning factor.
 */
void adjustWeights(int trainingSet)
{
   for (int k = 0; k < inputNodes; k++)
      for (int j = 0; j < hiddenLayerNodes[0]; j++)
         w[0][k][j] -= learningFactor * calculateInputDerivative(k, j, trainingSet);
   
   for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++)
      for (int i = 0; i < outputNodes; i++)
         w[hiddenLayers][j][i] -= learningFactor * calculateOutputDerivative(j, i, trainingSet);
   return;
}

/**
 * Prints all the weights, in addition to the indexes they correspond with.
 */
void printWeights(string weightsFile)
{
   ofstream out;
   out.open(weightsFile);
   for (int n = 0; n < hiddenLayers; n++)
      for (int j = 0; j < hiddenLayerNodes[n]; j++)
         for (int k = 0; k < inputNodes; k++)
            out << n << "," << k << "," << j << " " << w[n][k][j] << endl;
   
   for (int i = 0; i < outputNodes; i++)
      for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++)
         out << hiddenLayers << "," << j << "," << i << " " << w[hiddenLayers][j][i] << endl;
   return;
}

/**
 * Prints the outputs for each training set.
 */
void printOutputs()
{
   for (int t = 0; t < trainingSets; t++)
   {
      for (int i = 0; i < inputNodes; i++)
         cout << inputs[t][i] << ",";
      for (int i = 0; i < outputNodes; i++)
         cout << "\t" << outputs[t][i] << " (" << targets[t][i] << ")"; //* 16777216.0;
      cout << endl;
   }
   return;
}

/**
 * Main method that creates the activation arrays and calculates the hidden and final values.
 * It first prompts the user for input and reads from the input file, then it prints out the final result after calculating.
 * Performs a steepest descent using training data.
 */
int main()
{
   // included for faster I/O
   //ios_base::sync_with_stdio(false);
   //cin.tie(0);
   
   cout << "Press enter to use defaults (indicated in parentheses)." << endl;
   
   cout << "Configuration file: (./options.txt) ";
   string optionsFile;
   getline(cin, optionsFile);
   if (optionsFile.empty())
      optionsFile = "./options.txt";
   
   cout << "Training data file: (./training.txt) ";
   string trainingFile;
   getline(cin, trainingFile);
   if (trainingFile.empty())
      trainingFile = "./training.txt";
   
   // load in runtime options from configuration file
   ifstream options(optionsFile);
   options >> MIN_RAND;
   options >> MAX_RAND;
   options >> ITER_EPOCH;
   options >> MIN_ERR_CHANGE;
   options >> LEARN_FACTOR;
   options >> LEARN_MULTIPLIER;
   options >> MAX_LEARN;
   options >> MAX_ITERATIONS;
   options >> ERR_THRESHOLD;
   options >> WEIGHTS_FILE_OUT;
   int weightsRollback;
   options >> weightsRollback;
   if (weightsRollback == 0)
      SKIP_ROLLBACK = true;
   options.close();
   learningFactor = LEARN_FACTOR;
   
   // prompt the user for number of input nodes, output nodes, and hidden layers
   cout << "Input nodes: ";
   cin >> inputNodes;
   cout << "Output nodes: ";
   cin >> outputNodes;
   cout << "Hidden layers: ";
   cin >> hiddenLayers;
   
   hiddenLayerNodes.resize(hiddenLayers);
   layerSizes.resize(hiddenLayers+2);
   layerSizes[0] = inputNodes;
   layerSizes[hiddenLayers+1] = outputNodes;
   for (int i = 0; i < hiddenLayers; i++)
   {
      cout << "Nodes in hidden layer "+to_string(i)+": "; // prompt the user for each of the hidden layers
      cin >> hiddenLayerNodes[i];
      layerSizes[i+1] = hiddenLayerNodes[i];
   }
   
   // determine the array size needed
   int size = max(inputNodes, outputNodes);
   for (int i = 0; i < hiddenLayers; i++)
      if (hiddenLayerNodes[i] > size)
         size = hiddenLayerNodes[i];
   
   a.resize(hiddenLayers+2, vector<double>(size)); // 2 for input and output layer
   w.resize(hiddenLayers+1, vector<vector<double>>(size, vector<double>(size)));
   
   theta.resize(hiddenLayers+2, vector<double>(size));
   omega.resize(hiddenLayers+2, vector<double>(size));
   psi.resize(hiddenLayers+2, vector<double>(size));
   
   cout << "Randomize weights? (Y/n) ";
   cin.ignore();
   string response;
   getline(cin, response);
   
   if (response.substr(0, 1) == "n" || response.substr(0, 1) == "N")
   {
      // ask for weights file
      cout << "Weights file: (./weights.txt) ";
      string weightsFile;
      getline(cin, weightsFile);
      if (weightsFile.empty())
         weightsFile = "./weights.txt";
      ifstream weights(weightsFile); // read weights from file
      
      for (int k = 0; k < inputNodes; k++)
         for (int j = 0; j < hiddenLayerNodes[0]; j++)
            weights >> w[0][k][j];
      for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++)
         for (int i = 0; i < outputNodes; i++)
            weights >> w[hiddenLayers][j][i];
      weights.close();
   }
   else
      randomizeWeights();
   //printWeights(WEIGHTS_FILE_INIT);
   
   auto start = chrono::high_resolution_clock::now(); // start tracking execution time
   
   // read training data file
   ifstream training(trainingFile);
   training >> trainingSets;
   inputs.resize(trainingSets, vector<double>(inputNodes));
   targets.resize(trainingSets, vector<double>(outputNodes));
   
   // load training data into arrays
   for (int t = 0; t < trainingSets; t++)
   {
      for (int i = 0; i < inputNodes; i++)
      {
         training >> inputs[t][i];
         //inputs[t][i] /= 16777216.0;
      }
      for (int i = 0; i < outputNodes; i++)
      {
         training >> targets[t][i];
         //targets[t][i] /= 16777216.0;
      }
   }
   training.close();
   
   outputs.resize(trainingSets, vector<double>(outputNodes));
   /*for (int t = 0; t < trainingSets; t++) // run network for each training set
   {
      for (int k = 0; k < inputNodes; k++) // fill in the input layer with training values
         a[0][k] = inputs[t][k];
      //calculateOutput(t);
      runNetwork(t);
      for (int i = 0; i < outputNodes; i++)
         outputs[t][i] = a[hiddenLayers+1][i];
   }*/
   error = __DBL_MAX__;
   //prevW = w;
   
   int index = 0;
   int iterations = 1;
   while (iterations <= MAX_ITERATIONS && error >= ERR_THRESHOLD && abs(error-prevError) >= MIN_ERR_CHANGE && learningFactor != 0)
   {
      prevError = error;
      
      if (index >= trainingSets)           // set to 0 if index exceeds max sets
         index = 0;
      /*for (int k = 0; k < inputNodes; k++) // fill in input layer
         a[0][k] = inputs[index][k];*/
      
      //calculateOutput(index);
      //adjustWeights(index);
      trainNetwork();
      
      /*for (int t = 0; t < trainingSets; t++) // compute network for each training set
      {
         //calculateOutput(t);
         for (int i = 0; i < outputNodes; i++)
            outputs[t][i] = a[hiddenLayers+1][i];
      }*/
      
      error = calculateError();
      if (error < prevError || SKIP_ROLLBACK)
      {
         prevW = w;
         learningFactor *= LEARN_MULTIPLIER;
         if (learningFactor > MAX_LEARN)
            learningFactor = MAX_LEARN;
      }
      else // roll back weights
      {
         w = prevW;
         learningFactor /= LEARN_MULTIPLIER;
      }
      
      if (iterations % ITER_EPOCH == 0) // prints out progress every so often
      {
         cout << "Iteration " << iterations << "\t";
         cout << "Error: " << error << "\t";
         cout << "Learn: " << learningFactor << "\n";
      }
      iterations++;
      index++;
   } // while termination conditions have not been reached
   
   auto end = chrono::high_resolution_clock::now(); // stop tracking run time
   
   cout << endl << "Iterations: " << iterations-1 << endl;
   cout << "Final error: " << error << endl;
   cout << "Reason for termination: ";
   // termination condition determined
   if (iterations > MAX_ITERATIONS)
      cout << "Exceeded max iterations.";
   else if (error < ERR_THRESHOLD)
      cout << "Below minimum error threshold.";
   else if (abs(error-prevError) < MIN_ERR_CHANGE)
      cout << "Error change below allowed minimum.";
   else if (learningFactor == 0)
      cout << "Learning factor reached zero.";
   cout << endl << "Lambda: " << learningFactor << endl;
   
   ofstream outFile;
   outFile.open("out.txt");
   for (int i = 0; i < outputNodes; i++)
      outFile << outputs[0][i] /* * 16777216.0 */ << "\n";
   printWeights(WEIGHTS_FILE_OUT);
   cout << endl << "Outputs:" << endl;
   printOutputs();
   
   cout << endl << "Execution time: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << " ms!!!" << endl;
   return 0;
} // int main()