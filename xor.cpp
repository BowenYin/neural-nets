/**
 * Bowen Yin
 * 9/10/2019
 * XOR connectivity model based on the spreadsheet.
 * This program reads from a file inputs.txt, which contains the input activations in order (one in each line).
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
using namespace std;

const double MAX_RAND = 2.0;
const int ITER_EPOCH = 1000;
const double MIN_ERR_CHANGE = 1E-10;
const double LEARN_FACTOR = 0.1;
const double LEARN_MULTIPLIER = 1.01;
const double MAX_LEARN = 1.5;
const int MAX_ITERATIONS = 100000;
const double ERR_THRESHOLD = 1E-2;

int inputNodes;
int outputNodes;
int hiddenLayers;
int trainingSets;
vector<int> hiddenLayerNodes;
vector<vector<double>> a;
vector<vector<vector<double>>> w;
vector<vector<vector<double>>> prevW;
vector<double> errors;
vector<double> prevErrors;
vector<vector<double>> inputs;
vector<double> targets;
double error;
double prevError;
double learningFactor = LEARN_FACTOR;

/**
 * Threshold function that is executed on every activation calculation.
 * Performs the sigmoid function on the given value.
 */
double outputFunc(double x)
{
   return 1.0 / (1.0 + exp(-x));
}

/**
 * 
 */
double dFunc(double x)
{
   return outputFunc(x) * (1 - outputFunc(x));
}

/**
 * 
 */
void randomizeWeights()
{
   random_device rd;
   mt19937 mt(rd());
   //uniform_real_distribution<double> rand(0, MAX_RAND);
   normal_distribution<double> rand(1.0, 0.5);
   for (int k = 0; k < inputNodes; k++)
      for (int j = 0; j < hiddenLayerNodes[0]; j++)
      {
         w[0][k][j] = rand(mt);
      }
   for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++)
      for (int i = 0; i < outputNodes; i++)
      {
         w[hiddenLayers][j][i] = rand(mt);
      }
}

/**
 * 
 */
double calculateInputDerivative(int k, int j, int trainingSet, double outputs[])
{
   double sum1 = 0.0;
   for (int K = 0; K < inputNodes; K++)
   {
      sum1 += a[0][K] * w[0][K][j];
   }
   double sum2 = 0.0;
   for (int J = 0; J < hiddenLayerNodes[hiddenLayers-1]; J++)
   {
      sum2 += a[hiddenLayers][J] * w[hiddenLayers][J][0];
   }
   double partial = -a[0][k] * dFunc(sum1) * (targets[trainingSet] - outputs[trainingSet]) * dFunc(sum2) * w[hiddenLayers][j][0];
   return partial;
} // double calculateInputDerivative

/**
 * 
 */
double calculateOutputDerivative(int j, int i, int trainingSet, double outputs[])
{
   double sum2 = 0.0;
   for (int J = 0; J < hiddenLayerNodes[hiddenLayers-1]; J++)
   {
      sum2 += a[hiddenLayers][J] * w[hiddenLayers][J][0];
   }
   double partial = -(targets[trainingSet] - outputs[trainingSet]) * dFunc(sum2) * a[hiddenLayers][j];
   //cout<<"\t\t"<<(targets[0] - output)<<endl;
   return partial;
} // double calculateOutputDerivative

/**
 * 
 */
double calculateError(double outputs[])
{
   double error = 0.0;
   for (int i = 0; i < trainingSets; i++)
      error += (targets[i] - outputs[i]) * (targets[i] - outputs[i]);
   return error / 2;
}

/**
 * 
 */
double calculateOutput(int trainingSet)
{
   for (int k = 0; k < inputNodes; k++)
   {
      a[0][k] = inputs[trainingSet][k];
   }
   for (int n = 0; n < hiddenLayers; n++)           // compute each of the hidden layers
   {
      for (int j = 0; j < hiddenLayerNodes[n]; j++) // loop through each node in the layer
      {
         //cout << "DEBUG: a[" << n+1 << "][" << j << "] =";
         double sum = 0;
         for (int k = 0; k < inputNodes; k++) // loop through previous layer
         {
            //cout << " a[" << n << "][" << k << "] * w[" << n << "][" << k << "][" << j << "] +";
            sum += a[n][k] * w[n][k][j];
         }
         //cout << endl;
         a[n+1][j] = outputFunc(sum); // perform activation function on the dot product
      }
   }
   for (int i = 0; i < outputNodes; i++)                         // compute the final layer of outputs
   {
      double sum = 0;
      for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++) // loop through last hidden layer
      {
         sum += a[hiddenLayers][j] * w[hiddenLayers][j][i];
      }
      a[hiddenLayers+1][i] = outputFunc(sum); // perform activation function on the dot product
   }
   return a[hiddenLayers+1][0];
} // void calculateOutput()

/**
 * 
 */
void adjustWeights(double outputs[], int trainingSet)
{
   for (int k = 0; k < inputNodes; k++)
   {
      for (int j = 0; j < hiddenLayerNodes[0]; j++)
      {
         w[0][k][j] -= learningFactor * calculateInputDerivative(k, j, trainingSet, outputs);
      }
   }
   for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++)
   {
      for (int i = 0; i < outputNodes; i++)
      {
         w[hiddenLayers][j][i] -= learningFactor * calculateOutputDerivative(j, i, trainingSet, outputs);
      }
   }
   //cout << "///////////"<<w[hiddenLayers][0][0]<<endl;
}

/**
 * 
 */
void printWeights()
{
   for (int n = 0; n < hiddenLayers; n++)
      for (int j = 0; j < hiddenLayerNodes[n]; j++)
         for (int k = 0; k < inputNodes; k++)
            cout << n << "," << k << "," << j << " " << w[n][k][j] << endl;
   for (int i = 0; i < outputNodes; i++)
      for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++)
         cout << hiddenLayers << "," << j << "," << i << " " << w[hiddenLayers][j][i] << endl;
}

/**
 * 
 */
void printOutputs(double outputs[])
{
   for (int i = 0; i < trainingSets; i++)
   {
      for (int j = 0; j < inputNodes; j++)
         cout << inputs[i][j] << ",";
      cout << " " << outputs[i] << endl;
   }
}

/**
 * Main method that creates the activation arrays and calculates the hidden and final values.
 * It first prompts the user for input and reads from the input file, then it prints out the final result after calculating.
 */
int main()
{
   ios_base::sync_with_stdio(false);
   cin.tie(0);
   
   // prompt the user for number of input nodes, output nodes, and hidden layers
   cout << "Input nodes: ";
   cin >> inputNodes;
   cout << "Output nodes: ";
   cin >> outputNodes;
   cout << "Hidden layers: ";
   cin >> hiddenLayers;
   
   hiddenLayerNodes.resize(hiddenLayers);
   for (int i = 0; i < hiddenLayers; i++)
   {
      cout << "Nodes in hidden layer "+to_string(i)+": "; // prompt the user for each of the hidden layers
      cin >> hiddenLayerNodes[i];
   }
   
   //double a[hiddenLayers+2][100];      // activations array
   //double w[hiddenLayers+1][100][100]; // weights array
   int size = max(inputNodes, outputNodes);
   for (int i = 0; i < hiddenLayers; i++)
   {
      if (hiddenLayerNodes[i] > size)
         size = hiddenLayerNodes[i];
   }
   a.resize(hiddenLayers+2, vector<double>(size)); // 2 for input and output layer
   w.resize(hiddenLayers+1, vector<vector<double>>(size, vector<double>(size)));
   
   string response;
   cout << "Randomize weights? (y/n) ";
   cin >> response;
   auto start = chrono::high_resolution_clock::now();
   if (response.substr(0, 1) == "n" || response.substr(0, 1) == "N")
   {
      ifstream weights("weights.txt"); // read weights from file
      for (int k = 0; k < inputNodes; k++)
         for (int j = 0; j < hiddenLayerNodes[0]; j++)
            weights >> w[0][k][j];
      for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++)
         for (int i = 0; i < outputNodes; i++)
            weights >> w[0][j][i];
      weights.close();
   }
   else
   {
      randomizeWeights();
   }
   cout << endl << "Initial weights:" << endl;
   printWeights();
   
   ifstream training("training.txt");
   training >> trainingSets;
   inputs.resize(trainingSets, vector<double>(inputNodes));
   targets.resize(trainingSets);
   for (int i = 0; i < trainingSets; i++)
   {
      for (int j = 0; j < inputNodes; j++) {
         training >> inputs[i][j];
      }
      training >> targets[i];
   }
   training.close();
   
   /*ifstream inputs("inputs.txt"); // read input activations from file
   for (int i = 0; i < inputNodes; i++)
   {
      inputs >> a[0][i];
   }*/
   
   double outputs[trainingSets];
   for (int i = 0; i < trainingSets; i++)
   {
      outputs[i] = calculateOutput(i);
   }
   //cout << outputs[0] << "\t" << outputs[1] << "\t" << outputs[2] << "\t" << outputs[3] << endl;
   error = calculateError(outputs);
   prevW = w;
   
   int index = 0;
   int iterations = 1;
   while (iterations <= MAX_ITERATIONS &&
          error >= ERR_THRESHOLD &&
          abs(error-prevError) >= MIN_ERR_CHANGE &&
          learningFactor != 0)
   {
      prevError = error;
      if (index >= trainingSets) index = 0;
      for (int k = 0; k < inputNodes; k++)
      {
         a[0][k] = inputs[index][k];
      }
      calculateOutput(index);
      adjustWeights(outputs, index);
      //printWeights();
      for (int i = 0; i < trainingSets; i++)
         outputs[i] = calculateOutput(i);
      //cout << outputs[0] << "\t" << outputs[1] << "\t" << outputs[2] << "\t" << outputs[3] << endl;
      
      error = calculateError(outputs);
      if (error < prevError)
      {
         //cout << "ERRDOWN" << endl;
         prevW = w;
         learningFactor *= LEARN_MULTIPLIER;
         if (learningFactor > MAX_LEARN) learningFactor = MAX_LEARN;
      }
      else
      {
         //cout << "ERRUP" << endl;
         w = prevW;
         //outputs[index] = calculateOutput(index);
         learningFactor /= LEARN_MULTIPLIER;
      }
      if (iterations % ITER_EPOCH == 0)
      {
         cout << "Iteration " << iterations << "\t";
         cout << "Error: " << error << "\n";
         //cout << "\tLFAC: " << learningFactor << "\n";
      }
      iterations++;
      index++;
   } // while iterations <= MAX_ITERATIONS
   cout << endl << "Iterations: " << iterations << endl;
   cout << "Final error: " << error << endl;
   cout << "Reason for termination: ";
   if (iterations > MAX_ITERATIONS) cout << "Exceeded max iterations.";
   else if (error < ERR_THRESHOLD) cout << "Below minimum error threshold.";
   else if (abs(error-prevError) < MIN_ERR_CHANGE) cout << "Error change below allowed minimum.";
   else if (learningFactor == 0) cout << "Learning factor reached zero.";
   cout << endl << endl << "Weights:" << endl;
   printWeights();
   cout << endl << "Outputs:" << endl;
   printOutputs(outputs);
   
   auto end = chrono::high_resolution_clock::now();
   cout << endl << "Execution time: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << " ms" << endl;
   return 0;
} // int main()