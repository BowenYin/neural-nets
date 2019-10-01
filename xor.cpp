/**
 * Bowen Yin
 * 9/10/2019
 * XOR connectivity model based on the spreadsheet.
 * This program reads from a file inputs.txt, which contains the input activations in order (one in each line).
 */
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

int inputNodes;
int outputNodes;
int hiddenLayers;

/**
 * Threshold function that is executed on every activation calculation.
 * Performs the sigmoid function on the given value.
 */
double thresholdFunc(double value) {
   return 1.0 / (1.0 + exp(-1.0 * value));
}

/**
 * Prompts the user for the number of input nodes, output nodes, and hidden layers.
 */
void promptInputs() {
   cout << "Input nodes: ";
   cin >> inputNodes;
   cout << "Output nodes: ";
   cin >> outputNodes;
   cout << "Hidden layers: ";
   cin >> hiddenLayers;
}

/**
 * 
 */
double calculateError(double output) {
   return 0.5 * (1.0 - output) * (1.0 - output);
}

/**
 * Main method that creates the activation arrays and calculates the hidden and final values.
 * It first prompts the user for input and reads from the input file, then it prints out the final result after calculating.
 */
int main() {
   promptInputs();
   
   int hiddenLayerNodes[hiddenLayers];
   for (int i = 0; i < hiddenLayers; i++) {
      cout << "Nodes in hidden layer "+to_string(i)+": "; // prompt the user for each of the hidden layers
      cin >> hiddenLayerNodes[i];
   }
   
   double a[hiddenLayers+2][100];      // activations array
   double w[hiddenLayers+1][100][100]; // weights array
   
   w[0][0][0] = 0.55;
   w[0][0][1] = 0.2;
   w[0][1][0] = 0.825;
   w[0][1][1] = 0.08;
   w[1][0][0] = 0.3;
   w[1][1][0] = 0.9;
   
   ifstream file("inputs.txt"); // read input activations from file
   for (int i = 0; i < inputNodes; i++) {
      file >> a[0][i];
   }
   
   for (int n = 0; n < hiddenLayers; n++) {           // compute each of the hidden layers
      for (int j = 0; j < hiddenLayerNodes[n]; j++) { // loop through each node in the layer
         cout << "DEBUG: a[" << n+1 << "][" << j << "] =";
         double sum = 0;
         for (int k = 0; k < inputNodes; k++) { // loop through previous layer
            cout << " a[" << n << "][" << k << "] * w[" << n << "][" << k << "][" << j << "] +";
            sum += a[n][k] * w[n][k][j];
         }
         cout << endl;
         a[n+1][j] = thresholdFunc(sum); // perform activation function on the dot product
      }
   }
   
   for (int i = 0; i < outputNodes; i++) {                         // compute the final layer of outputs
      double sum = 0;
      for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++) { // loop through last hidden layer
         sum += a[hiddenLayers][j] * w[hiddenLayers][j][i];
      }
      a[hiddenLayers+1][i] = thresholdFunc(sum); // perform activation function on the dot product
   }
   
   //cout << "Final value: "+to_string(a[2][0]) << endl;
   cout << "Error: " << calculateError(a[2][0]) << endl;
   return 0;
} // int main()