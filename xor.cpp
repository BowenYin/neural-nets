/**
 * Bowen Yin
 * 9/10/2019
 * 
 */
#include <iostream>
#include <fstream>
using namespace std;

/**
 * 
 */
int main() {
  int inputNodes;
  int outputNodes;
  int hiddenLayers;
  cout << "Input Nodes: ";
  cin >> inputNodes;
  cout << "Output Nodes: ";
  cin >> outputNodes;
  cout << "Hidden Layer Nodes: ";
  cin >> hiddenLayers;
  int hiddenLayerNodes[hiddenLayers];
  for (int i = 1; i <= hiddenLayers; i++) {
    cout << "Nodes in hidden layer "+i;
    cin >> hiddenLayerNodes[i-1];
  }
  double a[1000][hiddenLayers+2];
  double w[hiddenLayers+1][1000][1000];
  for (int n = 1; n <= hiddenLayers; n++) {
    for (int j = 0; j < hiddenLayerNodes[n-1]; j++) {
      int sum = 0;
      for (int k = 0; k < inputNodes; k++) {
        sum += a[n-1][k]*w[n-1][k][j];
      }
      a[n-1][j] = sigmoidFunc(sum);
    }
  }
  for (int i = 0; i < outputNodes; i++) {
    int sum = 0;
    for (int j = 0; j < hiddenLayerNodes[hiddenLayers-1]; j++) {
      sum += a[hiddenLayers][j]*w[hiddenLayers][j][i];
    }
    a[hiddenLayers+1][i] = sigmoidFunc(sum);
  }
}
double sigmoidFunc(int value) {
  return value;
}