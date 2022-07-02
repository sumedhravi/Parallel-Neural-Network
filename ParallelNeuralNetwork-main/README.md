# ParallelNeuralNetwork
Guide contains steps to reproduce the results

## Installation

Unzip the the training data set from data folder and place it along with other files. Ensure naming of the files is as follows
```bash
trainImagePath = "train-images-idx3-ubyte";
testImagePath = "t10k-images-idx3-ubyte";
trainLabelsPath = "train-labels-idx1-ubyte";
testLabelsPath = "t10k-labels-idx1-ubyte";
```

## Usage
To get results for sequential and OpenMP, submit test.pbs job

```bash
qsub test.pbs
```

For cuda results, submit cuda_test.pbs
```bash
qsub cuda_test.pbs
```
