/* 
Authors: Ajay Mandadi, Kartik Angadi, Sumedh Ravi
Class: ECE 6122
Last Date Modified: 12/03/2021

Description:
Final Project

File containing code to train and test a Multi Layer Percepton model on 
MNIST image data on GPU.
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <chrono>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>

class Node
{
public:
    std::vector<double> weights;
    double bias;
    double activation;
    double output;
    double grad;
};

class Layer
{
public:
    std::vector<Node> nodes;
};

/* Activation function */
double sigmoid(double input, bool derivative = false);

/* Neural Network class */
class NeuralNetwork
{
public:
    std::default_random_engine rDist;
    std::vector<layer> layers;

    NeuralNetwork(unsigned long seed, const std::vector<int>& layerSizes);

    void initializeWeights();
    std::vector<double> evaluate(const std::vector<double>& expected);
    std::vector<double> cudaEvaluate(const std::vector<double>& inputActivations);
    void backPropagateError(const std::vector<double>& input, const std::vector<double>& output);
    void update(const std::vector<double>& input, double learningRate);
};

NeuralNetwork merge(const std::vector<NeuralNetwork>& models);

NeuralNetwork::NeuralNetwork(unsigned long seed, const std::vector<int>& layerSizes)
{
    rDist.seed(seed);

    /* First layer contains the number of inputs. So, we consider the number of layers as (size-1). */
    layers.resize(layerSizes.size() - 1);
    for (int layer = 0; layer < layers.size(); layer++)
    {
        /* Set the number of nodes in each layer  */
        layers[layer].nodes.resize(layerSizes[layer + 1]);
    }

    for (int layer = 0; layer < layers.size(); layer++)
    {
        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            /* Set the number of weights in the nth layer to the number of outputs from the (n-1)th layer */
            layers[layer].nodes[node].weights.resize(layerSizes[layer], 1.0);
        }
    }
}

/* Set random weights */
void NeuralNetwork::initializeWeights()
{
    std::uniform_real_distribution<double> uniform(0, 1);

    for (int layer = 0; layer < layers.size(); layer++)
    {
        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            /* Set bias component of each node to 0 */
            layers[layer].nodes[node].bias = 0.0;
            /* Set random weight */
            for (int weight = 0; weight < layers[layer].nodes[node].weights.size(); weight++)
            {
                layers[layer].nodes[node].weights[weight] = 2 * (uniform(rDist) - 0.5);
            }
        }
    }
}

/* Forward pass */
std::vector<double> NeuralNetwork::evaluate(const std::vector<double>& input)
{
    std::vector<double> activations = input;

    for (int layer = 0; layer < layers.size(); layer++)
    {
        std::vector<double> nextInput(layers[layer].nodes.size());

        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            double activation = layers[layer].nodes[node].bias;
            for (int i = 0; i < activations.size(); i++)
            {
                activation += layers[layer].nodes[node].weights[i] * activations[i];
            }
            layers[layer].nodes[node].activation = activation;
            layers[layer].nodes[node].output = sigmoid(layers[layer].nodes[node].activation, false);

            nextInput[node] = layers[layer].nodes[node].output;
        }
        activations = nextInput;
    }

    return activations;
}

__global__ void vectorProduct(double *input_activations, double *weights, double *output, int N)
{
    int curr = (blockIdx.x * blockDim.x) + threadIdx.x;

    {
        if (curr<N)
        {
            output[curr] = input_activations[curr] * weights [curr];
        }
    }
}

/* Matrix element multiplications for a layer */
__global__ void cuda_forward_layer(const double *input_activations, const double *weights, double *output, int N)
{
    int node = threadIdx.x;

    for (int i =0; i<N; i++)
    {
        output[node] += input_activations[i]*weights[node*N+i];
    }
}

/* Forward pass for CUDA */
std::vector<double> NeuralNetwork::cudaEvaluate(const std::vector<double>& input)
{
    std::vector<double> activations = input;
    double *in = &activations[0];
    int numThreads;

    for (int layer = 0; layer < layers.size(); layer++)
    {
        in = &activations[0];
        numThreads = layers[layer].nodes.size();

        long inputSize = activations.size() * sizeof(double);
        long outputSize = layers[layer].nodes.size()*sizeof(double);

        double *D_input_activations = NULL;
        int err = cudaMalloc((void**)&D_input_activations, inputSize);
        err = cudaMemcpy(D_input_activations, (double*)in, inputSize, cudaMemcpyHostToDevice);

        double *D_weights = NULL;
        err = cudaMalloc((void**)&D_weights, inputSize*layers[layer].nodes.size());

        double *D_activations = NULL;
        err = cudaMalloc((void**)&D_activations, outputSize);
        cudaMemset(D_activations, 0, outputSize);

        std::vector<double> nextInput(layers[layer].nodes.size());

        double *dst = D_weights;
        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            double *src = &layers[layer].nodes[node].weights[0];
            size_t sz = layers[layer].nodes[node].weights.size();
            cudaMemcpy(dst, src, sizeof(double)*sz, cudaMemcpyHostToDevice);
            dst += sz ;
        }

        cuda_forward_layer<<<1, numThreads>>>(D_input_activations, D_weights, D_activations, activations.size());
        double* output = (double*)malloc(outputSize);
        err = cudaMemcpy(output, D_activations, outputSize, cudaMemcpyDeviceToHost);

        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            layers[layer].nodes[node].activation = output[node] + layers[layer].nodes[node].bias;
            layers[layer].nodes[node].output = sigmoid(layers[layer].nodes[node].activation, false);
            nextInput[node] = layers[layer].nodes[node].output;
        }

        activations = nextInput;
        cudaFree(D_input_activations);
        cudaFree(D_weights);
        cudaFree(D_activations);
        free(output);
    }

    return input;
}

double sigmoid(double input, bool derivative)
{
    if (derivative)
        return input * (1.0 - input);
    else
        return 1.0 / (1.0 + std::exp(-input));
}

/* Backward propagation step. Calculate and store gradients */
void NeuralNetwork::backPropagateError(const std::vector<double>& input, const std::vector<double>& expected)
{
    std::vector<double> output = cudaEvaluate(input);

    for (int layer = layers.size() - 1; layer >= 0; layer--)
    {
        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            double error;
            if (layer == layers.size() - 1)
            {
                error = expected[node] - layers[layer].nodes[node].output;
            }
            else
            {
                error = 0.0;
                for (int weight = 0; weight < layers[layer + 1].nodes.size(); weight++)
                {
                    error += layers[layer + 1].nodes[weight].weights[node] * layers[layer + 1].nodes[weight].grad;
                }
            }
            layers[layer].nodes[node].grad = error * sigmoid(layers[layer].nodes[node].output, true);
        }
    }
}

/* Update weights according to calculated gradients */
void NeuralNetwork::update(const std::vector<double>& input, double learningRate)
{
    for (int layer = 0; layer < layers.size(); layer++)
    {
        std::vector<double> inputs;
        if (layer == 0)
        {
            inputs = input;
        }
        else
        {
            for (int i = 0; i < layers[layer - 1].nodes.size(); i++)
            {
                inputs.push_back(layers[layer - 1].nodes[i].output);
            }
        }

        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            for (int i = 0; i < inputs.size(); i++)
            {
                layers[layer].nodes[node].weights[i] += learningRate * layers[layer].nodes[node].grad * inputs[i];
            }
            layers[layer].nodes[node].bias += learningRate * layers[layer].nodes[node].grad;
        }
    }
}

/* Merge networks run on different threads by averaging the weights */
NeuralNetwork merge(const std::vector<NeuralNetwork>& models)
{
    NeuralNetwork nn = models[0];
    for (int layer = 0; layer < nn.layers.size(); layer++)
    {
        
        for (int node = 0; node < nn.layers[layer].nodes.size(); node++)
        {
            for (int model = 1; model < models.size(); model++)
            {
                nn.layers[layer].nodes[node].bias += models[model].layers[layer].nodes[node].bias;
            }
            nn.layers[layer].nodes[node].bias /= models.size();
            for (int weight = 0; weight < nn.layers[layer].nodes[node].weights.size(); weight++)
            {
                for (int m = 1; m < models.size(); m++)
                {
                    nn.layers[layer].nodes[node].weights[weight] += models[m].layers[layer].nodes[node].weights[weight];
                }
                nn.layers[layer].nodes[node].weights[weight] /= models.size();
            }
        }
    }
    return nn;
}

int convertToInt(char *number)
{
    return ((number[0] & 0xff) << 24) | ((number[1] & 0xff) << 16) | ((number[2] & 0xff) << 8) | ((number[3] & 0xff) << 0);
};

std::vector<std::vector<double>> readMnistImages(const std::string& filepath)
{
    std::vector<std::vector<char>> imageData;

    std::ifstream fileStream(filepath, std::ios::binary);
    if (!fileStream.good())
    {
        std::cout << "Error loading images from '" << filepath << "'\n";
        return std::vector<std::vector<double>>();
    }

    char magic[4];
    fileStream.read((char*)magic, 4);
    char nImages[4];
    fileStream.read((char*)nImages, 4);
    char nRows[4];
    fileStream.read((char*)nRows, 4);
    char nCols[4];
    fileStream.read((char*)nCols, 4);

    int magicNumber = convertToInt(magic);
    int numImages = convertToInt(nImages);
    int numRows = convertToInt(nRows);
    int numCols = convertToInt(nCols);
    int numPixels = numImages * numRows * numCols;
    std::vector<char> imagePixels(numPixels);
    fileStream.read(imagePixels.data(), numImages * numRows * numCols);

    for (int i = 0; i < numImages; i++)
    {
        imageData.push_back(std::vector<char>(imagePixels.begin() + (i * numRows * numCols), imagePixels.begin() + ((i + 1) * numRows * numCols)));
    }

    std::vector<std::vector<double>> fileData(imageData.size(), std::vector<double>(imageData[0].size()));
    for (int i = 0; i < imageData.size(); i++)
    {
        for (int j = 0; j < imageData[0].size(); j++)
        {
            /* Scaling pixel values between 0 and 1 */
            fileData[i][j] = ((double)((unsigned char)imageData[i][j])) / 255.0;
        }
    }

    return fileData;
}

std::vector<int> readMnistLabels(const std::string& filepath)
{
    std::vector<char> labelData;

    std::ifstream fileStream(filepath, std::ios::binary);
    if (!fileStream.good())
    {
        std::cout << "Error loading labels from '" << filepath << "'\n";
        return std::vector<int>();
    }

    char magic[4];
    fileStream.read((char*)magic, 4);
    char nImages[4];
    fileStream.read((char*)nImages, 4);
    int magicNumber = convertToInt(magic);
    int numImages = convertToInt(nImages);

    labelData.resize(numImages);
    fileStream.read(labelData.data(), numImages);
    std::vector<int> fileData(labelData.size());
    for (int i = 0; i < labelData.size(); i++)
    {
        fileData[i] = ((int)(unsigned char)labelData[i]);
    }

    return fileData;
}

/* Prepare Data in format appropriate for training and testing the neural network */
void prepareData(const std::vector<std::vector<double>>& imageData, const std::vector<int>& imageLabels, std::vector<std::vector<std::vector<double>>>& formattedData)
{
    for (int i = 0; i < imageData.size(); i++)
    {
        std::vector<double> output(10, 0.0);
        output[imageLabels[i]] = 1.0;
        formattedData.push_back( {imageData[i], output} );
    }
}

class CONFIG
{
public:
    int numOfEpochs = 10;
    bool executeParallel = false;
    double learningRate = 0.5;
    std::string trainImagePath = "train-images-idx3-ubyte";
    std::string testImagePath = "t10k-images-idx3-ubyte";
    std::string trainLabelsPath = "train-labels-idx1-ubyte";
    std::string testLabelsPath = "t10k-labels-idx1-ubyte";
};

int main(int argc, char** argv)
{

    CONFIG config = CONFIG();

    if(argc < 2){
        std::cout<<"input params required. usage ./executable [num_epochs]"<<std::endl;
        return -1;
    }

    int num_epochs = std::stoi(argv[1]);

    /* Load image and label data */
    std::vector<std::vector<double>> trainingImages = readMnistImages(config.trainImagePath);
    std::vector<int> trainingLabels = readMnistLabels(config.trainLabelsPath);
    std::vector<std::vector<double>> testingImages = readMnistImages(config.testImagePath);
    std::vector<int> testingLabels = readMnistLabels(config.testLabelsPath);

    if (trainingImages.empty() || trainingLabels.empty() || testingImages.empty() || testingLabels.empty())
    {
        std::cout << "Error loading data" << std::endl;
        return 1;
    }

    std::vector<std::vector<std::vector<double>>> trainData;
    prepareData(trainingImages, trainingLabels, trainData);

    std::vector<std::vector<std::vector<double>>> testData;
    prepareData(testingImages, testingLabels, testData);

    NeuralNetwork model = NeuralNetwork(2021, {784, 100, 10});
    model.initializeWeights();

    int numThreads = 1;
    int batchLength = 100;
    std::default_random_engine rEng(11162021);
    auto start = std::chrono::high_resolution_clock::now();

    if (config.executeParallel)
    {
        std::vector<NeuralNetwork> models;
        for (int n = 0; n < numThreads; n++)
        {
            models.push_back(model);
        }
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
            //std::cout << "Epoch " << (epoch+1) << " / " << config.numOfEpochs << std::endl;
            std::shuffle(trainData.begin(), trainData.end(), rEng);

            for (int i = 0; i < trainData.size(); i += numThreads * batchLength)
            {
                for (int n = 0; n < numThreads; n++)
                {
                    models[n] = model;
                }
                for (int n = 0; n < numThreads; n++)
                {
                    for (int m = 0; m < batchLength; m++)
                    {
                        int idx = i + n * batchLength + m;
                        if (idx < trainData.size())
                        {
                            models[n].backPropagateError(trainData[idx][0], trainData[idx][1]);
                            models[n].update(trainData[idx][0], config.learningRate);
                        }
                    }
                }
                model = merge(models);
            }
        }
    }
    else
    {
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
            //std::cout << "Epoch " << (epoch+1) << " / " << config.numOfEpochs << std::endl;
            std::shuffle(trainData.begin(), trainData.end(), rEng);
            for (int i = 0; i < trainData.size(); i += 1)
            {
                if (i < trainData.size())
                {
                    model.backPropagateError(trainData[i][0], trainData[i][1]);
                    model.update(trainData[i][0], config.learningRate);
                }
                else break;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    std::cout << "{" << std::endl;
    std::cout << "  \"number of epochs\" : \"" << num_epochs << "\"," << std::endl;
    std::cout << "  \"Training duration in seconds\" : \"" << duration.count() << "\"," << std::endl;

    int truePositive = 0;
    for (int i = 0; i < testData.size(); i++)
    {
        std::vector<double> output = model.cudaEvaluate(testData[i][0]);

        int prediction = 0;
        double maxOutput = output[0];
        for (int j = 1; j < 10; j++)
        {
            if (output[j] > maxOutput)
            {
                prediction = j;
                maxOutput = output[j];
            }
        }

        if (testData[i][1][prediction])
        {
            truePositive++;
        }
    }

    std::cout << "  \"Accuracy\" : \"" << 100.0 * truePositive / testingImages.size()<< "\"" << std::endl;
    std::cout << "}" << std::endl;
    return 0;
}