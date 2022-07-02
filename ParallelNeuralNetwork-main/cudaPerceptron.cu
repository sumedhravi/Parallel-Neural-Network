
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <chrono>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>

class node
{
public:
    std::vector<double> weights;
    double bias;

    double activation;
    double output;
    double grad;
};

class layer
{
public:
    std::vector<node> nodes;
};

/* Activation function */
double sigmoid(double input, bool derivative = false);

/* Neural Network class */
class neural_network
{
public:
    std::default_random_engine mRng;
    std::vector<layer> layers;

    neural_network(unsigned long seed, const std::vector<int>& layerSizes);

    void randomize_weights();
    std::vector<double> forward_propagate(const std::vector<double>& expected);
    std::vector<double> cuda_forward_propagate(const std::vector<double>& inputActivations);
    void backpropagate_error(const std::vector<double>& input, const std::vector<double>& output);
    void update_weights(const std::vector<double>& input, double learningRate);
    void create_network(const std::vector<int>& nodesPerLayer);
};

/* Merge networks run on different threads by averaging the weights */
neural_network merge_networks(const std::vector<neural_network>& nns);

neural_network::neural_network(unsigned long seed, const std::vector<int>& layerSizes)
{
    mRng.seed(seed);
    create_network(layerSizes);
}

void neural_network::create_network(const std::vector<int>& layerSizes)
{
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
void neural_network::randomize_weights()
{
    std::uniform_real_distribution<double> u01(0, 1);

    for (int layer = 0; layer < layers.size(); layer++)
    {
        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            /* Set bias component of each node to 0 */
            layers[layer].nodes[node].bias = 0.0;
            /* Set random weight */
            for (int i = 0; i < layers[layer].nodes[node].weights.size(); i++)
            {
                layers[layer].nodes[node].weights[i] = 2 * (u01(mRng) - 0.5);
            }
        }
    }
}

/* Forward pass */
std::vector<double> neural_network::forward_propagate(const std::vector<double>& inputActivations)
{
    std::vector<double> input = inputActivations;

    for (int layer = 0; layer < layers.size(); layer++)
    {
        std::vector<double> newInput(layers[layer].nodes.size());

        // CUDA
        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            // CUDA
            double activation = layers[layer].nodes[node].bias;
            for (int i = 0; i < input.size(); i++)
            {
                activation += layers[layer].nodes[node].weights[i] * input[i];
            }
            layers[layer].nodes[node].activation = activation;
            layers[layer].nodes[node].output = sigmoid(layers[layer].nodes[node].activation, false);

            newInput[node] = layers[layer].nodes[node].output;
        }
        input = newInput;
    }

    return input;
}

__global__ void reduce0(double *g_idata, double *g_odata)
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2)
    {
         if (tid % (2*s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void vectorProduct(double *input_activations, double *weights, double *output, int N)
{
    int curr = (blockIdx.x * blockDim.x) + threadIdx.x;

    //printf("blockidx.x %d blockidx.y %d blockdim.x %d blockdim.y %d threadIdx.x %d threadIdx.y %d i %d j %d N %d \n",blockIdx.x, blockIdx.y, blockDim.x, blockDim.y,threadIdx.x, threadIdx.y, i, j, curr);
    {
        if (curr<N)
        {
            output[curr] = input_activations[curr] * weights [curr];
        }
    }
}

/* Forward pass */
std::vector<double> neural_network::cuda_forward_propagate(const std::vector<double>& inputActivations)
{
    std::vector<double> input = inputActivations;
    double *in = &input[0];
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);

    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, 0);
    static int debug = 0;

    for (int layer = 0; layer < layers.size(); layer++)
    {
        std::vector<double> newInput(layers[layer].nodes.size());

        // CUDA
        double *D_input_activations = NULL;
        long size = input.size() * sizeof(double);
        int err = cudaMalloc((void**)&D_input_activations, size);
        err = cudaMemcpy(D_input_activations, (double*)in, size, cudaMemcpyHostToDevice);

        double *D_weights = NULL;
        err = cudaMalloc((void**)&D_weights, size);

        double *D_output_activations = NULL;
        err = cudaMalloc((void**)&D_output_activations, size);

        double *D_reductions = NULL;
        err = cudaMalloc((void**)&D_reductions, sizeof(double));

        int numBlocks = layers[layer].nodes.size();

        for (int node = 0; node < layers[layer].nodes.size(); node++)
        {
            // CUDA
            double activation = layers[layer].nodes[node].bias;

            double *w = &layers[layer].nodes[node].weights[0];
            err = cudaMemcpy(D_weights, w, size, cudaMemcpyHostToDevice);

            vectorProduct<<<numBlocks, input.size()>>>(D_input_activations, D_weights, D_output_activations, input.size());

            // double H_activations[input.size()];
            // err = cudaMemcpy(H_activations, D_output_activations, size, cudaMemcpyDeviceToHost);

            // for (int i = 0; i < input.size(); i++)
            // {
            //     activation += H_activations[i];
            // }

            double reduction[2];
            reduce0<<<numBlocks,input.size(), input.size()>>>(D_output_activations, D_reductions);
            err = cudaMemcpy((void*)reduction, D_reductions, sizeof(double), cudaMemcpyDeviceToHost);
            layers[layer].nodes[node].activation = activation + reduction[0];//+ (*(double*)reduction);
            layers[layer].nodes[node].output = sigmoid(layers[layer].nodes[node].activation, false);

            newInput[node] = layers[layer].nodes[node].output;
        }
        input = newInput;
    }

    // if(debug%1000 == 0){
    //     std::cout<<"completed images "<<debug<<std::endl;
    // }
    //     debug++;

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
void neural_network::backpropagate_error(const std::vector<double>& input, const std::vector<double>& expected)
{
    std::vector<double> output = cuda_forward_propagate(input);

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
                for (int i = 0; i < layers[layer + 1].nodes.size(); i++)
                {
                    error += layers[layer + 1].nodes[i].weights[node] * layers[layer + 1].nodes[i].grad;
                }
            }
            layers[layer].nodes[node].grad = error * sigmoid(layers[layer].nodes[node].output, true);
        }
    }
}

/* Update weights according to calculated gradients */
void neural_network::update_weights(const std::vector<double>& input, double learningRate)
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

neural_network merge_networks(const std::vector<neural_network>& nns)
{
    neural_network nn = nns[0];
    for (int i = 0; i < nn.layers.size(); i++)
    {
        for (int j = 0; j < nn.layers[i].nodes.size(); j++)
        {
            for (int l = 1; l < nns.size(); l++)
            {
                nn.layers[i].nodes[j].bias += nns[l].layers[i].nodes[j].bias;
            }
            nn.layers[i].nodes[j].bias /= nns.size();
            for (int k = 0; k < nn.layers[i].nodes[j].weights.size(); k++)
            {
                for (int l = 1; l < nns.size(); l++)
                {
                    nn.layers[i].nodes[j].weights[k] += nns[l].layers[i].nodes[j].weights[k];
                }
                nn.layers[i].nodes[j].weights[k] /= nns.size();
            }
        }
    }
    return nn;
}

int convertToInt(char *number)
{
    return ((number[0] & 0xff) << 24) | ((number[1] & 0xff) << 16) | ((number[2] & 0xff) << 8) | ((number[3] & 0xff) << 0);
};

std::vector<std::vector<double>> readMnistImages(const std::string& filename)
{
    std::vector<std::vector<char>> imageData;

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.good())
    {
        std::cout << "Failed to load images from '" << filename << "'\n";
        return std::vector<std::vector<double>>();
    }

    char magic[4];
    ifs.read((char*)magic, 4);
    char nImages[4];
    ifs.read((char*)nImages, 4);
    char nRows[4];
    ifs.read((char*)nRows, 4);
    char nCols[4];
    ifs.read((char*)nCols, 4);

    int magicNumber = convertToInt(magic);
    int numImages = convertToInt(nImages);
    int numRows = convertToInt(nRows);
    int numCols = convertToInt(nCols);

    int numPixels = numImages * numRows * numCols;
    std::vector<char> pixelData(numPixels);
    ifs.read(pixelData.data(), numImages * numRows * numCols);

    // TODO: Parallel
    for (int i = 0; i < numImages; i++)
    {
        imageData.push_back(std::vector<char>(pixelData.begin() + (i * numRows * numCols), pixelData.begin() + ((i + 1) * numRows * numCols)));
    }

    std::vector<std::vector<double>> imageDataDouble(imageData.size(), std::vector<double>(imageData[0].size()));
    for (int i = 0; i < imageData.size(); i++)
    {
        for (int j = 0; j < imageData[0].size(); j++)
        {
            /* Scaling pixel values between 0 and 1 */
            imageDataDouble[i][j] = ((double)((unsigned char)imageData[i][j])) / 255.0;
        }
    }

    return imageDataDouble;
}

std::vector<int> readMnistLabels(const std::string& filename)
{
    std::vector<char> labelData;

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.good())
    {
        std::cout << "Failed to load labels from '" << filename << "'\n";
        return std::vector<int>();
    }

    char magic[4];
    ifs.read((char*)magic, 4);
    char nImages[4];
    ifs.read((char*)nImages, 4);
    int magicNumber = convertToInt(magic);
    int numImages = convertToInt(nImages);
    labelData.resize(numImages);

    ifs.read(labelData.data(), numImages);
    std::vector<int> labelDataInt(labelData.size());
    for (int i = 0; i < labelData.size(); i++)
    {
        labelDataInt[i] = ((int)(unsigned char)labelData[i]);
    }

    return labelDataInt;
}

class CONFIG
{
public:
    int numOfEpochs = 1;
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

    /* Load image and label data */
    std::cout << "Loading data" << std::endl;
    // std::vector<std::vector<double>> trainingImages = readMnistImages("../data/train-images.idx3-ubyte");
    std::vector<std::vector<double>> trainingImages = readMnistImages(config.trainImagePath);
    // std::vector<double> trainingLabels = readMnistLabels("../data/train-labels.idx1-ubyte");
    std::vector<int> trainingLabels = readMnistLabels(config.trainLabelsPath);
    // std::vector<std::vector<double>> testingImages = readMnistImages("../data/t10k-images.idx3-ubyte");
    std::vector<std::vector<double>> testingImages = readMnistImages(config.testImagePath);
    // std::vector<double> testingLabels = readMnistLabels("../data/t10k-labels.idx1-ubyte");
    std::vector<int> testingLabels = readMnistLabels(config.testLabelsPath);

    if (trainingImages.empty() || trainingLabels.empty() || testingImages.empty() || testingLabels.empty())
    {
        std::cout << "Error loading data" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << trainingImages.size() << " training images" << std::endl;
    std::cout << "Loaded " << testingImages.size() << " testing images" << std::endl;
    std::cout << "Loaded " << trainingLabels.size() << " training labels" << std::endl;
    std::cout << "Loaded " << testingLabels.size() << " testing labels" << std::endl;

    std::vector<std::vector<std::vector<double>>> trainData;
    //std::vector<double> output(10, 0.0);
    for (int i = 0; i < trainingImages.size(); i++)
    {
        std::vector<double> output(10, 0.0);
        output[trainingLabels[i]] = 1.0;
        trainData.push_back({ trainingImages[i], output, { double(i) } } );
    }

    std::vector<std::vector<std::vector<double>>> testData;
    for (int i = 0; i < testingImages.size(); i++)
    {
        std::vector<double> output(10, 0.0);
        output[testingLabels[i]] = 1.0;
        testData.push_back({ testingImages[i], output, { double(i) } } );
    }

    neural_network nn = neural_network(1234, {784, 10, 10});
    nn.randomize_weights();

    int numThreads = 1;
    int batchLength = 100;
    std::default_random_engine rng(11162021);
    std::cout << "Begin training on " << numThreads << " thread(s)" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    if (config.executeParallel)
    {
        std::vector<neural_network> nns;
        for (int n = 0; n < numThreads; n++)
        {
            nns.push_back(nn);
        }
        for (int epoch = 0; epoch < config.numOfEpochs; epoch++)
        {
            std::cout << "Epoch " << (epoch+1) << " / " << config.numOfEpochs << std::endl;
            std::shuffle(trainData.begin(), trainData.end(), rng);

            for (int i = 0; i < trainData.size(); i += numThreads * batchLength)
            {
                for (int n = 0; n < numThreads; n++)
                {
                    nns[n] = nn;
                }
                for (int n = 0; n < numThreads; n++)
                {
                    for (int m = 0; m < batchLength; m++)
                    {
                        int idx = i + n * batchLength + m;
                        if (idx < trainData.size())
                        {
                            nns[n].backpropagate_error(trainData[idx][0], trainData[idx][1]);
                            nns[n].update_weights(trainData[idx][0], config.learningRate);
                        }
                    }
                }
                nn = merge_networks(nns);
            }
        }
    }
    else
    {
        for (int epoch = 0; epoch < config.numOfEpochs; epoch++)
        {
            std::cout << "Epoch " << (epoch+1) << " / " << config.numOfEpochs << std::endl;
            std::shuffle(trainData.begin(), trainData.end(), rng);
            for (int i = 0; i < trainData.size(); i += 1)
            {
                if (i < trainData.size())
                {
                    nn.backpropagate_error(trainData[i][0], trainData[i][1]);
                    nn.update_weights(trainData[i][0], config.learningRate);
                }
                else break;
            }
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(endTime - startTime);
    std::cout << "Training duration: " << duration.count() << " s" << std::endl;

    std::cout << "Begin testing" << std::endl;
    int truePositive = 0;
    for (int i = 0; i < testData.size(); i++)
    {
        std::vector<double> output = nn.cuda_forward_propagate(testData[i][0]);

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

    std::cout << "Accuracy : " << 100.0 * truePositive / testingImages.size() << " % ( " << truePositive << " / " << testingImages.size() << " )" << std::endl;

    return 0;
}
