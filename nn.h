
#ifndef NN_HEADER
#define NN_HEADER

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <termios.h>
#include <unistd.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.0005f
#define MOMENTUM 0.9f
#define EPOCHS 20
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8
#define PRINT_INTERVAL 1000

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct {
    float *weights, *biases, *weight_momentum, *bias_momentum;
    int input_size, output_size;
} Layer;

typedef struct {
    Layer hidden, output;
} Network;

typedef struct {
    unsigned char *images, *labels;
    int nImages;
} InputData;


void softmax(float *input, int size);
void init_layer(Layer *layer, int in_size, int out_size);
void forward(Layer *layer, float *input, float *output);
void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr);
float* train(Network *net, float *input, int label, float lr);
int predict(Network *net, float *input);
void read_mnist_images(const char *filename, unsigned char **images, int *nImages);
void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels);
void shuffle_data(unsigned char *images, unsigned char *labels, int n);
void save_network(Network *net, const char *filename);
void load_network(Network *net, const char *filename);



// Include implementation if `NN_IMPLEMENTATION` is defined
#ifdef NN_IMPLEMENTATION




void softmax(float *input, int size) {
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
        input[i] /= sum;
}

void init_layer(Layer *layer, int in_size, int out_size) {
    int n = in_size * out_size;
    float scale = sqrtf(2.0f / in_size);

    layer->input_size = in_size;
    layer->output_size = out_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc(out_size, sizeof(float));
    layer->weight_momentum = calloc(n, sizeof(float));
    layer->bias_momentum = calloc(out_size, sizeof(float));

    for (int i = 0; i < n; i++)
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
}

void forward(Layer *layer, float *input, float *output) {
    for (int i = 0; i < layer->output_size; i++)
        output[i] = layer->biases[i];

    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        float *weight_row = &layer->weights[j * layer->output_size];
        for (int i = 0; i < layer->output_size; i++) {
            output[i] += in_j * weight_row[i];
        }
    }

    for (int i = 0; i < layer->output_size; i++)
        output[i] = output[i] > 0 ? output[i] : 0;
}


void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr) {
    if (input_grad) {
        for (int j = 0; j < layer->input_size; j++) {
            input_grad[j] = 0.0f;
            float *weight_row = &layer->weights[j * layer->output_size];
            for (int i = 0; i < layer->output_size; i++) {
                input_grad[j] += output_grad[i] * weight_row[i];
            }
        }
    }

    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        float *weight_row = &layer->weights[j * layer->output_size];
        float *momentum_row = &layer->weight_momentum[j * layer->output_size];
        for (int i = 0; i < layer->output_size; i++) {
            float grad = output_grad[i] * in_j;
            momentum_row[i] = MOMENTUM * momentum_row[i] + lr * grad;
            weight_row[i] -= momentum_row[i];
            if (input_grad)
                input_grad[j] += output_grad[i] * weight_row[i];
        }
    }

    for (int i = 0; i < layer->output_size; i++) {
        layer->bias_momentum[i] = MOMENTUM * layer->bias_momentum[i] + lr * output_grad[i];
        layer->biases[i] -= layer->bias_momentum[i];
    }
}


float* train(Network *net, float *input, int label, float lr) {
    static float final_output[OUTPUT_SIZE];
    float hidden_output[HIDDEN_SIZE];
    float output_grad[OUTPUT_SIZE] = {0}, hidden_grad[HIDDEN_SIZE] = {0};

    forward(&net->hidden, input, hidden_output);
    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        output_grad[i] = final_output[i] - (i == label);

    backward(&net->output, hidden_output, output_grad, hidden_grad, lr);

    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0;  // ReLU derivative

    backward(&net->hidden, input, hidden_grad, NULL, lr);

    return final_output;
}

int predict(Network *net, float *input) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

    forward(&net->hidden, input, hidden_output);
    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (final_output[i] > final_output[max_index])
            max_index = i;

    return max_index;
}

void read_mnist_images(const char *filename, unsigned char **images, int *nImages) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp, rows, cols;
    fread(&temp, sizeof(int), 1, file);
    fread(nImages, sizeof(int), 1, file);
    *nImages = __builtin_bswap32(*nImages);

    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);

    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    *images = malloc((*nImages) * IMAGE_SIZE * IMAGE_SIZE);
    fread(*images, sizeof(unsigned char), (*nImages) * IMAGE_SIZE * IMAGE_SIZE, file);
    fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp;
    fread(&temp, sizeof(int), 1, file);
    fread(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    *labels = malloc(*nLabels);
    fread(*labels, sizeof(unsigned char), *nLabels, file);
    fclose(file);
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++) {
            unsigned char temp = images[i * INPUT_SIZE + k];
            images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
            images[j * INPUT_SIZE + k] = temp;
        }
        unsigned char temp = labels[i];
        labels[i] = labels[j];
        labels[j] = temp;
    }
}

void save_network(Network *net, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) exit(1);

    fwrite(net->hidden.weights, sizeof(float), net->hidden.input_size * net->hidden.output_size, file);
    fwrite(net->hidden.biases, sizeof(float), net->hidden.output_size, file);
    fwrite(net->output.weights, sizeof(float), net->output.input_size * net->output.output_size, file);
    fwrite(net->output.biases, sizeof(float), net->output.output_size, file);

    fclose(file);
}

void load_network(Network *net, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    fread(net->hidden.weights, sizeof(float), net->hidden.input_size * net->hidden.output_size, file);
    fread(net->hidden.biases, sizeof(float), net->hidden.output_size, file);
    fread(net->output.weights, sizeof(float), net->output.input_size * net->output.output_size, file);
    fread(net->output.biases, sizeof(float), net->output.output_size, file);

    fclose(file);
}




void print_image(unsigned char *image) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            printf("%c", image[i * IMAGE_SIZE + j] > 128 ? '#' : '.');
        }
        printf("\n");
    }
}

int get_key_press() {
    struct termios oldt, newt;
    int ch;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}


/*
// train network.  optionally load and save
int main1() {
    Network net;
    InputData data = {0};
    float learning_rate = LEARNING_RATE, img[INPUT_SIZE];
    clock_t start, end;
    double cpu_time_used;

    srand(time(NULL));

    init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
    read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

    shuffle_data(data.images, data.labels, data.nImages);

    int train_size = (int)(data.nImages * TRAIN_SPLIT);
    int test_size = data.nImages - train_size;

    // Optionally load the neural network
    // load_network(&net, "network.dat");

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        start = clock();
        float total_loss = 0;
        for (int i = 0; i < train_size; i++) {
            for (int k = 0; k < INPUT_SIZE; k++)
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;

            float* final_output = train(&net, img, data.labels[i], learning_rate);
            total_loss += -logf(final_output[data.labels[i]] + 1e-10f);
        }
        int correct = 0;
        for (int i = train_size; i < data.nImages; i++) {
            for (int k = 0; k < INPUT_SIZE; k++)
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
            if (predict(&net, img) == data.labels[i])
                correct++;
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n", 
               epoch + 1, (float)correct / test_size * 100, total_loss / train_size, cpu_time_used);
    }

    // Optionally save the neural network
    save_network(&net, "network.dat");

    free(net.hidden.weights);
    free(net.hidden.biases);
    free(net.hidden.weight_momentum);
    free(net.hidden.bias_momentum);
    free(net.output.weights);
    free(net.output.biases);
    free(net.output.weight_momentum);
    free(net.output.bias_momentum);
    free(data.images);
    free(data.labels);

    return 0;
}



#define TEST_IMG_PATH "data/t10k-images.idx3-ubyte"
#define TEST_LBL_PATH "data/t10k-labels.idx1-ubyte"

int visualize_predictions() {
    Network net;
    unsigned char *images, *labels;
    int nImages, nLabels;

    srand(time(NULL));

    init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    // Optionally load the neural network
    load_network(&net, "network.dat");

    // Load the test data
    read_mnist_images(TEST_IMG_PATH, &images, &nImages);
    read_mnist_labels(TEST_LBL_PATH, &labels, &nLabels);

    if (nImages != nLabels) {
        fprintf(stderr, "Image and label counts do not match\n");
        return 1;
    }

    int index = 0;
    while (1) {
        system("clear");

        //	check_single_image(&net, images, labels, index);
        printf("Displaying and predicting image at index %d:\n\n", index);
        print_image(&images[index * INPUT_SIZE]);

        float img[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++)
        img[i] = images[index * INPUT_SIZE + i] / 255.0f;

        int prediction = predict(&net, img);
        printf("Predicted Label: %d\n", prediction);
        printf("Actual Label: %d\n\n", labels[index]);

        // // Uncomment to just look at incorrect predictions
        // if (prediction == labels[index]) {
        //     index = (index + 1) % nImages;
        //     continue;
        // }

        printf("Press spacebar for next image, or 'q' to quit.\n");

        int key = get_key_press();
        if (key == 'q') break;
        if (key == ' ') index = (index + 1) % nImages;
    }

    // Free resources
    free(net.hidden.weights);
    free(net.hidden.biases);
    free(net.output.weights);
    free(net.output.biases);
    free(images);
    free(labels);

    return 0;
}

*/





#endif // NN_IMPLEMENTATION

#endif // NN_HEADER


