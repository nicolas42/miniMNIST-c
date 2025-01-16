// clear && gcc -O3 -march=native -ffast-math main.c -lm && ./a.out
// clang -Ofast main.c && ./a.out

#define NN_IMPLEMENTATION
#include "nn.h"



int visualize_predictions();
int train_network(int,int);



int main() {
  int load = 0, save = 1;
  train_network(load, save);
  visualize_predictions();
  return 0;
}



int train_network(int do_load, int do_save) {
  // optionally load and save
  
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

    if (do_load) {
      load_network(&net, "network.dat");
    }

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

    if (do_save) {
      save_network(&net, "network.dat");
    }
    
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

        // Uncomment to look at incorrect predictions only
        if (prediction == labels[index]) {
            index = (index + 1) % nImages;
            continue;
        }

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

