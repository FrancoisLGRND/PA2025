#include <cstdint>
#include <cstdlib>

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <stdio.h>
#include <string.h>
#include <fstream>

void log_to_file(const char* message) {
    std::ofstream out("log_cpp.txt", std::ios::app);
    out << message << std::endl;
    out.close();
}

float random_float(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}

float* define_bias(float* X, int dim){
    float* array = new float[dim+1];
    array[0] = 1.0;
    for(int i = 0; i < dim; i++){
        array[i + 1] = X[i];
    }
    return array;
}

class LinearModel {
public:
    int dim;
    float* w;

    LinearModel(int dim) {
        this->dim = dim;
        w = new float[dim + 1]; // On ajoute un poids pour le biais
        for(int i = 0; i < dim + 1; i++){
            w[i] = random_float(-1.0, 1.0);
        }
    }

    ~LinearModel() {
        delete[] w;
    }
};

extern "C" {
    DLLEXPORT LinearModel* create_linear_model(int32_t dim) {
        return new LinearModel(dim);
    }

    DLLEXPORT float predict_linear_model(LinearModel* model, float* X) {
        float sum = model->w[0] * 1.0f;
        for(int i = 1; i <= model->dim; ++i)
            sum += model->w[i] * X[i-1];
        return sum >= 0 ? 1.0f : -1.0f;
    }

    DLLEXPORT void train_linear_model(LinearModel *model, float* X, float *Y, int32_t epochs, float learning_rate, int32_t batch_size) {
            char buffer[512];
            sprintf(buffer, "Epoch %d, Batch Size %d, learning rate %f, X %f, Y %f ", epochs, batch_size, learning_rate, X[50], Y[2]);
            log_to_file(buffer);
            for (int i = 0; i < epochs; i++) {
                for (int j = 0; j < batch_size; j++) {
                    float* Xj = &X[j * model->dim];
                    float* Xj_bias = define_bias(Xj, model->dim);

                    float dot = 0.0f;
                    for (int k = 0; k < model->dim + 1; k++)
                        dot += model->w[k] * Xj_bias[k];

                    float predicted = dot >= 0 ? 1.0f : -1.0f;

                    if (predicted != Y[j]) {
                        for (int k = 0; k < model->dim + 1; k++) {
                            model->w[k] += learning_rate * Y[j] * Xj_bias[k];
                        }
                    }
                    if (i % 10 == 0 && j == 0) {
                        char buffer[512];
                        sprintf(buffer, "Epoch %d, weights: ", i);
                        for (int k = 0; k < model->dim + 1; k++)
                            sprintf(buffer + strlen(buffer), "%f ", model->w[k]);
                        log_to_file(buffer);
                    }
                    delete[] Xj_bias;
                }
            }
        }

    DLLEXPORT void release_linear_model(LinearModel *model) {
        delete model;
    }
}