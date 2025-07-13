#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <fstream>

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif



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

class OneVsOneClassifier {
public:
    int dim;
    int num_classes;
    int num_models;
    LinearModel** models;

    OneVsOneClassifier(int dim, int num_classes)
        : dim(dim), num_classes(num_classes) {
        num_models = num_classes * (num_classes - 1) / 2;
        models = new LinearModel*[num_models];
        for (int i = 0; i < num_models; i++)
            models[i] = new LinearModel(dim);
    }

    ~OneVsOneClassifier() {
        for (int i = 0; i < num_models; i++)
            delete models[i];
        delete[] models;
    }
};

int get_model_index(OneVsOneClassifier* ovo, int i, int j) {
    int n = ovo->num_classes;
    return i * n + j - ((i + 2) * (i + 1)) / 2;
}
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

    // multiclasse




DLLEXPORT OneVsOneClassifier* create_ovo_classifier(int32_t dim, int32_t num_classes) {
    return new OneVsOneClassifier(dim, num_classes);
}

DLLEXPORT void release_ovo_classifier(OneVsOneClassifier* ovo) {
    delete ovo;
}

DLLEXPORT void train_ovo_classifier(OneVsOneClassifier* ovo,
                                   float* X, int32_t* Y,
                                   int32_t num_samples,
                                   int32_t epochs,
                                   float learning_rate) {
    int dim = ovo->dim;
    int n_classes = ovo->num_classes;

    float* X_pair = new float[num_samples * dim];
    float* Y_pair = new float[num_samples];

    for (int i = 0; i < n_classes; i++) {
        for (int j = i + 1; j < n_classes; j++) {
            int n = 0;
            for (int k = 0; k < num_samples; k++) {
                if (Y[k] == i || Y[k] == j) {
                    for (int d = 0; d < dim; d++)
                        X_pair[n * dim + d] = X[k * dim + d];
                    Y_pair[n] = (Y[k] == i) ? 1.0f : -1.0f;
                    n++;
                }
            }
            int index = get_model_index(ovo, i, j);
            train_linear_model(ovo->models[index], X_pair, Y_pair, epochs, learning_rate, n);
        }
    }

    delete[] X_pair;
    delete[] Y_pair;
}

DLLEXPORT int32_t predict_ovo_classifier(OneVsOneClassifier* ovo, float* X) {
    int* votes = new int[ovo->num_classes]();
    int n_classes = ovo->num_classes;

    for (int i = 0; i < n_classes; i++) {
        for (int j = i + 1; j < n_classes; j++) {
            int idx = get_model_index(ovo, i, j);
            float pred = predict_linear_model(ovo->models[idx], X);
            votes[pred > 0 ? i : j]++;
        }
    }

    int max_class = 0;
    for (int i = 1; i < n_classes; i++) {
        if (votes[i] > votes[max_class]) max_class = i;
    }

    delete[] votes;
    return max_class;
}


}