#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <fstream>

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

typedef void (*LoggerCallback)(const char*);

LoggerCallback g_logger = nullptr;

void log_to_file(const char* message) {
    std::ofstream out("log_cpp.txt", std::ios::app);
    out << message << std::endl;
    out.close();
}

struct MLP {
    int* layer_sizes;
    int n_layers;
    float*** W;
    float** X;
    float** deltas;
};

inline float activate(float x) {
    return std::tanh(x);
}

// Dérivée à partir de la sortie déjà activée (tanh)
inline float activate_deriv_from_output(float activated_x) {
    return 1.0f - activated_x * activated_x;
}

void mlp_propagate(MLP* model, float* input, bool is_classification) {
    for (int j = 1; j <= model->layer_sizes[0]; ++j)
        model->X[0][j] = input[j - 1];

    for (int l = 1; l <= model->n_layers; ++l) {
        for (int j = 1; j <= model->layer_sizes[l]; ++j) {
            float sum = 0.0f;
            for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                sum += model->W[l][i][j] * model->X[l - 1][i];

            if (is_classification || l != model->n_layers)
                sum = activate(sum);

            model->X[l][j] = sum;
        }
    }
}

extern "C" {

    DLLEXPORT void set_logger(LoggerCallback cb) {
        g_logger = cb;
    }

    DLLEXPORT MLP* create_mlp_model(int* layer_sizes, int n_layers) {
        MLP* model = new MLP;
        model->n_layers = n_layers;
        model->layer_sizes = new int[n_layers + 1];
        std::memcpy(model->layer_sizes, layer_sizes, (n_layers + 1) * sizeof(int));

        model->W = new float**[n_layers + 1];
        model->X = new float*[n_layers + 1];
        model->deltas = new float*[n_layers + 1];

        for (int l = 0; l <= n_layers; ++l) {
            int cur_size = layer_sizes[l] + 1;
            model->X[l] = new float[cur_size];
            model->deltas[l] = new float[cur_size];

            std::fill(model->deltas[l], model->deltas[l] + cur_size, 0.0f);
            model->X[l][0] = 1.0f;  // biais
            for (int j = 1; j < cur_size; ++j)
                model->X[l][j] = 0.0f;

            if (l == 0) {
                model->W[l] = nullptr;
                continue;
            }

            int prev_size = layer_sizes[l - 1] + 1;
            model->W[l] = new float*[prev_size];

            for (int i = 0; i < prev_size; ++i) {
                model->W[l][i] = new float[cur_size];
                for (int j = 0; j < cur_size; ++j) {
                    // Initialisation aléatoire, y compris le biais
                    model->W[l][i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
                }
            }
        }

        return model;
    }

    DLLEXPORT float* predict_mlp_model(MLP* model, float* input, bool is_classification) {
        mlp_propagate(model, input, is_classification);
        int output_dim = model->layer_sizes[model->n_layers];
        float* output_predict = new float[output_dim];
        for (int j = 1; j <= output_dim; ++j)
            output_predict[j - 1] = model->X[model->n_layers][j];
        return output_predict;
    }

    DLLEXPORT void train_mlp_model(MLP* model, float* inputs, float* targets,
                                int epochs, float lr, int batch_size, bool is_classification,
                                float* test_inputs, float* test_targets, int test_size) {
    char buffer[512];
    sprintf(buffer, "Starting train: epochs=%d, batch_size=%d, lr=%f", epochs, batch_size, lr);
    log_to_file(buffer);

    int input_dim = model->layer_sizes[0];
    int output_dim = model->layer_sizes[model->n_layers];

    int* class_counts = new int[output_dim];
    float* class_weights = new float[output_dim];
    for (int i = 0; i < output_dim; ++i) {
        class_counts[i] = 0;
        class_weights[i] = 1.0f;
    }

    for (int i = 0; i < batch_size; ++i) {
        float* t = targets + i * output_dim;
        for (int c = 0; c < output_dim; ++c) {
            if (t[c] == 1.0f) {
                class_counts[c]++;
                break;
            }
        }
    }

    int total = batch_size;
    for (int c = 0; c < output_dim; ++c) {
        if (class_counts[c] > 0) {
            class_weights[c] = (float)total / (output_dim * class_counts[c]);
        }
    }

    for (int e = 0; e < epochs; ++e) {
        int k = rand() % batch_size;
        float* sample_input = inputs + k * input_dim;
        float* sample_target = targets + k * output_dim;

        mlp_propagate(model, sample_input, is_classification);

        for (int j = 1; j <= output_dim; ++j) {
            float out = model->X[model->n_layers][j];
            float target = sample_target[j - 1];
            float delta = out - target;

            if (is_classification)
                delta *= activate_deriv_from_output(out);

            if (is_classification && target == 1.0f)
                delta *= class_weights[j - 1];

            model->deltas[model->n_layers][j] = delta;
        }

        for (int l = model->n_layers - 1; l >= 1; --l) {
            for (int i = 1; i <= model->layer_sizes[l]; ++i) {
                float sum = 0.0f;
                for (int j = 1; j <= model->layer_sizes[l + 1]; ++j)
                    sum += model->W[l + 1][i][j] * model->deltas[l + 1][j];
                sum *= activate_deriv_from_output(model->X[l][i]);
                model->deltas[l][i] = sum;
            }
        }

        for (int l = 1; l <= model->n_layers; ++l) {
            for (int i = 0; i <= model->layer_sizes[l - 1]; ++i) {
                for (int j = 1; j <= model->layer_sizes[l]; ++j) {
                    model->W[l][i][j] -= lr * model->X[l - 1][i] * model->deltas[l][j];
                }
            }
        }

        if ((e + 1) % (epochs / 10) == 0 && g_logger) {
            int correct = 0;
            for (int i = 0; i < test_size; ++i) {
                float* test_input = test_inputs + i * input_dim;
                float* true_label = test_targets + i * output_dim;

                mlp_propagate(model, test_input, is_classification);

                int predicted = 0;
                float max_out = model->X[model->n_layers][1];
                for (int j = 2; j <= output_dim; ++j) {
                    if (model->X[model->n_layers][j] > max_out) {
                        max_out = model->X[model->n_layers][j];
                        predicted = j - 1;
                    }
                }

                int actual = 0;
                for (int j = 0; j < output_dim; ++j) {
                    if (true_label[j] == 1.0f) {
                        actual = j;
                        break;
                    }
                }

                if (predicted == actual)
                    correct++;
            }

            float acc = (float)correct / test_size * 100.0f;
            char msg[128];
            sprintf(msg, "Epoch %d: Accuracy = %.2f%%", e + 1, acc);
            g_logger(msg);
        }
    }

    delete[] class_counts;
    delete[] class_weights;
}


    DLLEXPORT void release_mlp_model(MLP* model) {
        for (int l = 1; l <= model->n_layers; ++l) {
            for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                delete[] model->W[l][i];
            delete[] model->W[l];
        }
        for (int l = 0; l <= model->n_layers; ++l) {
            delete[] model->X[l];
            delete[] model->deltas[l];
        }
        delete[] model->W;
        delete[] model->X;
        delete[] model->deltas;
        delete[] model->layer_sizes;
        delete model;
    }

}  // extern "C"
