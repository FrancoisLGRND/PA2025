#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <omp.h>

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

typedef void (*LoggerCallback)(const char*);
LoggerCallback g_logger = nullptr;

void log_to_csv(std::ofstream& out, int epoch, float train_acc, float train_loss, float test_acc, float test_loss) {
    out << epoch << "," << train_acc << "," << train_loss << "," << test_acc << "," << test_loss << "\n";
}

float activate(float x) {
    return std::tanh(x);
}

float activate_deriv_from_output(float activated_x) {
    return 1.0f - activated_x * activated_x;
}

class MLP {
public:
    int* layer_sizes;
    int n_layers;
    float*** W;
    int** confusion_matrix;

    MLP() : layer_sizes(nullptr), n_layers(0), W(nullptr), confusion_matrix(nullptr) {}
    ~MLP() { cleanup(); }

    void cleanup() {
        if (W) {
            for (int l = 1; l <= n_layers; ++l) {
                if (W[l]) {
                    for (int i = 0; i <= layer_sizes[l - 1]; ++i)
                        delete[] W[l][i];
                    delete[] W[l];
                }
            }
            delete[] W;
        }
        delete[] layer_sizes;
        if (confusion_matrix) {
            int output_dim = layer_sizes[n_layers];
            for (int i = 0; i < output_dim; ++i)
                delete[] confusion_matrix[i];
            delete[] confusion_matrix;
        }
    }

};

void mlp_propagate(const MLP* model, float* input, float** X_buffer) {
    for (int j = 1; j <= model->layer_sizes[0]; ++j)
        X_buffer[0][j] = input[j - 1];

    for (int l = 1; l <= model->n_layers; ++l) {
        #pragma omp parallel for
        for (int j = 1; j <= model->layer_sizes[l]; ++j) {
            float sum = 0.0f;
            for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                sum += model->W[l][i][j] * X_buffer[l - 1][i];
            X_buffer[l][j] = activate(sum);
        }
    }
}

extern "C" {

DLLEXPORT void set_logger(LoggerCallback cb) {
    g_logger = cb;
}

DLLEXPORT MLP* create_mlp_model(int* layer_sizes, int n_layers, int threads) {
    MLP* model = new MLP();
    model->n_layers = n_layers;
    model->layer_sizes = new int[n_layers + 1];
    std::memcpy(model->layer_sizes, layer_sizes, (n_layers + 1) * sizeof(int));
    omp_set_num_threads(threads);
    model->W = new float**[n_layers + 1];

    for (int l = 1; l <= n_layers; ++l) {
        int prev_size = layer_sizes[l - 1] + 1;
        int cur_size = layer_sizes[l] + 1;
        model->W[l] = new float*[prev_size];

        #pragma omp parallel for
        for (int i = 0; i < prev_size; ++i) {
            model->W[l][i] = new float[cur_size];
            for (int j = 0; j < cur_size; ++j)
                model->W[l][i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }

    int output_dim = layer_sizes[n_layers];
    model->confusion_matrix = new int*[output_dim];
    for (int i = 0; i < output_dim; ++i) {
        model->confusion_matrix[i] = new int[output_dim];
        std::memset(model->confusion_matrix[i], 0, output_dim * sizeof(int));
    }
    return model;
}

DLLEXPORT float* predict_mlp_model(MLP* model, float* input) {
    float** X = new float*[model->n_layers + 1];
    for (int l = 0; l <= model->n_layers; ++l) {
        X[l] = new float[model->layer_sizes[l] + 1]();
        X[l][0] = 1.0f;
    }
    mlp_propagate(model, input, X);
    int output_dim = model->layer_sizes[model->n_layers];
    float* output_predict = new float[output_dim];
    for (int j = 1; j <= output_dim; ++j)
        output_predict[j - 1] = X[model->n_layers][j];
    for (int l = 0; l <= model->n_layers; ++l)
        delete[] X[l];
    delete[] X;
    return output_predict;
}

DLLEXPORT void train_mlp_model(MLP* model, float* inputs, float* targets, int training_size, int epochs, float lr, int batch_size, float* test_inputs, float* test_targets, int test_size, char* csv_path) {
    int input_dim = model->layer_sizes[0];
    int output_dim = model->layer_sizes[model->n_layers];
    float best_test_loss = 10000;
    int epochs_without_improve = 0;
    int patience = 10;
    std::ofstream log_file;
    if (strlen(csv_path) > 0) {
        log_file.open(csv_path, std::ios::app);
    }


    float*** best_weights = new float**[model->n_layers + 1];
    for (int l = 1; l <= model->n_layers; ++l) {
        int prev_size = model->layer_sizes[l - 1] + 1;
        int cur_size = model->layer_sizes[l] + 1;
        best_weights[l] = new float*[prev_size];
        for (int i = 0; i < prev_size; ++i) {
            best_weights[l][i] = new float[cur_size];
        }
    }

    float**** grad_accum = new float***[model->n_layers + 1];
    for (int l = 1; l <= model->n_layers; ++l) {
        int prev_size = model->layer_sizes[l - 1] + 1;
        int cur_size = model->layer_sizes[l] + 1;
        grad_accum[l] = new float**[prev_size];
        for (int i = 0; i < prev_size; ++i) {
            grad_accum[l][i] = new float*[cur_size];
            for (int j = 0; j < cur_size; ++j)
                grad_accum[l][i][j] = new float(0.0f);
        }
    }



    float** X = new float*[model->n_layers + 1];
    float** deltas = new float*[model->n_layers + 1];
    for (int l = 0; l <= model->n_layers; ++l) {
        X[l] = new float[model->layer_sizes[l] + 1]();
        deltas[l] = new float[model->layer_sizes[l] + 1]();
        X[l][0] = 1.0f;
    }

    for (int e = 0; e < epochs; ++e) {
        for (int l = 1; l <= model->n_layers; ++l)
            #pragma omp parallel for
            for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                for (int j = 1; j <= model->layer_sizes[l]; ++j)
                    *grad_accum[l][i][j] = 0.0f;

        for (int b = 0; b < batch_size; ++b) {
            int k = rand() % training_size;
            float* xi = inputs + k * input_dim;
            float* ti = targets + k * output_dim;

            mlp_propagate(model, xi, X);

            for (int j = 1; j <= output_dim; ++j)
                deltas[model->n_layers][j] = X[model->n_layers][j] - ti[j - 1];

            for (int l = model->n_layers - 1; l >= 1; --l)
                for (int i = 1; i <= model->layer_sizes[l]; ++i) {
                    float s = 0.0f;
                    for (int j = 1; j <= model->layer_sizes[l + 1]; ++j)
                        s += model->W[l + 1][i][j] * deltas[l + 1][j];
                    deltas[l][i] = s * activate_deriv_from_output(X[l][i]);
                }

            for (int l = 1; l <= model->n_layers; ++l)
                for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                    for (int j = 1; j <= model->layer_sizes[l]; ++j)
                        *grad_accum[l][i][j] += X[l - 1][i] * deltas[l][j];
        }

        for (int l = 1; l <= model->n_layers; ++l)
            #pragma omp parallel for
            for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                for (int j = 1; j <= model->layer_sizes[l]; ++j)
                    model->W[l][i][j] -= (lr / batch_size) * (*grad_accum[l][i][j]);

        if (g_logger && ((e + 1) % 10 == 0)) {
            int train_correct = 0;
            float train_total_loss = 0.0f;

            for (int b = 0; b < batch_size; ++b) {
                int k = rand() % training_size;
                float* xi = inputs + k * input_dim;
                float* ti = targets + k * output_dim;

                mlp_propagate(model, xi, X);

                for (int j = 1; j <= output_dim; ++j) {
                    float o = X[model->n_layers][j];
                    float t = ti[j - 1];
                    float diff = o - t;
                    train_total_loss += diff * diff;
                }

                int pred = 0;
                float mv = X[model->n_layers][1];
                for (int j = 2; j <= output_dim; ++j) {
                    if (X[model->n_layers][j] > mv) {
                        mv = X[model->n_layers][j];
                        pred = j - 1;
                    }
                }

                int act = 0;
                for (int j = 0; j < output_dim; ++j) {
                    if (ti[j] == 1.0f) {
                        act = j;
                        break;
                    }
                }

                if (pred == act) train_correct++;
            }

            int correct = 0;
            float total_loss = 0.0f;
            for (int i = 0; i < test_size; ++i) {
                float* xi = test_inputs + i * input_dim;
                float* ti = test_targets + i * output_dim;
                mlp_propagate(model, xi, X);
                for (int j = 1; j <= output_dim; ++j) {
                    float o = X[model->n_layers][j];
                    float t = ti[j - 1];
                    float diff = o - t;
                    total_loss += diff * diff;
                }
                int pred = 0;
                float mv = X[model->n_layers][1];
                for (int j = 2; j <= output_dim; ++j)
                    if (X[model->n_layers][j] > mv) {
                        mv = X[model->n_layers][j];
                        pred = j - 1;
                    }
                int act = 0;
                for (int j = 0; j < output_dim; ++j)
                    if (ti[j] == 1.0f) {
                        act = j;
                        break;
                    }
                if (pred == act) correct++;
            }
            float train_acc = static_cast<float>(train_correct) / batch_size * 100.0f;
            float train_loss = train_total_loss / batch_size;
            float test_acc = static_cast<float>(correct) / test_size * 100.0f;
            float test_loss = total_loss / test_size;

            if (test_loss < best_test_loss) {
                best_test_loss = test_loss;

                for (int l = 1; l <= model->n_layers; ++l) {
                    int prev_size = model->layer_sizes[l - 1] + 1;
                    int cur_size = model->layer_sizes[l] + 1;
                    for (int i = 0; i < prev_size; ++i)
                        for (int j = 0; j < cur_size; ++j)
                            best_weights[l][i][j] = model->W[l][i][j];
                }

            }

            char buff2[256];
            sprintf(buff2, "Epoch %d: train_acc = %.2f%%, train_loss = %.4f | test_acc = %.2f%%, test_loss = %.4f",
                    e + 1, train_acc, train_loss, test_acc, test_loss);
            g_logger(buff2);
            if (strlen(csv_path) > 0) {
                log_to_csv(log_file, e + 1, train_acc, train_loss, test_acc, test_loss);
            }
        }
    }

    for (int l = 1; l <= model->n_layers; ++l) {
        int prev_size = model->layer_sizes[l - 1] + 1;
        int cur_size = model->layer_sizes[l] + 1;
        for (int i = 0; i < prev_size; ++i)
            for (int j = 0; j < cur_size; ++j)
                model->W[l][i][j] = best_weights[l][i][j];
    }


    for (int l = 1; l <= model->n_layers; ++l) {
        int prev_size = model->layer_sizes[l - 1] + 1;
        for (int i = 0; i < prev_size; ++i)
            delete[] best_weights[l][i];
        delete[] best_weights[l];
    }
    delete[] best_weights;

    for (int l = 0; l <= model->n_layers; ++l) {
        delete[] X[l];
        delete[] deltas[l];
    }
    delete[] X;
    delete[] deltas;

    for (int l = 1; l <= model->n_layers; ++l) {
        int prev_size = model->layer_sizes[l - 1] + 1;
        int cur_size = model->layer_sizes[l] + 1;
        for (int i = 0; i < prev_size; ++i) {
            for (int j = 0; j < cur_size; ++j) {
                delete grad_accum[l][i][j];
            }
            delete[] grad_accum[l][i];
        }
        delete[] grad_accum[l];
    }
    delete[] grad_accum;
}

DLLEXPORT void save_mlp_model(MLP* model, const char* filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return;
    out.write((char*)&model->n_layers, sizeof(int));
    out.write((char*)model->layer_sizes, (model->n_layers + 1) * sizeof(int));
    for (int l = 1; l <= model->n_layers; ++l) {
        int rows = model->layer_sizes[l - 1] + 1;
        int cols = model->layer_sizes[l] + 1;
        for (int i = 0; i < rows; ++i)
            out.write((char*)model->W[l][i], cols * sizeof(float));
    }
}

DLLEXPORT MLP* load_mlp_model(const char* filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return nullptr;
    int n_layers = 0;
    in.read((char*)&n_layers, sizeof(int));
    int* layer_sizes = new int[n_layers + 1];
    in.read((char*)layer_sizes, (n_layers + 1) * sizeof(int));
    MLP* model = create_mlp_model(layer_sizes, n_layers);
    delete[] layer_sizes;
    for (int l = 1; l <= model->n_layers; ++l) {
        int rows = model->layer_sizes[l - 1] + 1;
        int cols = model->layer_sizes[l] + 1;
        for (int i = 0; i < rows; ++i)
            in.read((char*)model->W[l][i], cols * sizeof(float));
    }
    return model;
}

    DLLEXPORT void evaluate_confusion_matrix(MLP* model, float* test_inputs, float* test_targets, int test_size) {
    int input_dim = model->layer_sizes[0];
    int output_dim = model->layer_sizes[model->n_layers];

    float** X = new float*[model->n_layers + 1];
    for (int l = 0; l <= model->n_layers; ++l) {
        X[l] = new float[model->layer_sizes[l] + 1]();
        X[l][0] = 1.0f;
    }

    for (int i = 0; i < output_dim; ++i)
        std::memset(model->confusion_matrix[i], 0, output_dim * sizeof(int));

    for (int i = 0; i < test_size; ++i) {
        float* xi = test_inputs + i * input_dim;
        float* ti = test_targets + i * output_dim;

        mlp_propagate(model, xi, X);

        int pred = 0;
        float mv = X[model->n_layers][1];
        for (int j = 2; j <= output_dim; ++j) {
            if (X[model->n_layers][j] > mv) {
                mv = X[model->n_layers][j];
                pred = j - 1;
            }
        }

        int act = 0;
        for (int j = 0; j < output_dim; ++j) {
            if (ti[j] == 1.0f) {
                act = j;
                break;
            }
        }

        model->confusion_matrix[act][pred]++;
    }

    for (int l = 0; l <= model->n_layers; ++l)
        delete[] X[l];
    delete[] X;
}

    DLLEXPORT void get_confusion_matrix(MLP* model, int* out_matrix) {
    int od = model->layer_sizes[model->n_layers];
    for (int i = 0; i < od; ++i)
        for (int j = 0; j < od; ++j)
            out_matrix[i * od + j] = model->confusion_matrix[i][j];
}


DLLEXPORT void release_mlp_model(MLP* model) {
    delete model;
}

}
