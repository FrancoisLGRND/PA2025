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

void log_to_csv(int epoch, float accuracy, float loss) {
    std::ofstream out("log.csv", std::ios::app);
    out << epoch << "," << accuracy << "," << loss << "\n";
}

// Activation tanh et dérivée pour backprop
float activate(float x) {
    return std::tanh(x);
}

float activate_deriv_from_output(float activated_x) {
    return 1.0f - activated_x * activated_x;
}


// Softmax pour sortie en classification
void softmax(float* input, int size) {
    float max_val = input[1];
    for (int i = 2; i <= size; ++i)
        if (input[i] > max_val)
            max_val = input[i];

    float sum = 0.0f;
    for (int i = 1; i <= size; ++i) {
        input[i] = std::exp(input[i] - max_val);
        sum += input[i];
    }

    for (int i = 1; i <= size; ++i)
        input[i] /= sum;
}

class MLP {
public:
    int* layer_sizes;
    int n_layers;
    float*** W;
    float** X;
    float** deltas;
    int** confusion_matrix;

    MLP() {
        layer_sizes = nullptr;
        n_layers = 0;
        W = nullptr;
        X = nullptr;
        deltas = nullptr;
        confusion_matrix = nullptr;
    }

    ~MLP() {
        cleanup();
    }

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

        if (X) {
            for (int l = 0; l <= n_layers; ++l)
                delete[] X[l];
            delete[] X;
        }

        if (deltas) {
            for (int l = 0; l <= n_layers; ++l)
                delete[] deltas[l];
            delete[] deltas;
        }

        if (confusion_matrix) {
            for (int i = 0; i < layer_sizes[n_layers]; ++i)
                delete[] confusion_matrix[i];
            delete[] confusion_matrix;
        }

        delete[] layer_sizes;
    }
};
void mlp_propagate(MLP* model, float* input, bool is_classification) {
    int num_classes = model->layer_sizes[model->n_layers];
    bool use_softmax = is_classification && num_classes > 2;

    for (int j = 1; j <= model->layer_sizes[0]; ++j)
        model->X[0][j] = input[j - 1];

    for (int l = 1; l <= model->n_layers; ++l) {
        for (int j = 1; j <= model->layer_sizes[l]; ++j) {
            float sum = 0.0f;
            for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                sum += model->W[l][i][j] * model->X[l - 1][i];
            model->X[l][j] = (l == model->n_layers) ? sum : activate(sum);
        }
    }

    if (use_softmax)
        softmax(model->X[model->n_layers], model->layer_sizes[model->n_layers]);
}

extern "C" {

DLLEXPORT void set_logger(LoggerCallback cb) {
    g_logger = cb;
}

DLLEXPORT MLP* create_mlp_model(int* layer_sizes, int n_layers) {
    MLP* model = new MLP();
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
        model->X[l][0] = 1.0f;
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
            for (int j = 0; j < cur_size; ++j)
                model->W[l][i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }

    int output_dim = model->layer_sizes[n_layers];
    model->confusion_matrix = new int*[output_dim];
    for (int i = 0; i < output_dim; ++i) {
        model->confusion_matrix[i] = new int[output_dim];
        std::fill(model->confusion_matrix[i], model->confusion_matrix[i] + output_dim, 0);
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
                               int training_size, int epochs, float lr, int batch_size, bool is_classification,
                               float* test_inputs, float* test_targets, int test_size) {
    int input_dim = model->layer_sizes[0];
    int output_dim = model->layer_sizes[model->n_layers];
    bool use_softmax = is_classification && output_dim > 2;


    std::vector<std::vector<std::vector<float>>> grad_accum(model->n_layers + 1);
    for (int l = 1; l <= model->n_layers; ++l) {
        int prev_size = model->layer_sizes[l - 1] + 1;
        int cur_size = model->layer_sizes[l] + 1;
        grad_accum[l].resize(prev_size, std::vector<float>(cur_size, 0.0f));
    }

    for (int e = 0; e < epochs; ++e) {

        for (int l = 1; l <= model->n_layers; ++l)
            for (auto& row : grad_accum[l])
                std::fill(row.begin(), row.end(), 0.0f);

        std::vector<std::vector<float>> local_X(model->n_layers + 1);
        std::vector<std::vector<float>> local_deltas(model->n_layers + 1);
        for (int l = 0; l <= model->n_layers; ++l) {
            local_X[l].resize(model->layer_sizes[l] + 1);
            local_deltas[l].resize(model->layer_sizes[l] + 1);
            local_X[l][0] = 1.0f;
        }

        for (int b = 0; b < batch_size; ++b) {
            int k = rand() % training_size;
            float* xi = inputs + k * input_dim;
            float* ti = targets + k * output_dim;

            for (int j = 1; j <= input_dim; ++j)
                local_X[0][j] = xi[j - 1];

            for (int l = 1; l <= model->n_layers; ++l) {
                for (int j = 1; j <= model->layer_sizes[l]; ++j) {
                    float sum = 0.0f;
                    for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                        sum += model->W[l][i][j] * local_X[l - 1][i];
                    local_X[l][j] = (l == model->n_layers) ? sum : activate(sum);
                }
            }

            if (use_softmax){
                softmax(local_X[model->n_layers].data(), output_dim);
            }

            for (int j = 1; j <= output_dim; ++j) {
                float out = local_X[model->n_layers][j];
                float tgt = ti[j - 1];
                local_deltas[model->n_layers][j] = out - tgt;
            }

            for (int l = model->n_layers - 1; l >= 1; --l) {
                for (int i = 1; i <= model->layer_sizes[l]; ++i) {
                    float s = 0.0f;
                    for (int j = 1; j <= model->layer_sizes[l + 1]; ++j)
                        s += model->W[l + 1][i][j] * local_deltas[l + 1][j];
                    local_deltas[l][i] = s * activate_deriv_from_output(local_X[l][i]);
                }
            }

            for (int l = 1; l <= model->n_layers; ++l)
                for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                    for (int j = 1; j <= model->layer_sizes[l]; ++j)
                        grad_accum[l][i][j] += local_X[l - 1][i] * local_deltas[l][j];
        }

        for (int l = 1; l <= model->n_layers; ++l)
            for (int i = 0; i <= model->layer_sizes[l - 1]; ++i)
                for (int j = 1; j <= model->layer_sizes[l]; ++j)
                    model->W[l][i][j] -= (lr / batch_size) * grad_accum[l][i][j];

        if (g_logger && (e+1) % (epochs/100) == 0 || epochs<100) {
            int correct = 0;
            float total_loss = 0.0f;
            for (int i = 0; i < test_size; ++i) {
                float* xi = test_inputs + i * input_dim;
                float* ti = test_targets + i * output_dim;
                mlp_propagate(model, xi, is_classification);
                for (int j = 1; j <= output_dim; ++j) {
                    float o = model->X[model->n_layers][j];
                    float t = ti[j - 1];
                    float c = std::max(std::min(o, 1.0f-1e-8f), 1e-8f);
                    total_loss += -t * std::log(c);
                }
                int pred = 0;
                float mv = model->X[model->n_layers][1];
                for (int j = 2; j <= output_dim; ++j)
                    if (model->X[model->n_layers][j] > mv) {
                        mv = model->X[model->n_layers][j];
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
            float acc = (float)correct / test_size * 100.0f;
            float avg_loss = total_loss / test_size;
            char buff2[128];
            sprintf(buff2, "Epoch %d: acc = %.2f%%, loss = %.4f", e+1, acc, avg_loss);
            g_logger(buff2);
            log_to_csv(e+1, acc, avg_loss);
        }
    }
}

DLLEXPORT void evaluate_confusion_matrix(MLP* model, float* test_inputs, float* test_targets, int test_size, bool is_classification) {
    int input_dim = model->layer_sizes[0];
    int output_dim = model->layer_sizes[model->n_layers];

    for (int i = 0; i < output_dim; ++i)
        std::fill(model->confusion_matrix[i], model->confusion_matrix[i] + output_dim, 0);
    for (int i = 0; i < test_size; ++i) {
        float* xi = test_inputs + i * input_dim;
        float* ti = test_targets + i * output_dim;
        mlp_propagate(model, xi, is_classification);
        int pred = 0;
        float mv = model->X[model->n_layers][1];
        for (int j = 2; j <= output_dim; ++j)
            if (model->X[model->n_layers][j] > mv) {
                mv = model->X[model->n_layers][j];
                pred = j - 1;
            }
        int act = 0;
        for (int j = 0; j < output_dim; ++j)
            if (ti[j] == 1.0f) {
                act = j;
                break;
            }
        model->confusion_matrix[act][pred]++;
    }
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

DLLEXPORT void save_mlp_model(MLP* model, const char* filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return;

    out.write((char*)&model->n_layers, sizeof(int));
    out.write((char*)model->layer_sizes, (model->n_layers + 1) * sizeof(int));

    for (int l = 1; l <= model->n_layers; ++l) {
        int rows = model->layer_sizes[l - 1] + 1;
        int cols = model->layer_sizes[l] + 1;
        for (int i = 0; i < rows; ++i) {
            out.write((char*)model->W[l][i], cols * sizeof(float));
        }
    }

    out.close();
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
        for (int i = 0; i < rows; ++i) {
            in.read((char*)model->W[l][i], cols * sizeof(float));
        }
    }

    in.close();
    return model;
}
}
