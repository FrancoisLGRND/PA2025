#include <cstdint>
#include <cstdlib>

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

float random_float(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}

float* define_bias(float* X, int dim){
    float* array = new float[dim+1];
    array[0] = 1.0;
    for(int i = 0; i < dim; i++){
        array[i+1] = X[0];
    }
    return array;
}

class LinearModel {
public:
    int dim;
    float* w;

    LinearModel(int dim) {
        this->dim = dim;
        w = new float[dim];
        for(int i = 0; i < dim; i++){
            w[i] = random_float(-1.0, 1.0);
        }
    }
};

extern "C" {
DLLEXPORT LinearModel* create_linear_model(int32_t dim) {
    return new LinearModel(dim);
}

DLLEXPORT float predict_linear_model(LinearModel* model, float* X) {

    float somme_pond = 0;
    for(int i = 0; i < model->dim; i++){
        somme_pond += model->w[i] * X[i];
    }
    if(somme_pond >= 0){
        return 1.0;
    }
    
    return -1.0;
}

DLLEXPORT void train_linear_model(LinearModel *model, float* X, float *Y, int32_t epochs, float learning_rate, int32_t batch_size) {
    for(int i = 0; i < epochs; i++){
        for (int j = 0; j < batch_size; j++){
            for(int k = 0; k < model->dim; k++){
                float *Xk = define_bias(X, model->dim);
                float gXk = predict_linear_model(model, Xk);
                model->w[k] += learning_rate * (Y[k] - gXk) * X[k];
                }
            }
        }
    }



DLLEXPORT void release_linear_model(LinearModel *model) {
    delete model;
}

/* DLLEXPORT float sum_array(const float* array, int32_t array_length) {
    float sum = 0.0;
    for (auto i = 0; i < array_length; i++) {
        sum += array[i];
    }
    return sum;
}

DLLEXPORT float* get_array_of_incrementing_numbers(int32_t num_elements) {
    auto array = new float[num_elements];
    for (auto i = 0; i < num_elements; i++) {
        array[i] = (float) i;
    }
    return array;
}

DLLEXPORT void delete_array(const float* array, int32_t array_length) {
    delete [] array;
}

DLLEXPORT int32_t my_add(int32_t a, int32_t b) {
    return a + b;
}
}*/
} 