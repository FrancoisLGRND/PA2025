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
    int num_class;
    float* w;

    LinearModel(int dim, int num_class) {
        this->dim = dim;
        w = new float[num_class];
        for(int i = 0; i < dim; i++){
            w[i] = random_float(-1.0, 1.0);
        }
    }
};

extern "C" {
DLLEXPORT LinearModel* create_linear_model(int32_t dim, int32_t num_class) {
    return new LinearModel(dim, num_class);
}

DLLEXPORT int32_t predict_binary_linear_model(LinearModel* model, float* X) { 

    int somme_pond = 0;
    for(int i = 0; i < model->dim; i++){
        somme_pond += model->w[i] * X[i];
    }
    if(somme_pond >= 0){
        return 1.0;
    }
    
    return -1.0;
}

DLLEXPORT float train_binary_linear_model(LinearModel *model, float** X, int32_t *Y, int32_t epochs, float learning_rate) {
    for(int i = 0; i < epochs; i++){
        for (int j = 0; j < sizeof(X) / sizeof(X[0]); i++){
            for(int k = 0; k < model->dim; i++){
                float *Xk = define_bias(X[k], model->dim);
                float predicted = predict_binary_linear_model(model, X[j]);
                int actual = Y[j];
                for(int k = 0; k < model->dim; k++){
                    if(predicted != actual){
                        model->w[k] += learning_rate; 
                    }
                }
            }
        }
    }
}

DLLEXPORT void release_linear_model(LinearModel *model) {
    delete model;
}

DLLEXPORT float sum_array(const float* array, int32_t array_length) {
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
}
}