#include <cstdint>

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

class LinearModel {
public:
    float a;
    float b;

    LinearModel(float a, float b) {
        this->a = a;
        this->b = b;
    }
};

extern "C" {
DLLEXPORT LinearModel *create_linear_model(float a, float b) {
    return new LinearModel(a, b);
}

DLLEXPORT float predict_linear_model(LinearModel *model) {
    return model->a + model->b;
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