import ctypes
import numpy as np

def run():
    # C++ lib
    lib = ctypes.cdll.LoadLibrary("../_2025_3A_IABD1_DemoInteropCpp/cmake-build-debug/_2025_3A_IABD1_DemoInteropCpp.dll")

    # # Rust lib
    # lib = ctypes.cdll.LoadLibrary(
    #     "../_2025_3A_IABD1_DemoInteropRust/target/debug/_2025_3A_IABD1_DemoInteropRust.dll")

    lib.create_linear_model.argtypes = [ctypes.c_int32]
    lib.create_linear_model.restype = ctypes.c_void_p

    lib.predict_linear_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.predict_linear_model.restype = ctypes.c_float

    lib.release_linear_model.argtypes = [ctypes.c_void_p]
    lib.release_linear_model.restype = None

    lib.train_linear_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_float, ctypes.c_int32]
    lib.train_linear_model.restype = None

    model = lib.create_linear_model(42.0, 51.0)
    print(lib.predict_linear_model(model))
    lib.release_linear_model(model)

    float_array = np.array([66.0, 44.0], dtype=np.float32)
    float_array_pointer = np.ctypeslib.as_ctypes(float_array)
    print(lib.sum_array(float_array_pointer, len(float_array)))

    array_length = 10
    array_pointer = lib.get_array_of_incrementing_numbers(array_length)
    array = np.ctypeslib.as_array(array_pointer, (array_length,))
    print(array)
    lib.delete_array(array_pointer, array_length)

    print(lib.my_add(33, 22))


if __name__ == "__main__":
    run()