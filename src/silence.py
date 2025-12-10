from llama_cpp import llama_log_set
import ctypes

# Per silenziare il logs di llama_cpp
log_callback = ctypes.CFUNCTYPE(
    None, 
    ctypes.c_int, 
    ctypes.c_char_p, 
    ctypes.c_void_p
)(lambda l, m, u: None)

llama_log_set(log_callback, ctypes.c_void_p())
