

def flatten(x):
    dim = x.shape
    arg2 = 1
    for i in range(len(dim) - 1):
        arg2 = dim[i+1] * arg2
    return x.reshape((dim[0], arg2))

def normalize(x):
    x = x.astype('float32')
    return x/255

