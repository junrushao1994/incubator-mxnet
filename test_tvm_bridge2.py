import mxnet as mx

a = mx.nd.array([1, 2, 3, 4, 5], ctx=mx.gpu(0))
b = mx.nd.array([5, 4, 3, 2, 1], ctx=mx.gpu(0))
c = mx.nd.tvm_vector_add(a, b)
print(a.asnumpy())
print(b.asnumpy())
print(c.asnumpy())
c.wait_to_read()
print("Program shutdown!!!")
