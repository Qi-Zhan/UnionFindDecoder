from math import sqrt
from qecsim import paulitools as pt
from UnionFindDecoder import UnionFindDecoder
from qecsim import app
import numpy as np
from qecsim.models.generic import DepolarizingErrorModel, PhaseFlipErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode
import matplotlib.pyplot as plt
my_error_model = DepolarizingErrorModel()
my_decoder = UnionFindDecoder()
error_probability = 0.1
result = []
errorList = [0.14, 0.12, 0.10, 0.08,0.05, 0.01]
# 算时间
maxsize = 21
x = [i*i for i in range(7, maxsize, 2)]
for err in errorList:
    result = []
    error_probability = err
    for i in range(7, maxsize, 2):
        my_code = RotatedPlanarCode(i, i)
        res = app.run(my_code, my_error_model, my_decoder,
                       error_probability, max_runs=10000)
        result.append(100*(res['logical_failure_rate']))
    plt.plot(x, result, label='p={}'.format(err))


plt.title('Performance of the rotated surface code for Z type error')
plt.xlabel('Number of qubits')
plt.ylabel('Logical error rate(%)')
plt.legend()
# plt.show()
plt.savefig('test.png')
