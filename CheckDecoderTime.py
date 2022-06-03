from math import sqrt
from qecsim import paulitools as pt
from UnionFindDecoder import UnionFindDecoder
from qecsim import app
import numpy as np
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode
import matplotlib.pyplot as plt
import time
import random
# sys.setrecursionlimit(3000)
# my_code = RotatedPlanarCode(7, 7)
my_error_model = DepolarizingErrorModel()
my_decoder = UnionFindDecoder()
error_probability = 0.1
result = []
errorList = [0.1,0.05,0.03,0.02,0.01]
# 算时间
maxsize = 70
x = [i*i for i in range(7, maxsize, 4)]
for err in errorList:
    result = []
    error_probability = err
    for i in range(7, maxsize, 4):
        all = 0
        my_code = RotatedPlanarCode(i, i)
        for j in range(10000):
            rng = np.random.default_rng(random.randint(1,100000))
            error = my_error_model.generate(my_code, error_probability, rng)
            syndrome = pt.bsp(error, my_code.stabilizers.T)
            begin = time.perf_counter()
            recovery = my_decoder.decode(my_code, syndrome)
            end = time.perf_counter()
            all += (end-begin)
        # print(sqrt(all),all)
        result.append(all)
    plt.plot(x, result,label='p={}'.format(err))
# print(result)
# 画图


plt.title('Average running time')
plt.xlabel('Number of qubits')
plt.ylabel('Time to decode 10000 samples(s)')
plt.legend()
# plt.show()
plt.savefig('test.png')
error01 = [6.72487099999999, 14.44702159999997, 28.70565619999993, 47.09786509999944, 67.66742739999876, 93.39242120000429, 125.90741399999445, 162.2740033999953,
    203.56844550000483, 250.20119479999153, 302.77094569998235, 360.29361200000403, 421.42012969996995, 489.30232379992503, 560.2247984000678, 648.1841177000606]
