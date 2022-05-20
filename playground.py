import numpy as np
from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode
from UnionFindDecoder import UnionFindDecoder

my_code = RotatedPlanarCode(7, 7)
my_error_model = DepolarizingErrorModel()
# set physical error probability to 10%
error_probability = 0.1
rng = np.random.default_rng(14)
# error: random error based on error probability
error = my_error_model.generate(my_code, error_probability, rng)
print('error:\n{}'.format(my_code.new_pauli(error)))
syndrome = pt.bsp(error, my_code.stabilizers.T)
print('syndrome:\n{}'.format(my_code.ascii_art(syndrome)))
test = UnionFindDecoder()
recovery = test.decode(my_code, syndrome)
print('recovery:\n{}'.format(my_code.new_pauli(recovery)))
print('recovery ^ error:\n{}'.format(my_code.new_pauli(recovery ^ error)))
if pt.bsp(recovery ^ error, my_code.logicals.T)[0] == 0 and pt.bsp(recovery ^ error, my_code.logicals.T)[1] == 0:
    print('decode success')
else:
    print('decode fail')
