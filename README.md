# UnionFindDecoder

This project provide unionfind decoder for rotated surface code based qecsim package.

## Usage
```bash
pip install qecsim
python playground.py
```
The programe `CheckDecoderTime.py` can draw this decoder's decode time with the change of the number of qubits, which shows almost linear time. 

The programe `Calthreshold.py` can draw this decoder's logical error rate with the change of physical error rate, which shows the threshold.
## Extension
The code about decoder is based on decoder graph, so it can decode arbitray graph/surface code. If you want to extend it to decode other suface code e.g. toric code, 
just add something to `DecoderGraph` and `CalMidPos` (it is nontrival for rotated surface code since there are boundary vertex).
