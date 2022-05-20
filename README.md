# UnionFindDecoder

This project provide unionfind decoder for rotated surface code based qecsim package.

## Usage
```bash
pip install qecsim
python playground.py
```

## Extension
The code about decoder is based on decoder graph, so it can decode arbitray graph/surface code. If you want to extend it to decode other suface code e.g. toric code, 
just add something to `DecoderGraph` and `CalMidPos` (it is nontrival for rotated surface code since there are boundary vertex).
