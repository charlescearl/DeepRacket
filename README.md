# DeepRacket

This package provides a set of interfaces for doing deep learning in the [Racket](https://racket-lang.org/) (a Scheme/Lisp dialect) programming language. 

The project is still in the growing pains phase, so please excuse the mess. 

The code here is split into two parts. The first and most useful for now is a very preliminary interface to the [Dynet](https://github.com/clab/dynet) neural 
network library. Dynet seems to have many of the features that you would expect in a lisp like language, foremost is the dynamic specification of neural 
networks. A few simple cases are included. 

The second approach is a low level interface to the  NVIDIA [cudnn](https://developer.nvidia.com/cudnn) deep learning library. While this gives a lot of flexibility,
there is a lot more to do here. The next big hurdle is including loss calculaton and weight updates. The code now provides the ability to create cudnn objects (e.g RNNs) 
and perform simple forward calculations on models.

The approach I've taken is to use the [Torch](https://github.com/soumith/cudnn.torch/blob/master/ffi.lua) Deep Learning library wrapper as a guide.

Suggestions welcome.
