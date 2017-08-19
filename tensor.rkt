;;Typed tensors
#lang typed/racket
(require/typed
 "lib-cudann.rkt"
 ffi/unsafe
 math/base
 math/matrix)

(struct tensor ([in-vect : Float] [gpu-vect : Pointer]))
