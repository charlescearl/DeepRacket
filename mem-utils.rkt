;;Typed tensors
#lang typed/racket
(require "ffi-functional.rkt")
(require
 math/array)
;; Return pointer for start of an array
;;(array->cptr (array #[#[1.0 2.24554] #[3.2323 4.001] #[5.454 5.86868]]))
(: array->cptr (-> (Array Float) CPointer))
(define (array->cptr arr)
  (flvector->cpointer (flarray-data (array->flarray arr))))

;; Utility for getting a referenced value from an array
(: ptr-array-ref (-> CPointer Exact-Nonnegative-Integer Flonum))
(define (ptr-array-ref arr-ptr idx)
  (cast (ptr-ref arr-ptr _double idx) Flonum))

;; Utility for getting a referenced value from an array
(: ptr-array-set! (-> CPointer Exact-Nonnegative-Integer Flonum Void))
(define (ptr-array-set! arr-ptr idx val)
  (ptr-set! arr-ptr _double idx val))


(: FLOAT-SIZE Exact-Nonnegative-Integer)
(define FLOAT-SIZE (ctype-sizeof _float))

(: DOUBLE-SIZE Exact-Nonnegative-Integer)
(define DOUBLE-SIZE (ctype-sizeof _double))

(: INT-SIZE Exact-Nonnegative-Integer)
(define INT-SIZE (ctype-sizeof _int))

(: get-int-array-block ((Listof Nonnegative-Integer) -> CPointer))
(define (get-int-array-block shape )
  (malloc 'atomic-interior (* INT-SIZE (foldl * 1 shape))))

(: get-float-array-block ((Listof Nonnegative-Integer) -> CPointer))
(define (get-float-array-block shape )
  (malloc 'atomic-interior (* FLOAT-SIZE (foldl * 1 shape))))

(: get-double-array-block ((Listof Nonnegative-Integer) -> CPointer))
(define (get-double-array-block shape )
  (malloc 'atomic-interior (* DOUBLE-SIZE (foldl * 1 shape))))

(: copy-float-to-block (CPointer (Listof Float) -> Void))
(define (copy-float-to-block block arr)
  (for ([el : Float arr]
	[idx : Nonnegative-Integer (in-range (length arr))])
    (ptr-set! block _float idx el)))

(: copy-double-to-block (CPointer (Listof Flonum) -> Void))
(define (copy-double-to-block block arr)
  (for ([el : Flonum arr]
	[idx : Nonnegative-Integer (in-range (length arr))])
    (ptr-set! block _double idx el)))

(: copy-int-to-block (CPointer (Listof Integer) -> Void))
(define (copy-int-to-block block arr)
  (for ([el : Integer arr]
	[idx : Nonnegative-Integer (in-range (length arr))])
    (ptr-set! block _int idx el)))

(: print-float-block (CPointer Nonnegative-Integer -> Void))
(define (print-float-block block size)
  (for ([idx : Nonnegative-Integer (in-range size)])
    (print (ptr-ref block _float idx))))

(: print-double-block (CPointer Nonnegative-Integer -> Void))
(define (print-double-block block size)
  (for ([idx : Nonnegative-Integer (in-range size)])
    (display (format "~a, " (ptr-ref block _double idx))))
  (display (format "\n")))



(provide
 FLOAT-SIZE
 DOUBLE-SIZE
 INT-SIZE
 ptr-array-set!
 ptr-array-ref
 ptr-add
 array->cptr
 get-double-array-block
 get-float-array-block
 get-int-array-block
 copy-float-to-block
 copy-double-to-block
 copy-int-to-block
 print-double-block
 print-float-block
  )
