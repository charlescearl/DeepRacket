#lang typed/racket
(require typed/racket/unsafe)

(unsafe-require/typed
 ffi/unsafe
 [#:opaque CPointer cpointer?]
 [#:opaque CType ctype?])

(require/typed
  ffi/unsafe
  [_double CType]
  [_float CType]
  [_uintptr CType]
  [_pointer CType]
  [_int CType]
  [_size CType]
  [flvector->cpointer (FlVector -> CPointer)]
  [ptr-ref (case->
	    [CPointer CType -> Any]
	    [CPointer CType Exact-Nonnegative-Integer -> Any])]
  [ptr-set! (case->
	     [CPointer CType  Any -> Void]
	     [CPointer CType Exact-Nonnegative-Integer Any -> Void])]
  [ptr-add (CPointer Exact-Nonnegative-Integer CType -> CPointer)]
  [ctype-sizeof (CType -> Exact-Nonnegative-Integer)]
  [malloc (case->
           [Nonnegative-Integer -> CPointer]
           [Symbol Nonnegative-Integer -> CPointer])]
 
  )

(unsafe-require/typed
 ffi/unsafe/cvector
 [#:opaque CVector cvector?]
 )

(require/typed
 ffi/unsafe/cvector
 [cvector (CType Any -> CVector)])

(provide
 CPointer
 CType
 _double
 _float
 _uintptr
 _pointer
 _int
 _size 
 flvector->cpointer 
 ptr-ref 
 ptr-set!
 ptr-add
 ctype-sizeof 
 malloc)
 
