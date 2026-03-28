#lang racket/base

(require "simple-dynet-api.rkt")
(provide (all-from-out "simple-dynet-api.rkt"))
(require "rnn.rkt")
(provide (all-from-out "rnn.rkt"))
(require "mem-utils.rkt")
(provide (all-from-out "mem-utils.rkt"))
(require "lib-cudann.rkt")
(provide (all-from-out "lib-cudann.rkt"))


(require "ffi-utils.rkt")
(provide (all-from-out "ffi-utils.rkt"))
(require "ffi-functional.rkt")
(provide (all-from-out "ffi-functional.rkt"))
(require "cudnn-tensor.rkt")
(provide (all-from-out "cudnn-tensor.rkt"))

(require "cudnn-dropout.rkt")
(provide (all-from-out "cudnn-dropout.rkt"))
(require "cudnn-api.rkt")
(provide (all-from-out "cudnn-api.rkt"))
(require "cuda-api.rkt")
(provide (all-from-out "cuda-api.rkt"))

(module+ test
  (require rackunit))

;; Notice
;; To install (from within the package directory):
;;   $ raco pkg install
;; To install (once uploaded to pkgs.racket-lang.org):
;;   $ raco pkg install <<name>>
;; To uninstall:
;;   $ raco pkg remove <<name>>
;; To view documentation:
;;   $ raco docs <<name>>
;;
;; For your convenience, we have included LICENSE-MIT and LICENSE-APACHE files.
;; If you would prefer to use a different license, replace those files with the
;; desired license.
;;
;; Some users like to add a `private/` directory, place auxiliary files there,
;; and require them in `main.rkt`.
;;
;; See the current version of the racket style guide here:
;; http://docs.racket-lang.org/style/index.html

;; Code here



(module+ test
  ;; Any code in this `test` submodule runs when this file is run using DrRacket
  ;; or with `raco test`. The code here does not run when this file is
  ;; required by another module.

  ;(check-equal? (+ 2 2) 4)
  )

(module+ main
  ;; (Optional) main submodule. Put code here if you need it to be executed when
  ;; this file is run using DrRacket or the `racket` executable.  The code here
  ;; does not run when this file is required by another module. Documentation:
  ;; http://docs.racket-lang.org/guide/Module_Syntax.html#%28part._main-and-test%29

;  (require racket/cmdline)
;  (define who (box "world"))
;  (command-line
;    #:program "my-program"
;    #:once-each
;    [("-n" "--name") name "Who to say hello to" (set-box! who name)]
;    #:args ()
;    (printf "hello ~a~n" (unbox who)))

  )
