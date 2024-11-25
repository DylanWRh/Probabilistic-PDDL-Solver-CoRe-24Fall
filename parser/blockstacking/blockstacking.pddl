(define (domain blockstacking)

    (:requirements :strips :negative-preconditions)

    (:predicates 
        (clear ?x)      ; nothing is on x
        (on ?x ?y)      ; x is on y
        (onTable ?x)    ; x is on the table
    )
    
    ; move x on y
    (:action move_to
        :parameters (?x ?y)
        :precondition (and 
            (clear ?x) (clear ?y) (onTable ?x))
        :effect (and 
            (not (clear ?y)) (on ?x ?y) (not (onTable ?x)))
    )

    ; move x on the table
    (:action move_toTable
        :parameters (?x ?y)
        :precondition (and (clear ?x) (on ?x ?y))
        :effect (and (onTable ?x) (not (on ?x ?y)) (clear ?y))
    )
)

