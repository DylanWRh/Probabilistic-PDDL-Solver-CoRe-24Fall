(define (problem problem1)
	(:domain blockstacking)
	(:objects A B)
	(:init 
		(clear A)
		(clear B)
		(onTable A)
		(onTable B)
	)
	(:goal (and 
		(on A B)
		(onTable B)
	))
)
