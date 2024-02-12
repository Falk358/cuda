# Power iteration Computation using Julia

The task was to compute the *Eigenvector* associated to the dominant *Eigenvalue* of a Matrix, using the **Julia** programming Language and **CUDA**. We were tasked with implementing the *Power Iteration* method.

Unfortunately, it is not guaranteed that *Power Iteration* converges to a correct result, which happened with our matrix choice. Our implementation should be correct nonetheless, since we checked the result of our cuda norm computation, which should coincide with the highest eigenvalue of our Matrix:
```
 Maximal eigenvalue of matrix computed by cuda norm is: [36.209372712298546]
```

Compared this to the eigenvalue list computed using Julias `LinearAlgebra` package with the `eigen()` method:
```
The eigenvalues of our given matrix computed with LinearAlgebra package are:
-2.2093727122985456
-4.707175062775106e-16
9.647496950486919e-16
36.20937271229853
```
We can see that the last eigenvalue is the largest of all of them and is approximated by our own cuda norm.