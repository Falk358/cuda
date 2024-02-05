# Measurements for shared memory Matmul


## Results on V100

```
Time: regular: 0.500188 shared memory: 0.548289
```

Interestingly enough, the shared memory Implementation on a V100 GPU is slower than a regular approach.

## Results on A100


```
Time: regular: 0.431562  shared memory: 0.389876
```

Unsuprisingly, the newer A100 GPU is faster on both calculations. However, the shared memory implementation on the A100 also performs better than the regular implementation on the A100.


## Discussion

We can only speculate why the V100 GPU performs worse when using shared memory than without. It is possible that the A100 features better optimizations when it comes to shared memory syncronization. The problem size could be too small for the speedup of using shared memory to take proper effect, and the syncronization overhead of the V100 could be bigger than the actual computation itself.