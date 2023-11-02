## Task


execute `cos.cu` with parameter taken from `cos.txt`, which calculates the *cosine* with the cpu and gpu implementation.



## Results

we get the following output:
```
GPU: cos(5992555) = -0.382683306932449
CPU: cos(5992555) = 3.32090451138356e-07

```

As we can see, the results diverge significantly. According to chapter 13.1 from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions:

>Function      Maximum ulp Error
>cosf(x)       2 (full range)

Therefore, our diverging result makes sense. This inaccuracy needs to be kept in mind for applications where precision is critical.