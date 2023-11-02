## Task description

find out what bugs are in with syncthreads.cu


## Result

The original file works as intended on first try with the following output:
```
Running computation...
Computation finished!
```

This indicates that verification is successful as well, which is unexpected given the task. 

I tried the suggested solution from the lecture slides:
```
    if(arr[idx] != 2) {
        local_array[threadIdx.x] = arr[idx] + 1;
    } else {
        local_array[threadIdx.x] = arr[idx];
    }
    __syncthreads();
```
This yields exactly the same output.