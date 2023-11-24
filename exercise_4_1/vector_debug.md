========= CUDA-MEMCHECK
========= Internal error
========= No CUDA-MEMCHECK results found
[training2@mp-gpu3-login exercise_4_1]$ ~/
.cache/          cuda/            .nv/             slurm_bash.sh    storage/         vscode-cpptools/ 
.config/         .local/          .pki/            .ssh/            .vim/            .vscode-server/  
[training2@mp-gpu3-login exercise_4_1]$ ~/slurm_bash.sh 
[training2@mp-gpu3-a100 exercise_4_1]$ cuda-gdb ../build/vector
NVIDIA (R) CUDA Debugger
11.2 release
Portions Copyright (C) 2007-2020 NVIDIA Corporation
GNU gdb (GDB) 8.3.1
Copyright (C) 2019 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ../build/vector...
(cuda-gdb) cuda-memcheck
Undefined command: "cuda-memcheck".  Try "help".
(cuda-gdb) exit
Undefined command: "exit".  Try "help".
(cuda-gdb) quit
[training2@mp-gpu3-a100 exercise_4_1]$ cuda-memcheck ../build/vector
========= CUDA-MEMCHECK
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (264,0,0) in block (0,0,0)
=========     Address 0x7f350c000420 is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (263,0,0) in block (0,0,0)
=========     Address 0x7f350c00041c is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (262,0,0) in block (0,0,0)
=========     Address 0x7f350c000418 is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (261,0,0) in block (0,0,0)
=========     Address 0x7f350c000414 is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (260,0,0) in block (0,0,0)
=========     Address 0x7f350c000410 is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (259,0,0) in block (0,0,0)
=========     Address 0x7f350c00040c is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (258,0,0) in block (0,0,0)
=========     Address 0x7f350c000408 is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (257,0,0) in block (0,0,0)
=========     Address 0x7f350c000404 is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Invalid __global__ write of size 4
=========     at 0x00000250 in /home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*)
=========     by thread (256,0,0) in block (0,0,0)
=========     Address 0x7f350c000400 is out of bounds
=========     Device Frame:/home/training2/cuda/exercise_4_1/vector.cu:9:KrnlDmmy(int*) (KrnlDmmy(int*) : 0x250)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/vector [0x829b]
=========     Host Frame:../build/vector [0x54d50]
=========     Host Frame:../build/vector [0x3dc7]
=========     Host Frame:../build/vector [0x3c7d]
=========     Host Frame:../build/vector [0x3ca5]
=========     Host Frame:../build/vector [0x3ae4]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/lib64/libcuda.so.1 [0x37a373]
=========     Host Frame:../build/vector [0x359b7]
=========     Host Frame:../build/vector [0x3ae9]
*** error synchronizing device, 1 ***
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaFree.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/lib64/libcuda.so.1 [0x37a373]
=========     Host Frame:../build/vector [0x3ebe5]
=========     Host Frame:../build/vector [0x3b35]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/vector [0x392e]
=========
========= ERROR SUMMARY: 11 errors