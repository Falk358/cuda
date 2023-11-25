========= CUDA-MEMCHECK
Running computation...
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (63,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (62,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (61,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (60,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (59,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (58,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (57,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (56,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (55,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (54,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (53,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (52,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (51,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (50,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (49,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (48,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (47,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (46,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (45,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (44,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (43,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (42,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (41,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (40,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (39,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (38,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (37,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (36,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (35,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (34,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (33,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x000008f0 in /home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*)
=========     by thread (32,0,0) in block (0,0,0)
=========     Device Frame:/home/training2/cuda/exercise_4_2/syncthreads.cu:20:globFunction(int*) (globFunction(int*) : 0x8f0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:../build/syncthreads [0x836b]
=========     Host Frame:../build/syncthreads [0x54e20]
=========     Host Frame:../build/syncthreads [0x3e8e]
=========     Host Frame:../build/syncthreads [0x3d44]
=========     Host Frame:../build/syncthreads [0x3d6c]
=========     Host Frame:../build/syncthreads [0x3b65]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x223d5]
=========     Host Frame:../build/syncthreads [0x397e]
=========
Computation finished!
========= ERROR SUMMARY: 32 errors








the issue here is the following code snippet from the kernel:
```
 // read values and increase if not already 2
    if(arr[idx] != 2) {
        local_array[threadIdx.x] = arr[idx] + 1;
        __syncthreads();
    } else {
        local_array[threadIdx.x] = arr[idx];
        __syncthreads();
    }
```
As we heard already in a previous lecture, calling `__syncthreads()` within an `if` statement can cause problems. The issue arises if the statement within the `if` conditional doesn't evaluate to the same value for each thread within the same block. This causes undefined behaviour. We can fix this by moving the `__syncthreads()` call after the `if` block:
```
 // read values and increase if not already 2
    if(arr[idx] != 2) {
        local_array[threadIdx.x] = arr[idx] + 1;
    } else {
        local_array[threadIdx.x] = arr[idx];
    }
    __syncthreads();
```