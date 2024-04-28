# Build
replace sm86 with your architecture.
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=86 ..
compdb -p build/ list > compile_commands.json

# What is the plan here
I just want to simulate some particles as fast as I possibly can.
I want to simulate like 1 million and see how many fps I can get.
I will progressively optimize the cuda kernels and keep a record
of the performance improvements and save the files as I progress.

# draw_particles test doesn't work. Oh well.
Seems to be some kind of unknowable floating point error.
And I don't care enough to figure out the floating point differences
between cpu and gpu. So I'm going to move on and accept the error.

# Ideas for going even more faster
Tile it. To fit in the shared memory. Create a new kernel.
So a warp is 32 threads.
Lets say a block is 64 threads.
Grid will be a 2d array of blocks.

The first 64 particle positions will be stored in shared memory.
Then each block will compute the interactions between the
first 0-63 particles.
The result will be the incomplete forces on the first 64 particles.
Results will be reduced along the gridDim.x direction to get the complete forces
acting on the first 64 particles.
Results will for the next 64 particles will be computed by moving along the gridDim.y
direction.

# Experiment with a grid level sync?
GPU is not meant to do this, the internet says this will be shit.
I still want to try though.


# Benchmark
100 calls to runIteration()

| Version | Run Time | Improvement |
|---------+----------+-------------|
| 1       | 40.2377  | garbo |
| 2       | 2.05122  | Combined kernels into 1 big one. Removed cudaDeviceSynchronize. |
| 3       | 1.50391  | Vectorized the shit into struct of float*  instead of Particle* |
| 3.1     | 1.38789  | Removed Particle* from the engine entirely. |
| 3.2     | 1.37076  | Malloc everything as one big ass block. |
| 3.3     | 1.26551  | Added restrict to a bunch of the arguments. |
| 3.4     | 0.737728 | Computed inverse of dist and stored it. Replaced divides with multiplies. |
| 3.5     | 0.574305 | Get rid of sqrt. |
| 3.6     | 0.389918 | Added optimization flags. |
| 6       | 0.322963  | Tiled the computation. Lol! |

