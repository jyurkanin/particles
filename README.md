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
