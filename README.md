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

| Version | Run Time |
|---------+----------|
| 1       | 4.23772  |
| 2       | |
