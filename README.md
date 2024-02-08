# Build
replace sm86 with your architecture.
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=86 ..
compdb -p build/ list > compile_commands.json

# What is the plan here
I just want to simulate some particles as fast as I possibly can.
I want to simulate like 1 million and see how many fps I can get.
I will progressively optimize the cuda kernels and keep a record
of the performance improvements and save the files as I progress.

