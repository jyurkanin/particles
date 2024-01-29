# Build
replace sm86 with your architecture.
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=86 ..

# What is the plan here
Particle Engine class? Sure
particle kernels for gpu wizardry and going fast

Hopefully we can keep all particle data on the gpu, and just copy
it onto the cpu after each iteration. idk.

