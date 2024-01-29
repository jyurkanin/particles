#pragma once
#include "Particle.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>



__global__ void compute_forces(Particle *particles, int num_particles);
__global__ void euler_update(Particle *particles, int num_particles);


__global__ void get_min_x(const Particle *particles, const int num_particles, float &min);
__global__ void get_min_y(const Particle *particles, const int num_particles, float &min);
__global__ void get_max_x(const Particle *particles, const int num_particles, float &max);
__global__ void get_max_y(const Particle *particles, const int num_particles, float &max);


__global__ void draw_particles(const Particle *particles, const int num_particles,
                               const float min_x, const float min_y,
                               const float max_x, const float max_y,
                               unsigned int *pixelbuf, const int width, const int height);



