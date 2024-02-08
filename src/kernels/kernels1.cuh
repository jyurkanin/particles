#pragma once

#include "Particle.h"

namespace kernels1
{
__global__ void compute_forces(Particle *particles, int num_particles);
__global__ void euler_update(Particle *particles, int num_particles);

__global__ void get_min_max(const Particle *particles,
                            const int num_particles,
                            float *min_x,
                            float *max_x,
                            float *min_y,
                            float *max_y);

__global__ void draw_particles(const Particle *particles, const int num_particles,
                               const float min_x, const float min_y,
                               const float max_x, const float max_y,
                               unsigned int *pixelbuf, const int width, const int height);



}
