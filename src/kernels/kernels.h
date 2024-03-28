#pragma once

#include "Particle.h"

namespace kernels
{
    
void compute_forces(Particle *particles, int num_particles);
void euler_update(Particle *particles, int num_particles);
void get_min_max(const Particle *particles,
                 const int num_particles,
                 float *min_x, float *max_x,
                 float *min_y, float *max_y);
    
void draw_particles(const Particle *particles, const int num_particles,
                    const float min_x, const float max_x,
                    const float min_y, const float max_y,
                    unsigned int *pixelbuf, const int width, const int height);

void mega_kernel(Particle *particles, const int num_particles,
                 const float min_x, const float max_x,
                 const float min_y, const float max_y,
                 unsigned int *pixelbuf, const int width, const int height);

}
