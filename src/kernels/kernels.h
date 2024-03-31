#pragma once

#include "Particle.h"

namespace kernels
{

void compute_forces(float *x_vec, float *y_vec, float *z_vec,
                    float *vx_vec, float *vy_vec, float *vz_vec,
                    float *ax_vec, float *ay_vec, float *az_vec,
                    float *mass_vec, float *type_vec,
                    int num_particles);

void euler_update(float *x_vec, float *y_vec, float *z_vec,
                  float *vx_vec, float *vy_vec, float *vz_vec,
                  float *ax_vec, float *ay_vec, float *az_vec,
                  float *mass_vec, float *type_vec,
                  int num_particles);
    
void get_min_max(const Particle *particles,
                 const int num_particles,
                 float *min_x, float *max_x,
                 float *min_y, float *max_y);
    
void draw_particles(float *x_vec, float *y_vec, float *z_vec,
                    float *vx_vec, float *vy_vec, float *vz_vec,
                    float *ax_vec, float *ay_vec, float *az_vec,
                    float *mass_vec, float *type_vec,
                    const int num_particles,
					const float min_x, const float max_x,
					const float min_y, const float max_y,
					unsigned *pixel_buf, const int width, const int height);

void mega_kernel(float *x_vec, float *y_vec, float *z_vec,
                 float *vx_vec, float *vy_vec, float *vz_vec,
                 float *ax_vec, float *ay_vec, float *az_vec,
                 float *mass_vec, float *type_vec,
                 const int num_particles,
				 const float min_x, const float max_x,
				 const float min_y, const float max_y,
				 unsigned *pixel_buf, const int width, const int height);
    
void init();
}
