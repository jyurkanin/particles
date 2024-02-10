#include "kernels1.h"
#include "Parameters.h"

namespace kernels1
{

__global__ void cuda_euler_update(Particle *particles, int num_particles)
{

}

__global__  void cuda_compute_forces(Particle *particles, int num_particles)
{
    
}

// TODO: Optimize this reduction kernel https://cuvilib.com/Reduction.pdf
__global__ void cuda_get_min_max(const Particle *particles,
                                 const int num_particles,
                                 float *min_x, float *max_x,
                                 float *min_y, float *max_y)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = idx; i < num_particles; i += stride)
    {
        *min_x = fminf(*min_x, particles[i].m_x);
        *max_x = fminf(*max_x, particles[i].m_x);
        *min_y = fminf(*min_y, particles[i].m_y);
        *max_y = fminf(*max_y, particles[i].m_y);
    }
}

// TODO: Anti-Aliasing
__global__ void cuda_draw_particles(const Particle *particles, const int num_particles,
                                    const float min_x, const float min_y,
                                    const float max_x, const float max_y,
                                    unsigned int *pixelbuf, const int width, const int height)
{
    
}







void compute_forces(Particle *particles, int num_particles) {}
void euler_update(Particle *particles, int num_particles) {}
void get_min_max(const Particle *particles,
                 const int num_particles,
                 float *min_x, float *max_x,
                 float *min_y, float *max_y)
{
    cuda_get_min_max<<<Parameters::num_blocks, Parameters::blocksize>>>(particles, num_particles,
                                                                        min_x, max_x,
                                                                        min_y, max_y);
}

void draw_particles(const Particle *particles, const int num_particles,
                    const float min_x, const float max_x,
                    const float min_y, const float max_y,
                    unsigned int *pixelbuf, const int width, const int height){}






    
}
