#include "kernels1.cuh"

namespace kernels1
{

__device__ float force_func(const float dist)
{
    return 0.1*dist;
}

__global__ void euler_update(Particle *particles, int num_particles)
{

}

__global__  void compute_forces(Particle *particles, int num_particles)
{
    
}

// TODO: Optimize this reduction kernel https://cuvilib.com/Reduction.pdf
__global__ void get_min_max(const Particle *particles,
                            const int num_particles,
                            float *min_x,
                            float *max_x,
                            float *min_y,
                            float *max_y)
{
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
__global__ void draw_particles(const Particle *particles, const int num_particles,
                               const float min_x, const float min_y,
                               const float max_x, const float max_y,
                               unsigned int *pixelbuf, const int width, const int height)
{
	    
}

}
