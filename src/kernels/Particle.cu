#include "Particle.cuh"


__device__ float force_func(const float dist)
{
    return 0.1*dist;
}

__global__ void euler_update(Particle *particles, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const float timestep = 1e-3;
    for(int i = idx; i < num_particles; i += stride)
    {
        particles[i].m_x += particles[i].m_vx*timestep;
        particles[i].m_y += particles[i].m_vy*timestep;
        particles[i].m_z += particles[i].m_vz*timestep;
    }
}


__global__  void compute_forces(Particle *particles, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = idx; i < num_particles; i += stride)
    {
        float fx = 0;
        float fy = 0;
        float fz = 0;
        
        for(int j = 0; j < num_particles; j++)
        {
            float dx = particles[i].m_x - particles[j].m_x;
            float dy = particles[i].m_y - particles[j].m_y;
            float dz = particles[i].m_z - particles[j].m_z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            float force = force_func(dist);
            
            fx += (dx*force) / dist;
            fy += (dy*force) / dist;
            fz += (dz*force) / dist;
        }
        
        particles[i].m_vx = fx / particles[i].m_mass;
        particles[i].m_vy = fy / particles[i].m_mass;
        particles[i].m_vz = fz / particles[i].m_mass;
    }    
}

// TODO: Optimize this reduction kernel https://cuvilib.com/Reduction.pdf
__global__ void get_min_x(const Particle *particles, const int num_particles, float &min)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num_particles; i += stride)
    {
        min = fminf(min, particles[i].m_x);
    }
}

__global__ void get_min_y(const Particle *particles, const int num_particles, float &min)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num_particles; i += stride)
    {
        min = fminf(min, particles[i].m_y);
    }
}

__global__ void get_max_x(const Particle *particles, const int num_particles, float &max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num_particles; i += stride)
    {
        max = fmaxf(max, particles[i].m_x);
    }
}

__global__ void get_max_y(const Particle *particles, const int num_particles, float &max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num_particles; i += stride)
    {
        max = fmaxf(max, particles[i].m_y);
    }
}

// TODO: Anti-Aliasing
__global__ void draw_particles(const Particle *particles, const int num_particles,
                               const float min_x, const float min_y,
                               const float max_x, const float max_y,
                               unsigned int *pixelbuf, const int width, const int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = idx; i < num_particles; i += stride)
    {
        int x = __float2int_rd(width*(particles[i].m_x - min_x)/ (max_x - min_x));
        int y = __float2int_rd(height*(particles[i].m_y - min_y)/ (max_y - min_y));
        
        pixelbuf[y*width + x] = 0xFF; // Blue
    }
}
