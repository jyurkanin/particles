#include "kernels.h"
#include "Parameters.h"

#include <cassert>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace kernels
{


__global__ void cuda_mega_kernel(const Particle* __restrict__ in_particles,
                                 Particle* __restrict__ out_particles,
                                 const int num_particles,
                                 const float min_x, const float max_x,
                                 const float min_y, const float max_y,
                                 unsigned *pixel_buf, const int width, const int height)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned num_threads = gridDim.x * blockDim.x;
    
    const float scalar = 1e-4;
    
    // compute accelerations
    for(unsigned ii = idx; ii < num_particles; ii += num_threads)
    {
        float damp = 1e-2;
        float fx = -damp*in_particles[ii].m_vx;
        float fy = -damp*in_particles[ii].m_vy;
        float fz = -damp*in_particles[ii].m_vz;
        
        for(unsigned jj = 0; jj < num_particles; jj++)
        {
            if(ii == jj) continue; //pls no divergent branch.
            
            float dx = in_particles[jj].m_x - in_particles[ii].m_x;
            float dy = in_particles[jj].m_y - in_particles[ii].m_y;
            float dz = in_particles[jj].m_z - in_particles[ii].m_z;
            float dist = fmaxf(1e-6, sqrtf(dx*dx + dy*dy + dz*dz));
            float force = in_particles[jj].m_mass * in_particles[ii].m_mass / dist;
                
            fx += (dx * force) / dist;
            fy += (dy * force) / dist;
            fz += (dz * force) / dist;
        }
            
        out_particles[ii].m_ax = fx / in_particles[ii].m_mass;
        out_particles[ii].m_ay = fy / in_particles[ii].m_mass;
        out_particles[ii].m_az = fz / in_particles[ii].m_mass;
        
        // Euler update
        out_particles[ii].m_vx = in_particles[ii].m_vx + (out_particles[ii].m_ax*scalar);
        out_particles[ii].m_vy = in_particles[ii].m_vy + (out_particles[ii].m_ay*scalar);
        out_particles[ii].m_vz = in_particles[ii].m_vz + (out_particles[ii].m_az*scalar);

        out_particles[ii].m_x = in_particles[ii].m_x + (out_particles[ii].m_vx*scalar);
        out_particles[ii].m_y = in_particles[ii].m_y + (out_particles[ii].m_vy*scalar);
        out_particles[ii].m_z = in_particles[ii].m_z + (out_particles[ii].m_vz*scalar);
            
        // Draw
        int x = (int)(width * (out_particles[ii].m_x - min_x) / (max_x - min_x));
        int y = (int)(height * (out_particles[ii].m_y - min_y) / (max_y - min_y));
        
        if((x >= 0) && (x < width) && (y >= 0) && (y < height))
        {
            float z_clamp_blue = fmaxf(-10.0, fminf(10.0, out_particles[ii].m_z));
            unsigned z_blue = floorf(0xFF * 0.9*((z_clamp_blue + 10.0) / 20.0) + 0.1);
            pixel_buf[(y*width) + x] = z_blue;
            pixel_buf[(y*width) + x] |= (0xFF0000*in_particles[ii].m_type);
        }
    }

}

__global__ void cuda_copy_kernel(const Particle* __restrict__ out_particles,
                                 Particle* __restrict__ in_particles,
                                 unsigned num_particles)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned num_threads = gridDim.x * blockDim.x;
    
    for(unsigned ii = idx; ii < num_particles; ii += num_threads)
    {
        in_particles[ii].m_x = out_particles[ii].m_x;
        in_particles[ii].m_y = out_particles[ii].m_y;
        in_particles[ii].m_z = out_particles[ii].m_z;
        in_particles[ii].m_vx = out_particles[ii].m_vx;
        in_particles[ii].m_vy = out_particles[ii].m_vy;
        in_particles[ii].m_vz = out_particles[ii].m_vz;
        in_particles[ii].m_ax = out_particles[ii].m_ax;
        in_particles[ii].m_ay = out_particles[ii].m_ay;
        in_particles[ii].m_az = out_particles[ii].m_az;
    }    
}



Particle *output_particles;
void init()
{
    cudaMalloc(&output_particles, Parameters::num_particles * sizeof(Particle));
}

void mega_kernel(Particle *particles, const int num_particles,
                 const float min_x, const float max_x,
                 const float min_y, const float max_y,
                 unsigned *pixel_buf, const int width, const int height)
{
    for(unsigned i = 0; i < Parameters::num_iterations; i++)
    {
        cuda_mega_kernel<<<Parameters::num_blocks, Parameters::blocksize>>>(particles, output_particles, num_particles,
                                                                            min_x, max_x,
                                                                            min_y, max_y,
                                                                            pixel_buf, width, height);
        cuda_copy_kernel<<<Parameters::num_blocks, Parameters::blocksize>>>(output_particles, particles, num_particles);
    }
}
    
    
void compute_forces(Particle *particles, int num_particles)
{
    float min_x = -10.0;
    float min_y = -10.0;
    float max_x = 10.0;
    float max_y = 10.0;
    
    unsigned *pixel_buf;
    cudaMalloc(&pixel_buf, sizeof(unsigned) * Parameters::width * Parameters::height);

    mega_kernel(particles, num_particles,
                min_x, max_x, min_y, max_y,
                pixel_buf, Parameters::width, Parameters::height);
    
    cudaFree(pixel_buf);
}
void euler_update(Particle *particles, int num_particles)
{
    float min_x = -10.0;
    float min_y = -10.0;
    float max_x = 10.0;
    float max_y = 10.0;

    unsigned *pixel_buf;
    cudaMalloc(&pixel_buf, sizeof(unsigned) * Parameters::width * Parameters::height);

    mega_kernel(particles, num_particles,
                min_x, max_x, min_y, max_y,
                pixel_buf, Parameters::width, Parameters::height);
    
    cudaFree(pixel_buf);
}

void draw_particles(Particle *particles, const int num_particles,
                    const float min_x, const float max_x,
                    const float min_y, const float max_y,
                    unsigned *pixel_buf, const int width, const int height)
{
    mega_kernel(particles, num_particles,
                min_x, max_x, min_y, max_y,
                pixel_buf, width, height);
}






    

// min max stuff.
// TODO: Optimize this reduction kernel https://cuvilib.com/Reduction.pdf
__global__ void cuda_get_min_max(const int num_items,
                                 const float *min_x_in, const float *max_x_in,
                                 const float *min_y_in, const float *max_y_in,
                                 float *min_x, float *max_x,
                                 float *min_y, float *max_y)
{
    extern __shared__ float sdata[];

    float *min_x_sdata = &sdata[0*blockDim.x];
    float *min_y_sdata = &sdata[1*blockDim.x];
    float *max_x_sdata = &sdata[2*blockDim.x];
    float *max_y_sdata = &sdata[3*blockDim.x];
    
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_items)
    {
        min_x_sdata[tid] = min_x_in[idx];
        max_x_sdata[tid] = max_x_in[idx];
        min_y_sdata[tid] = min_y_in[idx];
        max_y_sdata[tid] = max_y_in[idx];
    }
    else
    {
        // If we have more threads than items, then just repeat the first item.
        min_x_sdata[tid] = min_x_in[0];
        max_x_sdata[tid] = max_x_in[0];
        min_y_sdata[tid] = min_y_in[0];
        max_y_sdata[tid] = max_y_in[0];
    }
    __syncthreads();
    
    for(unsigned i = 1; i < blockDim.x; i *= 2)
    {
        if(tid % (2*i) == 0)
        {
            min_x_sdata[tid] = fminf(min_x_sdata[tid+i], min_x_sdata[tid]);
            max_x_sdata[tid] = fmaxf(max_x_sdata[tid+i], max_x_sdata[tid]);
            min_y_sdata[tid] = fminf(min_y_sdata[tid+i], min_y_sdata[tid]);
            max_y_sdata[tid] = fmaxf(max_y_sdata[tid+i], max_y_sdata[tid]);
        }

        __syncthreads();
    }

    if(tid == 0)
    {
        min_x[blockIdx.x] = min_x_sdata[0];
        max_x[blockIdx.x] = max_x_sdata[0];
        min_y[blockIdx.x] = min_y_sdata[0];
        max_y[blockIdx.x] = max_y_sdata[0];
    }
}

__global__ void cuda_get_xy_vec(const Particle *particles,
                                const unsigned num_particles,
                                float *x, float *y)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned num_threads = gridDim.x * blockDim.x;
    
    for(unsigned i = idx; i < num_particles; i += num_threads)
    {
        x[i] = particles[i].m_x;
        y[i] = particles[i].m_y;
    }
}
    
void get_min_max(const Particle *particles,
                 const int num_particles,
                 float *min_x, float *max_x,
                 float *min_y, float *max_y)
{
    assert(num_particles <= (Parameters::num_blocks*Parameters::blocksize));
    
    // Use min_x and min_y as temporary variables
    cuda_get_xy_vec<<<Parameters::num_blocks, Parameters::blocksize>>>(particles, num_particles, min_x, min_y);
    
    cuda_get_min_max<<<Parameters::num_blocks, Parameters::blocksize, Parameters::blocksize*4>>>
        (num_particles,
         min_x, min_x,
         min_y, min_y,
         min_x, max_x,
         min_y, max_y);
    
    unsigned num_items_to_reduce = Parameters::num_blocks;
    
    cuda_get_min_max<<<Parameters::num_blocks, Parameters::blocksize, Parameters::blocksize*4>>>
        (num_items_to_reduce,
         min_x, max_x,
         min_y, max_y,
         min_x, max_x,
         min_y, max_y);
    
    num_items_to_reduce = std::ceil((float)num_items_to_reduce / Parameters::blocksize);
    
    for(unsigned i = 0; i < num_items_to_reduce; i++)
    {
        min_x[0] = std::min(min_x[0], min_x[i]);
        max_x[0] = std::max(max_x[0], max_x[i]);
        min_y[0] = std::min(min_y[0], min_y[i]);
        max_y[0] = std::max(max_y[0], max_y[i]);
    }
}
    
    
}
