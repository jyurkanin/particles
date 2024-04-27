#include "kernels.h"
#include "Parameters.h"

#include <cassert>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace kernels
{

__global__ void cuda_damping(const float* __restrict__ in_vx_vec, const float* __restrict__ in_vy_vec, const float* __restrict__ in_vz_vec,
                             const float* __restrict__ in_mass_vec,
                             float* __restrict__ out_ax_vec, float* __restrict__ out_ay_vec, float* __restrict__ out_az_vec,
                             const int num_particles)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned num_threads = gridDim.x * blockDim.x;

    if(idx < num_particles)
    {
        const float inv_mass = in_mass_vec[idx];
        const float damp_inv_mass = (1e-2 / inv_mass);
        out_ax_vec[idx] = -damp_inv_mass * in_vx_vec[idx];
        out_ay_vec[idx] = -damp_inv_mass * in_vy_vec[idx];
        out_az_vec[idx] = -damp_inv_mass * in_vz_vec[idx];
    }
}
    

// out_big_ax_vec has shape [gridDim.y, num_particles] 
__global__ void cuda_tiled_forces(const float* __restrict__ in_x_vec, const float* __restrict__ in_y_vec, const float* __restrict__ in_z_vec,
                                  const float* __restrict__ in_vx_vec, const float* __restrict__ in_vy_vec, const float* __restrict__ in_vz_vec,
                                  const float* __restrict__ in_ax_vec, const float* __restrict__ in_ay_vec, const float* __restrict__ in_az_vec,
                                  const float* __restrict__ in_mass_vec,
                                  float* __restrict__ out_big_ax_vec, float* __restrict__ out_big_ay_vec, float* __restrict__ out_big_az_vec,
                                  const int num_particles)
{    
    __shared__ float x_slice_other[blockDim.x];
    __shared__ float y_slice_other[blockDim.x];
    __shared__ float z_slice_other[blockDim.x];
    __shared__ float mass_slice_other[blockDim.x];
    
	unsigned num_threads = gridDim.x * gridDim.y * blockDim.x;
    unsigned idx = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned idx_other = threadIdx.x + (blockIdx.y * blockDim.x);
    // assert(gridDim.x == gridDim.y);
    // This will be true because both gridDim.x and gridDim.y
    // will be computed to be the minimize size necessary such that, gridDim.x * blockDim.x

    if(idx < num_particles)
    {    
        float x_for_idx = in_x_vec[idx];
        float y_for_idx = in_y_vec[idx];
        float z_for_idx = in_z_vec[idx];

        if(idx_other < num_particles)
        {
            x_slice_other[threadIdx.x] = in_x_vec[idx_other];
            y_slice_other[threadIdx.x] = in_y_vec[idx_other];
            z_slice_other[threadIdx.x] = in_z_vec[idx_other];
            mass_slice_other[threadIdx.x] = in_mass_vec[idx_other];   
        }
        else
        {
            x_slice_other[threadIdx.x] = x_for_idx; // This results in zero force and no effect on the rest of the program.
            y_slice_other[threadIdx.x] = y_for_idx;
            z_slice_other[threadIdx.x] = z_for_idx;
            mass_slice_other[threadIdx.x] = 1.0; // Arbitrary
        }
    }
    
    __syncthreads(); // Guarantees all shared memory will be loaded.
    // TODO: This only works when the num_threads > num_particles...
    
    if(idx < num_particles)    
    {
        for(unsigned ii = 0; ii < blockDim.x; ii++)
        {
            float dx = x_for_idx - x_slice_other[ii];
            float dy = y_for_idx - y_slice_other[ii];
            float dz = z_for_idx - z_slice_other[ii];
            float force_temp = mass_slice_other[ii] / fmaxf(1e-6, dx*dx + dy*dy + dz*dz);
            ax += (force_temp * dx);
            ay += (force_temp * dy);
            az += (force_temp * dz);
        }
    
        // Collect all the grid results for reduction later.
        // when out_big_ax has shape [gridDim.y, num_particles] then it has coalesced memory access. (Good)
        const unsigned particle_and_grid_idx = (blockIdx.y * num_particles) + idx;
        out_big_ax_vec[particle_and_grid_idx] = ax;
        out_big_ay_vec[particle_and_grid_idx] = ay;
        out_big_az_vec[particle_and_grid_idx] = az;
    }
}

__global__ void cuda_gay()
{
	// compute accelerations
	for(unsigned ii = idx; ii < num_particles; ii += num_threads)
	{
        const float inv_mass = 1.0 / in_mass_vec[ii];
		const float damp = 1e-2;
		float ax = -damp*in_x_vec[ii] * inv_mass;
		float ay = -damp*in_y_vec[ii] * inv_mass;
		float az = -damp*in_z_vec[ii] * inv_mass;
        
		for(unsigned jj = 0; jj < num_particles; jj++)
		{
			float dx = in_x_vec[jj] - in_x_vec[ii];
			float dy = in_y_vec[jj] - in_y_vec[ii];
			float dz = in_z_vec[jj] - in_z_vec[ii];
            float force_temp = in_mass_vec[jj] / fmaxf(1e-6, dx*dx + dy*dy + dz*dz);
            
            ax += force_temp * dx;
            ay += force_temp * dy;
            az += force_temp * dz;            
		}
        
        out_ax_vec[ii] = ax;
        out_ay_vec[ii] = ay;
        out_az_vec[ii] = az;
    }
}

__global__ void cuda_elementwise2(const float* __restrict__ in_x_vec, const float* __restrict__ in_y_vec, const float* __restrict__ in_z_vec,
                                  const float* __restrict__ in_vx_vec, const float* __restrict__ in_vy_vec, const float* __restrict__ in_vz_vec,
                                  float* __restrict__ out_x_vec, float* __restrict__ out_y_vec, float* __restrict__ out_z_vec,
                                  float* __restrict__ out_vx_vec, float* __restrict__ out_vy_vec, float* __restrict__ out_vz_vec,
                                  const float* __restrict__ out_ax_vec, const float* __restrict__ out_ay_vec, const float* __restrict__ out_az_vec,
                                  const float* __restrict__ in_type_vec,
                                  const int num_particles,
                                  unsigned* __restrict__ pixel_buf, const int width, const int height)
{
    const float scalar = 1e-4;
    const float min_x = -100;
    const float max_x = 100;
    const float min_y = -100;
    const float max_y = 100;
    
    const float width_max_min_x_inv = width / (max_x - min_x);
    const float height_max_min_y_inv = height / (max_y - min_y);

	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned num_threads = gridDim.x * blockDim.x;
    
	// compute accelerations
	for(unsigned ii = idx; ii < num_particles; ii += num_threads)
	{		
		// Euler update        
        out_vx_vec[ii] = in_vx_vec[ii] + (out_ax_vec[ii]*scalar);
        out_vy_vec[ii] = in_vy_vec[ii] + (out_ay_vec[ii]*scalar);
        out_vz_vec[ii] = in_vz_vec[ii] + (out_az_vec[ii]*scalar);

        out_x_vec[ii] = in_x_vec[ii] + (out_vx_vec[ii]*scalar);
        out_y_vec[ii] = in_y_vec[ii] + (out_vy_vec[ii]*scalar);
        out_z_vec[ii] = in_z_vec[ii] + (out_vz_vec[ii]*scalar);        

		// Draw
		int x = (int)((out_x_vec[ii] - min_x) * width_max_min_x_inv);
		int y = (int)((out_y_vec[ii] - min_y) * height_max_min_y_inv);
		
		if((x >= 0) && (x < width) && (y >= 0) && (y < height))
		{
			float z_clamp_blue = fmaxf(-10.0, fminf(10.0, out_z_vec[ii]));
			unsigned z_color = floorf(0xFF * 0.9 * (0.05 * (z_clamp_blue + 10.0)) + 0.1);
            unsigned type_color = (unsigned)(0xFF0000*in_type_vec[ii]);
            
			pixel_buf[(y*width) + x] = z_color | type_color;
		}
	}
}

__global__ void cuda_copy_old_new(float* __restrict__ in_x_vec, float* __restrict__ in_y_vec, float* __restrict__ in_z_vec,
                                  float* __restrict__ in_vx_vec, float* __restrict__ in_vy_vec, float* __restrict__ in_vz_vec,
                                  const float* __restrict__ out_x_vec, const float* __restrict__ out_y_vec, const float* __restrict__ out_z_vec,
                                  const float* __restrict__ out_vx_vec, const float* __restrict__ out_vy_vec, const float* __restrict__ out_vz_vec,
                                  const unsigned num_particles
                                  )
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned num_threads = gridDim.x * blockDim.x;
	
	for(unsigned ii = idx; ii < num_particles; ii += num_threads)
	{
        in_x_vec[ii] = out_x_vec[ii];
        in_y_vec[ii] = out_y_vec[ii];
        in_z_vec[ii] = out_z_vec[ii];
        
        in_vx_vec[ii] = out_vx_vec[ii];
        in_vy_vec[ii] = out_vy_vec[ii];
        in_vz_vec[ii] = out_vz_vec[ii];
    }
}   

float *out_x_vec;
float *out_y_vec;
float *out_z_vec;
float *out_vx_vec;
float *out_vy_vec;
float *out_vz_vec;
float *out_ax_vec;
float *out_ay_vec;
float *out_az_vec;

float *temp_big_ax_vec;
float *temp_big_ay_vec;
float *temp_big_az_vec;
    
unsigned g_grid_dim;

void init()
{
	cudaMalloc(&out_x_vec, Parameters::num_particles * sizeof(float));
    cudaMalloc(&out_y_vec, Parameters::num_particles * sizeof(float));
    cudaMalloc(&out_z_vec, Parameters::num_particles * sizeof(float));
	cudaMalloc(&out_vx_vec, Parameters::num_particles * sizeof(float));
    cudaMalloc(&out_vy_vec, Parameters::num_particles * sizeof(float));
    cudaMalloc(&out_vz_vec, Parameters::num_particles * sizeof(float));
	cudaMalloc(&out_ax_vec, Parameters::num_particles * sizeof(float));
    cudaMalloc(&out_ay_vec, Parameters::num_particles * sizeof(float));
    cudaMalloc(&out_az_vec, Parameters::num_particles * sizeof(float));

    g_grid_dim = 1 + (Parameters::num_particles / Parameters::blocksize);

    unsigned num_bytes_needed_for_shared_mem = Parameters::blocksize * sizeof(float) * 4;
    assert(num_bytes_needed_for_shared_mem <= (4096*1024));
    
    cudaMalloc(&temp_big_ax_vec, Parameters::num_particles * g_grid_dim * sizeof(float));
    cudaMalloc(&temp_big_ay_vec, Parameters::num_particles * g_grid_dim * sizeof(float));
    cudaMalloc(&temp_big_az_vec, Parameters::num_particles * g_grid_dim * sizeof(float));
}

void mega_kernel(float *x_vec, float *y_vec, float *z_vec,
                 float *vx_vec, float *vy_vec, float *vz_vec,
                 float *ax_vec, float *ay_vec, float *az_vec,
                 float *mass_vec, float *type_vec,
                 const int num_particles,
				 unsigned *pixel_buf, const int width, const int height)
{
	for(unsigned i = 0; i < Parameters::num_iterations; i++)
	{
        cuda_damping<<<Parameters::num_blocks, Parameters::blocksize>>>
            (vx_vec, vy_vec, vz_vec,
             mass_vec,
             ax_vec, ay_vec, az_vec,
             num_particles);
        
		cuda_tiled_forces<<<Parameters::num_blocks, Parameters::blocksize>>>
    (x_vec, y_vec, z_vec,
            vx_vec, vy_vec, vz_vec,
            ax_vec, ay_vec, az_vec,
            mass_vec,
            temp_big_ax_vec, temp_big_ay_vec, temp_big_az_vec,
            num_particles);

    cuda_sum_along_blocks<<<Parameters::num_blocks, Parameters::blocksize>>>
        (ax_vec, ay_vec, az_vec,
            temp_big_ax_vec, temp_big_ay_vec, temp_big_az_vec, num_particles);
    
    cuda_elementwise2<<<Parameters::num_blocks, Parameters::blocksize>>>
        (x_vec, y_vec, z_vec,
             vx_vec, vy_vec, vz_vec,
             out_x_vec,  out_y_vec,  out_z_vec,
             out_vx_vec, out_vy_vec, out_vz_vec,
             out_ax_vec, out_ay_vec, out_az_vec,
             type_vec,
             num_particles,
             pixel_buf, width, height);
        
        cuda_copy_old_new<<<Parameters::num_blocks, Parameters::blocksize>>>
            (x_vec, y_vec, z_vec,
             vx_vec, vy_vec, vz_vec,
             out_x_vec, out_y_vec, out_z_vec,
             out_vx_vec, out_vy_vec, out_vz_vec,
             num_particles);

	}
}
	
	
void compute_forces(float *x_vec, float *y_vec, float *z_vec,
                    float *vx_vec, float *vy_vec, float *vz_vec,
                    float *ax_vec, float *ay_vec, float *az_vec,
                    float *mass_vec, float *type_vec,
                    const int num_particles)
{
	unsigned *pixel_buf;
	cudaMalloc(&pixel_buf, sizeof(unsigned) * Parameters::width * Parameters::height);

	mega_kernel(x_vec, y_vec, z_vec,
                vx_vec, vy_vec, vz_vec,
                ax_vec, ay_vec, az_vec,
                mass_vec, type_vec,
                num_particles,
				pixel_buf, Parameters::width, Parameters::height);
	
	cudaFree(pixel_buf);
}
void euler_update(float *x_vec, float *y_vec, float *z_vec,
                  float *vx_vec, float *vy_vec, float *vz_vec,
                  float *ax_vec, float *ay_vec, float *az_vec,
                  float *mass_vec, float *type_vec,
                  const int num_particles)
{
	unsigned *pixel_buf;
	cudaMalloc(&pixel_buf, sizeof(unsigned) * Parameters::width * Parameters::height);

	mega_kernel(x_vec, y_vec, z_vec,
                vx_vec, vy_vec, vz_vec,
                ax_vec, ay_vec, az_vec,
                mass_vec, type_vec,
                num_particles,
				pixel_buf, Parameters::width, Parameters::height);    
	
	cudaFree(pixel_buf);
}

void draw_particles(float *x_vec, float *y_vec, float *z_vec,
                    float *vx_vec, float *vy_vec, float *vz_vec,
                    float *ax_vec, float *ay_vec, float *az_vec,
                    float *mass_vec, float *type_vec,
                    const int num_particles,
					unsigned *pixel_buf, const int width, const int height)
{
    mega_kernel(x_vec, y_vec, z_vec,
                vx_vec, vy_vec, vz_vec,
                ax_vec, ay_vec, az_vec,
                mass_vec, type_vec,
                num_particles,
				pixel_buf, Parameters::width, Parameters::height);
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
