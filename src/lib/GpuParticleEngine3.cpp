#include "GpuParticleEngine3.h"
#include "kernels/kernels.h"
#include "Parameters.h"


#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdlib>


inline float getRand(const float min, const float max)
{
    return ((max-min)*((float)rand() / (float)RAND_MAX)) + min;
}

GpuParticleEngine3::GpuParticleEngine3()
{
    //todo: replace managed with device
    unsigned bytes_per_vec = Parameters::num_particles * sizeof(float);
    unsigned num_vec = 11;
    unsigned num_pixel_bytes = sizeof(unsigned int) * Parameters::width * Parameters::height;
    
    cudaMallocManaged(&m_mem_vec, (bytes_per_vec*num_vec) + num_pixel_bytes);

    m_x_vec = (float*)(m_mem_vec + (0*bytes_per_vec));
    m_y_vec = (float*)(m_mem_vec + (1*bytes_per_vec));
    m_z_vec = (float*)(m_mem_vec + (2*bytes_per_vec));
    
    m_vx_vec = (float*)(m_mem_vec + (3*bytes_per_vec));
    m_vy_vec = (float*)(m_mem_vec + (4*bytes_per_vec));
    m_vz_vec = (float*)(m_mem_vec + (5*bytes_per_vec));
     
    m_ax_vec = (float*)(m_mem_vec + (6*bytes_per_vec));
    m_ay_vec = (float*)(m_mem_vec + (7*bytes_per_vec));
    m_az_vec = (float*)(m_mem_vec + (8*bytes_per_vec));
    
    m_mass_vec = (float*)(m_mem_vec + (9*bytes_per_vec));
    m_type_vec = (float*)(m_mem_vec + (10*bytes_per_vec));

    m_cuda_pixel_buf = (unsigned*) (m_mem_vec + (num_vec*bytes_per_vec));
        
    kernels::init();
}



GpuParticleEngine3::~GpuParticleEngine3()
{
    cudaFree(m_mem_vec);
}


void GpuParticleEngine3::initialize()
{
    srand(Parameters::seed);
    
    m_x_vec[0] = -40.0;
    m_y_vec[0] = 0.0;
    m_z_vec[0] = 0.0;
    m_vx_vec[0] = 0.0;
    m_vy_vec[0] = -1000.0;
    m_vz_vec[0] = 0.0;
    m_ax_vec[0] = 0.0;
    m_ay_vec[0] = 0.0;
    m_az_vec[0] = 0.0;
    m_mass_vec[0] = 1.0e6;
    m_type_vec[0] = 1;

    m_x_vec[1] = 40.0;
    m_y_vec[1] = 0.0;
    m_vx_vec[1] = 0.0;
    m_vy_vec[1] = 1000.0;
    m_mass_vec[1] = 1.0e6;
    m_type_vec[1] = 1;
    
    for(int i = 2; i < Parameters::num_particles; i++)
    {
        float angle = (i*2*M_PI) / Parameters::num_particles;
        
        m_x_vec[i] = 10.0*std::cos(angle);
        m_y_vec[i] = 10.0*std::sin(angle);
        m_z_vec[i] = ((20.0*i) / (Parameters::num_particles - 1)) - 10.0;
        
        m_vx_vec[i] = -100*std::sin(angle);
        m_vy_vec[i] = 100*std::cos(angle);
        m_vz_vec[i] = 0.0;
        
        m_ax_vec[i] = 0.0;
        m_ay_vec[i] = 0.0;
        m_az_vec[i] = 0.0;
        
        m_mass_vec[i] = 10.0;
        m_type_vec[i] = 0;
    }

}

void GpuParticleEngine3::clearPixelBuf(int cnt)
{
    if(cnt % 1 == 0)
    {
        for(int i = 0; i < (Parameters::width*Parameters::height); i++)
        {
            m_cuda_pixel_buf[i] = 0;
        }
    }    
}

void GpuParticleEngine3::runIteration()
{    
    // create runIteration kernel
    kernels::mega_kernel(m_x_vec, m_y_vec, m_z_vec,
                         m_vx_vec, m_vy_vec, m_vz_vec,
                         m_ax_vec, m_ay_vec, m_az_vec,
                         m_mass_vec, m_type_vec,
                         Parameters::num_particles,
                         m_cuda_pixel_buf, Parameters::width, Parameters::height);
}

void GpuParticleEngine3::draw(unsigned int *pixbuf)
{
    const int num_pixels = Parameters::width*Parameters::height;
    cudaMemcpy(pixbuf, m_cuda_pixel_buf, sizeof(unsigned int)*num_pixels, cudaMemcpyDeviceToHost);
}

void GpuParticleEngine3::compute_forces()
{
    kernels::compute_forces(m_x_vec, m_y_vec, m_z_vec,
                            m_vx_vec, m_vy_vec, m_vz_vec,
                            m_ax_vec, m_ay_vec, m_az_vec,
                            m_mass_vec, m_type_vec,
                            Parameters::num_particles);
    cudaDeviceSynchronize();
}

void GpuParticleEngine3::euler_update()
{
    kernels::euler_update(m_x_vec, m_y_vec, m_z_vec,
                          m_vx_vec, m_vy_vec, m_vz_vec,
                          m_ax_vec, m_ay_vec, m_az_vec,
                          m_mass_vec, m_type_vec,
                          Parameters::num_particles);
    cudaDeviceSynchronize();
}

void GpuParticleEngine3::draw_particles()
{
    kernels::draw_particles(m_x_vec, m_y_vec, m_z_vec,
                            m_vx_vec, m_vy_vec, m_vz_vec,
                            m_ax_vec, m_ay_vec, m_az_vec,
                            m_mass_vec, m_type_vec,
                            Parameters::num_particles,
                            m_cuda_pixel_buf,
                            Parameters::width, Parameters::height);
    cudaDeviceSynchronize();
}


