#pragma once

#include "Parameters.h"
#include "Particle.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <array>
#include <vector>

class GpuParticleEngine3
{
public:
    GpuParticleEngine3();
    ~GpuParticleEngine3();

    std::array<Particle, Parameters::num_particles> getParticles()
    {
        std::array<Particle, Parameters::num_particles> particles;
        for(unsigned i = 0; i < particles.size(); i++)
        {
            particles[i].m_x = m_x_vec[i];
            particles[i].m_y = m_y_vec[i];
            particles[i].m_z = m_z_vec[i];

            particles[i].m_vx = m_vx_vec[i];
            particles[i].m_vy = m_vy_vec[i];
            particles[i].m_vz = m_vz_vec[i];

            particles[i].m_ax = m_ax_vec[i];
            particles[i].m_ay = m_ay_vec[i];
            particles[i].m_az = m_az_vec[i];

            particles[i].m_mass = m_mass_vec[i];
            particles[i].m_type = m_type_vec[i];            
        }
        
        return particles;
    }
    std::array<unsigned int, Parameters::width*Parameters::height> getPixelBuf()
    {
        std::array<unsigned int, Parameters::width*Parameters::height> pixel_buf;
        for(unsigned i = 0; i < pixel_buf.size(); i++)
        {
            pixel_buf[i] = m_cuda_pixel_buf[i];
        }
        return pixel_buf;
    }
    
    void initialize();

    void clearPixelBuf(int cnt);
    void runIteration();
    void draw(unsigned int *pixbuf);

    void compute_forces();
    void euler_update();
    void draw_particles();

    
private:
    unsigned int *m_cuda_pixel_buf;

    void *m_mem_vec;
    
    float *m_x_vec;
    float *m_y_vec;
    float *m_z_vec;
    
    float *m_vx_vec;
    float *m_vy_vec;
    float *m_vz_vec;
    
    float *m_ax_vec;
    float *m_ay_vec;
    float *m_az_vec;
    
    float *m_mass_vec;
    float *m_type_vec;
};

