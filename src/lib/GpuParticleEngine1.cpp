#include "GpuParticleEngine1.h"
#include "kernels/kernels1.cuh"
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

GpuParticleEngine1::GpuParticleEngine1()
{
    //todo: replace managed with device
    cudaMallocManaged(&m_particles, Parameters::num_particles * sizeof(Particle));
    cudaMallocManaged(&m_cuda_pixel_buf, sizeof(unsigned int) * Parameters::width * Parameters::height);
}



GpuParticleEngine1::~GpuParticleEngine1()
{
    cudaFree(m_particles);
    cudaFree(m_cuda_pixel_buf);
}


void GpuParticleEngine1::initialize()
{
    srand(Parameters::seed);
    
    Particle particles[Parameters::num_particles];
    particles[0].m_x = 0.0;
    particles[0].m_y = 0.0;
    particles[0].m_vx = 0.0;
    particles[0].m_vy = 0.0;
    particles[0].m_mass = 1.0e6;
    particles[0].m_type = 1;
    
    for(int i = 1; i < Parameters::num_particles; i++)
    {
        float angle = (i*2*M_PI) / Parameters::num_particles;
        
        particles[i].m_x = 10.0*std::cos(angle);
        particles[i].m_y = 10.0*std::sin(angle);
        particles[i].m_vx = getRand(0,100)*std::sin(angle);
        particles[i].m_vy = getRand(0,100)*std::cos(angle);
        particles[i].m_vz = getRand(0,10);
    }
        
    cudaMemcpy(m_particles, particles, sizeof(Particle)*Parameters::num_particles, cudaMemcpyHostToDevice);
}


void GpuParticleEngine::get_min_max(float &min_x, float &max_x, float &min_y, float &max_y)
{
    float *gpu_min_x;
    float *gpu_max_x;
    float *gpu_min_y;
    float *gpu_max_y;
    
    cudaMallocManaged(&gpu_min_x, sizeof(float));
    cudaMallocManaged(&gpu_max_x, sizeof(float));
    cudaMallocManaged(&gpu_min_y, sizeof(float));
    cudaMallocManaged(&gpu_max_y, sizeof(float));
        
    kernels1::get_min_max<<<m_num_blocks, m_block_size>>>(m_particles,
                                                          Parameters::num_particles,
                                                          gpu_min_x,
                                                          gpu_max_x,
                                                          gpu_min_y,
                                                          gpu_max_y);
    cudaDeviceSynchronize();
    
    min_x = *gpu_min_x;
    max_x = *gpu_max_x;
    min_y = *gpu_min_y;
    max_y = *gpu_max_y;
}


void GpuParticleEngine1::runIteration(int cnt)
{
    float max_x = 100;
    float min_x = -100;
    float max_y = 100;
    float min_y = -100;

    kernels1::get_min_max(m_particles, Parameters::num_particles, min_x, max_x, min_y, max_y);
    cudaDeviceSynchronize();
    
    if(cnt % 1 == 0)
    {
        for(int i = 0; i < (Parameters::width*Parameters::height); i++)
        {
            m_cuda_pixel_buf[i] = 0;
        }
    }
    
    for(int i = 0; i < 1000; i++)
    {
        // kernels1::compute_forces(m_particles, Parameters::num_particles);
        // kernels1::euler_update(m_particles,   Parameters::num_particles);
        // kernels1::draw_particles(m_particles, Parameters::num_particles,
        //                          min_x, min_y, max_x, max_y,
        //                          m_cuda_pixel_buf, Parameters::width, Parameters::height);

    }
    
}

void GpuParticleEngine1::draw(unsigned int *pixbuf)
{
    const int num_pixels = Parameters::width*Parameters::height;
    cudaMemcpy(pixbuf, m_cuda_pixel_buf, sizeof(unsigned int)*num_pixels, cudaMemcpyDeviceToHost);
}

std::vector<std::string> GpuParticleEngine1::getParticleText()
{
    std::vector<std::string> particle_info;

    double energy;
    // kernels1::getTotalEnergy(m_particles, Parameters::num_particles, energy);
    cudaDeviceSynchronize();
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << energy; //TODO: do I need to memcpy from device to host
    particle_info.push_back(ss.str());
    
    for(int i = 0; i < Parameters::num_particles; i++)
    {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << m_particles[i].m_x << "," << std::setw(6);
        stream << std::fixed << std::setprecision(2) << m_particles[i].m_y << "," << std::setw(6);
        stream << std::fixed << std::setprecision(2) << m_particles[i].m_z;

        particle_info.push_back(stream.str());
    }
    
    return particle_info;
}

void GpuParticleEngine1::compute_forces(Particle *particles, int num_particles)
{

}

void GpuParticleEngine1::euler_update(Particle *particles, int num_particles)
{

}

void GpuParticleEngine1::draw_particles(const Particle *particles, const int num_particles,
                                        const float min_x, const float min_y,
                                        const float max_x, const float max_y,
                                        unsigned int *pixelbuf, const int width, const int height)
{
    
}


