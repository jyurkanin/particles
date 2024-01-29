#include "ParticleEngine.h"
#include "kernels/Particle.cuh"
#include "Parameters.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

ParticleEngine::ParticleEngine()
{
    cudaMallocManaged(&m_particles, m_num_particles*sizeof(Particle));
    // cudaMallocManaged(&m_cuda_pixel_buf, sizeof(unsigned char)*3*Parameters::m_width * Parameters::m_height);
    m_cuda_pixel_buf = new unsigned char[3*Parameters::m_width * Parameters::m_height];
}



ParticleEngine::~ParticleEngine()
{
    cudaFree(m_cuda_pixel_buf);
    cudaFree(m_particles);
}


void ParticleEngine::initialize()
{
    //std::vector<Particle> cpu_particles(m_num_particles);
    Particle cpu_particles[m_num_particles];

    for(int i = 0; i < m_num_particles; i++)
    {
        float angle = (i*2*M_PI) / m_num_particles;
        cpu_particles[i].m_x = std::cos(angle);
        cpu_particles[i].m_y = std::sin(angle);
        
    }

    cudaMemcpy(m_particles, cpu_particles, sizeof(Particle)*m_num_particles, cudaMemcpyHostToDevice);
}


void ParticleEngine::runIteration()
{
    cpu_compute_forces(m_particles, m_num_particles);
    cpu_euler_update(m_particles,   m_num_particles);
    
    float max_x = 10;
    float min_x = -10;
    float max_y = 10;
    float min_y = -10;
    
    // get_max_x(m_particles, m_num_particles, max_x);
    // get_min_x(m_particles, m_num_particles, min_x);
    // get_max_y(m_particles, m_num_particles, max_y);
    // get_min_y(m_particles, m_num_particles, min_y);
    
    // draw_particles(m_particles, m_num_particles,
    //                min_x, min_y, max_x, max_y,
    //                m_cuda_pixel_buf, Parameters::m_width, Parameters::m_height);
    
    // cudaDeviceSynchronize();

    cpu_draw_particles(m_particles, m_num_particles,
                       min_x, min_y, max_x, max_y,
                       m_cuda_pixel_buf, Parameters::m_width, Parameters::m_height);
    
    
    //std::cout << m_particles[1].m_x << ", " << m_particles[1].m_vx << "\n";
}

void ParticleEngine::draw(unsigned char *pixbuf)
{
    const int num_pixels = Parameters::m_width * Parameters::m_height;
    
    cudaMemcpy(pixbuf, m_cuda_pixel_buf, sizeof(unsigned char)*3*num_pixels, cudaMemcpyDeviceToHost);
}

void ParticleEngine::cpu_compute_forces(Particle *particles, int num_particles)
{
    for(int i = 0; i < num_particles; i++)
    {
        float fx = 0;
        float fy = 0;
        float fz = 0;
        
        for(int j = 0; j < num_particles; j++)
        {
            if(i == j) continue;
            
            float dx = particles[i].m_x - particles[j].m_x;
            float dy = particles[i].m_y - particles[j].m_y;
            float dz = particles[i].m_z - particles[j].m_z;
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            float force = 0.1*dist;
            
            fx += (dx*force) / dist;
            fy += (dy*force) / dist;
            fz += (dz*force) / dist;
        }
        
        particles[i].m_vx = fx / particles[i].m_mass;
        particles[i].m_vy = fy / particles[i].m_mass;
        particles[i].m_vz = fz / particles[i].m_mass;
    }    

}

void ParticleEngine::cpu_euler_update(Particle *particles, int num_particles)
{
    const float timestep = 1e-3;
    for(int i = 0; i < num_particles; i++)
    {
        particles[i].m_x += particles[i].m_vx*timestep;
        particles[i].m_y += particles[i].m_vy*timestep;
        particles[i].m_z += particles[i].m_vz*timestep;
    }

}

void ParticleEngine::cpu_draw_particles(const Particle *particles, const int num_particles,
                                        const float min_x, const float min_y,
                                        const float max_x, const float max_y,
                                        unsigned char *pixelbuf, const int width, const int height)
{
    for(int i = 0; i < num_particles; i++)
    {
        int x = (int)(width * (particles[i].m_x - min_x) / (max_x - min_x));
        int y = (int)(height * (particles[i].m_y - min_y) / (max_y - min_y));
        
        pixelbuf[3*(y*width + x)] = 0xFF; // Blue
    }
}


