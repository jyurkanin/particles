#include "ParticleEngine.h"
#include "kernels/Particle.cuh"
#include "Parameters.h"


#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdlib>

inline void compute1(const Particle &p1,
                     const Particle &p2,
                     float &fx,
                     float &fy,
                     float &fz)
{
    float dx = p1.m_x - p2.m_x;
    float dy = p1.m_y - p2.m_y;
    float dz = p1.m_z - p2.m_z;
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    float force = p1.m_mass * p2.m_mass / dist;
            
    fx = (dx*force) / dist;
    fy = (dy*force) / dist;
    fz = (dz*force) / dist;
}



inline float getRand(const float min, const float max)
{
    return ((max-min)*((float)rand() / (float)RAND_MAX)) + min;
}


ParticleEngine::ParticleEngine()
{
    //cudaMallocManaged(&m_particles, m_num_particles*sizeof(Particle));
    // cudaMallocManaged(&m_cuda_pixel_buf, sizeof(unsigned int)*Parameters::m_width * Parameters::m_height);

    m_particles = new Particle[m_num_particles];
    m_cuda_pixel_buf = new unsigned int[Parameters::m_width * Parameters::m_height];
}



ParticleEngine::~ParticleEngine()
{
    cudaFree(m_particles);
    cudaFree(m_cuda_pixel_buf);
}


void ParticleEngine::initialize()
{
    //std::vector<Particle> cpu_particles(m_num_particles);
    Particle cpu_particles[m_num_particles];
    
    for(int i = 0; i < m_num_particles; i++)
    {
        float angle = (i*2*M_PI) / m_num_particles;
        
        cpu_particles[i].m_x = 10.0*std::cos(angle);
        cpu_particles[i].m_y = 10.0*std::sin(angle);
        cpu_particles[i].m_vx = getRand(0,100)*std::sin(angle);
        cpu_particles[i].m_vy = getRand(0,100)*std::cos(angle);
        cpu_particles[i].m_vz = getRand(0,10);
    }
    
    cpu_particles[0].m_x = 0.0;
    cpu_particles[0].m_y = 0.0;
    cpu_particles[0].m_vx = 0.0;
    cpu_particles[0].m_vy = 0.0;
    cpu_particles[0].m_mass = 1.0e6;
    cpu_particles[0].m_type = 1;
    
    //cudaMemcpy(m_particles, cpu_particles, sizeof(Particle)*m_num_particles, cudaMemcpyHostToDevice);
    memcpy(m_particles, cpu_particles, sizeof(Particle)*m_num_particles);
}


void ParticleEngine::cpu_get_min_max(float &min_x,
                                     float &max_x,
                                     float &min_y,
                                     float &max_y)
{
    min_x = m_particles[0].m_x;
    max_x = m_particles[0].m_x;
    min_y = m_particles[0].m_y;
    max_y = m_particles[0].m_y;
    
    for(int i = 1; i < m_num_particles; i++)
    {
        min_x = std::min(min_x, m_particles[i].m_x);
        max_x = std::max(max_x, m_particles[i].m_x);
        min_y = std::min(min_y, m_particles[i].m_y);
        max_y = std::max(max_y, m_particles[i].m_y);
    }
}

void ParticleEngine::runIteration(int cnt)
{
    float max_x = 100;
    float min_x = -100;
    float max_y = 100;
    float min_y = -100;

    cpu_get_min_max(min_x, max_x, min_y, max_y);

    if(cnt % 1 == 0)
    {
        for(int i = 0; i < (Parameters::m_width*Parameters::m_height); i++)
        {
            m_cuda_pixel_buf[i] = 0;
        }
    }
    
    for(int i = 0; i < 1000; i++)
    {
        cpu_compute_forces(m_particles, m_num_particles);
        cpu_euler_update(m_particles,   m_num_particles);
        cpu_draw_particles(m_particles, m_num_particles,
                           min_x, min_y, max_x, max_y,
                           m_cuda_pixel_buf, Parameters::m_width, Parameters::m_height);

    }
    
}

void ParticleEngine::draw(unsigned int *pixbuf)
{
    const int num_pixels = Parameters::m_width*Parameters::m_height;
    memcpy(pixbuf, m_cuda_pixel_buf, sizeof(unsigned int)*num_pixels);
    //cudaMemcpy(pixbuf, m_cuda_pixel_buf, sizeof(unsigned int)*num_pixels, cudaMemcpyDeviceToHost);
}

float ParticleEngine::getTotalEnergy()
{
    float kinetic_energy = 0;
    for(int i = 0; i < m_num_particles; i++)
    {
        kinetic_energy += m_particles[i].m_vx*m_particles[i].m_vx;
        kinetic_energy += m_particles[i].m_vy*m_particles[i].m_vy;
        kinetic_energy += m_particles[i].m_vz*m_particles[i].m_vz;
    }
    kinetic_energy *= 0.5;

    float potential_energy = 0;
    for(int i = 0; i < m_num_particles; i++)
    {
        for(int j = i+1; j < m_num_particles; j++)
        {
            float dx = m_particles[j].m_x - m_particles[i].m_x;
            float dy = m_particles[j].m_y - m_particles[i].m_y;
            float dz = m_particles[j].m_z - m_particles[i].m_z;
            
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            potential_energy += m_particles[i].m_mass * m_particles[j].m_mass / std::max(1e-6f, dist);
        }
    }
    
    return kinetic_energy - potential_energy;
}

std::vector<std::string> ParticleEngine::getParticleText()
{
    std::vector<std::string> particle_info;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << getTotalEnergy();
    particle_info.push_back(ss.str());
    
    for(int i = 0; i < m_num_particles; i++)
    {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << m_particles[i].m_x << "," << std::setw(6);
        stream << std::fixed << std::setprecision(2) << m_particles[i].m_y << "," << std::setw(6);
        stream << std::fixed << std::setprecision(2) << m_particles[i].m_z;

        particle_info.push_back(stream.str());
    }
    
    return particle_info;
}

void ParticleEngine::cpu_compute_forces(Particle *particles, int num_particles)
{
    for(int i = 0; i < num_particles; i++)
    {
        float fx = 0;
        float fy = 0;
        float fz = 0;

        // fx += -1e-3*particles[i].m_vx;
        // fy += -1e-3*particles[i].m_vy;
        // fz += -1e-3*particles[i].m_vz;
        
        for(int j = 0; j < num_particles; j++)
        {
            if(i == j) continue;
            
            float dfx;
            float dfy;
            float dfz;
            
            compute1(particles[j], particles[i], dfx, dfy, dfz);
            
            fx += dfx;
            fy += dfy;
            fz += dfz;
        }
        
        particles[i].m_ax = fx / particles[i].m_mass;
        particles[i].m_ay = fy / particles[i].m_mass;
        particles[i].m_az = fz / particles[i].m_mass;
    }    

}

void ParticleEngine::cpu_euler_update(Particle *particles, int num_particles)
{
    const float timestep = 1e-4;
    float max_norm = 0;
    for(int i = 0; i < num_particles; i++)
    {
        float ax = particles[i].m_ax;
        float ay = particles[i].m_ay;
        float az = particles[i].m_az;
        
        float vx = particles[i].m_vx;
        float vy = particles[i].m_vy;
        float vz = particles[i].m_vz;
        
        float norm = std::sqrt(ax*ax + ay*ay + az*az + vx*vx + vy*vy + vz*vz);
        
        max_norm = std::max(max_norm, norm);
    }
    
    float scalar = timestep / std::min(1e2f, max_norm);
    for(int i = 0; i < num_particles; i++)
    {        
        particles[i].m_vx += particles[i].m_ax*scalar;
        particles[i].m_vy += particles[i].m_ay*scalar;
        particles[i].m_vz += particles[i].m_az*scalar;

        particles[i].m_x += particles[i].m_vx*scalar;
        particles[i].m_y += particles[i].m_vy*scalar;
        particles[i].m_z += particles[i].m_vz*scalar;
    }

}

void ParticleEngine::cpu_draw_particles(const Particle *particles, const int num_particles,
                                        const float min_x, const float min_y,
                                        const float max_x, const float max_y,
                                        unsigned int *pixelbuf, const int width, const int height)
{
    float min_z = particles[0].m_z;
    float max_z = particles[0].m_z + 1e-6;
    for(int i = 1; i < num_particles; i++)
    {
        min_z = std::min(min_z, particles[i].m_z);
        max_z = std::max(max_z, particles[i].m_z);
    }
    
    for(int i = 0; i < num_particles; i++)
    {
        int x = (int)(width * (particles[i].m_x - min_x) / (max_x - min_x));
        int y = (int)(height * (particles[i].m_y - min_y) / (max_y - min_y));

        // * (particles[i].m_z - min_z) / (max_z - min_z);
        if((x >= 0) && (x < width) && (y >= 0) && (y < height))
        {
            if(particles[i].m_type == 0)
                pixelbuf[(y*width + x)] = 0xFF * (particles[i].m_z - min_z) / (max_z - min_z);
            if(particles[i].m_type == 1)
                pixelbuf[(y*width + x)] = 0xFF00;
        }
    }
}


