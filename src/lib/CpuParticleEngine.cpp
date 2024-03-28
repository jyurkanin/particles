#include "CpuParticleEngine.h"
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
    float dist = std::max(1e-6f, std::sqrt(dx*dx + dy*dy + dz*dz));
    float force = p1.m_mass * p2.m_mass / dist;
    fx = (dx*force) / dist;
    fy = (dy*force) / dist;
    fz = (dz*force) / dist;
}



inline float getRand(const float min, const float max)
{
    return ((max-min)*((float)rand() / (float)RAND_MAX)) + min;
}


CpuParticleEngine::CpuParticleEngine()
{
    
}



CpuParticleEngine::~CpuParticleEngine()
{
    
}


void CpuParticleEngine::initialize()
{
    srand(Parameters::seed);
    
    m_particles[0].m_x = -40.0;
    m_particles[0].m_y = 0.0;
    m_particles[0].m_vx = 0.0;
    m_particles[0].m_vy = -1000.0;
    m_particles[0].m_mass = 1.0e6;
    m_particles[0].m_type = 1;

    m_particles[1].m_x = 40.0;
    m_particles[1].m_y = 0.0;
    m_particles[1].m_vx = 0.0;
    m_particles[1].m_vy = 1000.0;
    m_particles[1].m_mass = 1.0e6;
    m_particles[1].m_type = 1;
    
    for(int i = 2; i < Parameters::num_particles; i++)
    {
        float angle = (i*2*M_PI) / Parameters::num_particles;
        
        m_particles[i].m_x = 10.0*std::cos(angle);
        m_particles[i].m_y = 10.0*std::sin(angle);
        m_particles[i].m_z = ((20.0*i) / (Parameters::num_particles - 1)) - 10.0;
        m_particles[i].m_vx = -100*std::sin(angle);
        m_particles[i].m_vy = 100*std::cos(angle);
        m_particles[i].m_vz = 0.0;
    }
}

void CpuParticleEngine::get_min_max(float &min_x,
                                    float &max_x,
                                    float &min_y,
                                    float &max_y)
{
    min_x = m_particles[0].m_x;
    max_x = m_particles[0].m_x;
    min_y = m_particles[0].m_y;
    max_y = m_particles[0].m_y;
    
    for(int i = 1; i < Parameters::num_particles; i++)
    {
        min_x = std::min(min_x, m_particles[i].m_x);
        max_x = std::max(max_x, m_particles[i].m_x);
        min_y = std::min(min_y, m_particles[i].m_y);
        max_y = std::max(max_y, m_particles[i].m_y);
    }
}

void CpuParticleEngine::runIteration(int cnt)
{
    float max_x = 100;
    float min_x = -100;
    float max_y = 100;
    float min_y = -100;

    get_min_max(min_x, max_x, min_y, max_y);

    if(cnt % 1 == 0)
    {
        for(int i = 0; i < (Parameters::width*Parameters::height); i++)
        {
            m_pixel_buf[i] = 0;
        }
    }
    
    for(int i = 0; i < 10; i++)
    {
        compute_forces();
        euler_update();
        draw_particles(min_x, max_x, min_y, max_y);

    }
    
}

void CpuParticleEngine::draw(unsigned int *pixbuf)
{
    const int num_pixels = Parameters::width*Parameters::height;
    for(int i = 0; i < num_pixels; i++)
    {
        pixbuf[i] = m_pixel_buf[i];
    }
}

float CpuParticleEngine::getTotalEnergy()
{
    float kinetic_energy = 0;
    for(int i = 0; i < Parameters::num_particles; i++)
    {
        kinetic_energy += m_particles[i].m_vx*m_particles[i].m_vx;
        kinetic_energy += m_particles[i].m_vy*m_particles[i].m_vy;
        kinetic_energy += m_particles[i].m_vz*m_particles[i].m_vz;
    }
    kinetic_energy *= 0.5;

    float potential_energy = 0;
    for(int i = 0; i < Parameters::num_particles; i++)
    {
        for(int j = i+1; j < Parameters::num_particles; j++)
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

std::vector<std::string> CpuParticleEngine::getParticleText()
{
    std::vector<std::string> particle_info;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << getTotalEnergy();
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

void CpuParticleEngine::compute_forces()
{
    for(unsigned i = 0; i < m_particles.size(); i++)
    {
        float damp = 1e-2;
        float fx = -damp*m_particles[i].m_vx;
        float fy = -damp*m_particles[i].m_vy;
        float fz = -damp*m_particles[i].m_vz;
        
        for(unsigned j = 0; j < m_particles.size(); j++)
        {
            if(i == j) continue;
            
            float dfx;
            float dfy;
            float dfz;
            
            compute1(m_particles[j], m_particles[i], dfx, dfy, dfz);
            
            fx += dfx;
            fy += dfy;
            fz += dfz;
        }
        
        m_particles[i].m_ax = fx / m_particles[i].m_mass;
        m_particles[i].m_ay = fy / m_particles[i].m_mass;
        m_particles[i].m_az = fz / m_particles[i].m_mass;
    }    

}

void CpuParticleEngine::euler_update()
{
    const float timestep = 1e-4;
    float max_norm = 0;
    for(unsigned i = 0; i < m_particles.size(); i++)
    {
        float ax = m_particles[i].m_ax;
        float ay = m_particles[i].m_ay;
        float az = m_particles[i].m_az;
        
        float vx = m_particles[i].m_vx;
        float vy = m_particles[i].m_vy;
        float vz = m_particles[i].m_vz;
        
        float norm = std::sqrt(ax*ax + ay*ay + az*az + vx*vx + vy*vy + vz*vz);
        
        max_norm = std::max(max_norm, norm);
    }
    
    float scalar = timestep; // / std::min(1e2f, max_norm);
    for(unsigned i = 0; i < m_particles.size(); i++)
    {        
        m_particles[i].m_vx += m_particles[i].m_ax*scalar;
        m_particles[i].m_vy += m_particles[i].m_ay*scalar;
        m_particles[i].m_vz += m_particles[i].m_az*scalar;

        m_particles[i].m_x += m_particles[i].m_vx*scalar;
        m_particles[i].m_y += m_particles[i].m_vy*scalar;
        m_particles[i].m_z += m_particles[i].m_vz*scalar;
    }

}

void CpuParticleEngine::draw_particles(const float min_x, const float max_x,
                                       const float min_y, const float max_y)
{

    const int width = Parameters::width;
    const int height = Parameters::height;
        
    for(unsigned i = 0; i < m_particles.size(); i++)
    {
        int x = (int)(width * (m_particles[i].m_x - min_x) / (max_x - min_x));
        int y = (int)(height * (m_particles[i].m_y - min_y) / (max_y - min_y));

        if((x >= 0) && (x < width) && (y >= 0) && (y < height))
        {
            float z_clamp_blue = fmaxf(-10.0, fminf(10.0, m_particles[i].m_z));
            unsigned z_blue = std::floor(0xFF * 0.9*((z_clamp_blue + 10.0) / 20.0) + 0.1);
            m_pixel_buf[(y*width) + x] = z_blue;
            //m_pixel_buf[(y*width) + x] |= (0xFF0000*m_particles[i].m_type);
        }
    }
}


