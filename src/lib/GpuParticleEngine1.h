#pragma once

#include "Parameters.h"
#include "Particle.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <array>
#include <vector>

class GpuParticleEngine1
{
public:
    GpuParticleEngine1();
    ~GpuParticleEngine1();

    std::array<Particle, Parameters::num_particles> getParticles()
    {
        std::array<Particle, Parameters::num_particles> particles;
        for(unsigned i = 0; i < particles.size(); i++)
        {
            particles[i] = m_particles[i];
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
    void initialize1();
    void initialize2();

    void clearPixelBuf(int cnt);
    void runIteration();
    void draw(unsigned int *pixbuf);
    std::vector<std::string> getParticleText();
    float getTotalEnergy();

    // Wrappers around kernels for testing
    void get_min_max(float &min_x, float &max_x,
                     float &min_y, float &max_y);
    void compute_forces();
    void euler_update();
    void draw_particles(const float min_x, const float max_x,
                        const float min_y, const float max_y);

    
private:
    unsigned int *m_cuda_pixel_buf;
    Particle *m_particles;

    float *m_gpu_min_x;
    float *m_gpu_max_x;
    float *m_gpu_min_y;
    float *m_gpu_max_y;
};

