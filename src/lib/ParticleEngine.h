#pragma once

#include "Parameters.h"
#include "Particle.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>

class ParticleEngine
{
public:
    ParticleEngine();
    ~ParticleEngine();

    void initialize();
    void runIteration();
    void draw(unsigned int *pixbuf);
    std::vector<std::string> getParticleText();
    float getTotalEnergy();

    void cpu_get_min_max(float &min_x, float &max_x, float &min_y, float &max_y);
    void cpu_compute_forces(Particle *particles, int num_particles);
    void cpu_euler_update(Particle *particles, int num_particles);
    void cpu_draw_particles(const Particle *particles, const int num_particles,
                            const float min_x, const float min_y,
                            const float max_x, const float max_y,
                            unsigned int *pixelbuf, const int width, const int height);

private:
    static constexpr int m_num_particles = 5;
    
    unsigned int *m_cuda_pixel_buf;
    Particle *m_particles;
    
    
};

