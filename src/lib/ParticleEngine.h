#pragma once

#include "Parameters.h"
#include "Particle.h"

#include <cuda_runtime.h>

class ParticleEngine
{
public:
    ParticleEngine();
    ~ParticleEngine();

    void initialize();
    void runIteration();
    void draw(unsigned char *pixbuf);
    void cpu_compute_forces(Particle *particles, int num_particles);
    void cpu_euler_update(Particle *particles, int num_particles);
    void cpu_draw_particles(const Particle *particles, const int num_particles,
                            const float min_x, const float min_y,
                            const float max_x, const float max_y,
                            unsigned char *pixelbuf, const int width, const int height);

private:
    static constexpr int m_num_particles = 4;
    
    unsigned char *m_cuda_pixel_buf;
    Particle *m_particles;
    
    
};
