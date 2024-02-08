#pragma once

#include "Parameters.h"
#include "Particle.h"

#include <string>
#include <vector>
#include <array>

class CpuParticleEngine
{
public:
    CpuParticleEngine();
    ~CpuParticleEngine();

    std::array<unsigned int, Parameters::width*Parameters::height> getPixelBuf() { return m_pixel_buf; }
    std::array<Particle, Parameters::num_particles> getParticles() { return m_particles; }

    void initialize();
    void runIteration(int cnt);
    void draw(unsigned int *pixbuf);
    std::vector<std::string> getParticleText();
    float getTotalEnergy();

    void get_min_max(float &min_x, float &max_x, float &min_y, float &max_y);
    void compute_forces(std::array<Particle, Parameters::num_particles> &particles);
    void euler_update(std::array<Particle, Parameters::num_particles> &particles);
    
    void draw_particles(const std::array<Particle, Parameters::num_particles> &particles,
                        const float min_x, const float min_y,
                        const float max_x, const float max_y,
                        std::array<unsigned int, Parameters::width*Parameters::height> &pixelbuf,
                        const int width, const int height);

private:
    std::array<unsigned int, Parameters::width*Parameters::height> m_pixel_buf;
    std::array<Particle, Parameters::num_particles> m_particles;
    
    
};

