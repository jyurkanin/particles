#include <CpuParticleEngine.h>
#include <GpuParticleEngine1.h>
#include <Particle.h>

#include <array>
#include "gtest/gtest.h"

TEST(ParticleEngineTest, initialize)
{
    CpuParticleEngine  cpu_engine;
    GpuParticleEngine1 gpu_engine1;
    
    cpu_engine.initialize();
    gpu_engine1.initialize();
    
    std::array<Particle, Parameters::num_particles> cpu_particles = cpu_engine.getParticles();
    std::array<Particle, Parameters::num_particles> gpu_particles1 = gpu_engine1.getParticles();

    for(int i = 0; i < cpu_particles.size(); i++)
    {
        EXPECT_EQ(cpu_particles[i].m_x, gpu_particles1[i].m_x);
        EXPECT_EQ(cpu_particles[i].m_y, gpu_particles1[i].m_y);
        EXPECT_EQ(cpu_particles[i].m_vx, gpu_particles1[i].m_vx);
        EXPECT_EQ(cpu_particles[i].m_vy, gpu_particles1[i].m_vy);
    }
}

TEST(ParticleEngineTest, min_max)
{
    CpuParticleEngine  cpu_engine;
    GpuParticleEngine1 gpu_engine1;
    
    cpu_engine.initialize();
    gpu_engine1.initialize();

    float cpu_min_x;
    float cpu_max_x;
    float cpu_min_y;
    float cpu_max_y;
    cpu_engine.get_min_max(cpu_min_x, cpu_max_x,
                           cpu_min_y, cpu_max_y);
    
    float gpu_min_x;
    float gpu_max_x;
    float gpu_min_y;
    float gpu_max_y;
    gpu_engine1.get_min_max(gpu_min_x, gpu_max_x,
                            gpu_min_y, gpu_max_y);
    
    EXPECT_EQ(cpu_min_x, gpu_min_x);
    EXPECT_EQ(cpu_max_x, gpu_max_x);
    EXPECT_EQ(cpu_min_y, gpu_min_y);
    EXPECT_EQ(cpu_max_y, gpu_max_y);
}
