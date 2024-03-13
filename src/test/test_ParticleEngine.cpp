#include <CpuParticleEngine.h>
#include <GpuParticleEngine1.h>
#include <Particle.h>

#include <array>
#include "gtest/gtest.h"

TEST(ParticleEngineTest, initialize)
{
    CpuParticleEngine  cpu_engine;
    GpuParticleEngine1 gpu_engine;
    
    cpu_engine.initialize();
    gpu_engine.initialize();
    
    std::array<Particle, Parameters::num_particles> cpu_particles = cpu_engine.getParticles();
    std::array<Particle, Parameters::num_particles> gpu_particles = gpu_engine.getParticles();

    for(int i = 0; i < Parameters::num_particles; i++)
    {
        EXPECT_EQ(cpu_particles[i].m_x, gpu_particles[i].m_x);
        EXPECT_EQ(cpu_particles[i].m_y, gpu_particles[i].m_y);
        EXPECT_EQ(cpu_particles[i].m_vx, gpu_particles[i].m_vx);
        EXPECT_EQ(cpu_particles[i].m_vy, gpu_particles[i].m_vy);
    }
}

TEST(ParticleEngineTest, min_max)
{
    CpuParticleEngine  cpu_engine;
    GpuParticleEngine1 gpu_engine;
    
    cpu_engine.initialize();
    gpu_engine.initialize();

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
    gpu_engine.get_min_max(gpu_min_x, gpu_max_x,
                            gpu_min_y, gpu_max_y);
    
    EXPECT_EQ(cpu_min_x, gpu_min_x);
    EXPECT_EQ(cpu_max_x, gpu_max_x);
    EXPECT_EQ(cpu_min_y, gpu_min_y);
    EXPECT_EQ(cpu_max_y, gpu_max_y);
}

TEST(ParticleEngineTest, compute_forces)
{
    CpuParticleEngine cpu_engine;
    GpuParticleEngine1 gpu_engine;
    
    cpu_engine.initialize();
    gpu_engine.initialize();

    cpu_engine.compute_forces();
    gpu_engine.compute_forces();

    std::array<Particle, Parameters::num_particles> cpu_particles = cpu_engine.getParticles();
    std::array<Particle, Parameters::num_particles> gpu_particles = gpu_engine.getParticles();

    const float eps = 1e-1;
    for(unsigned i = 0; i < Parameters::num_particles; i++)
    {
        EXPECT_NEAR(cpu_particles[i].m_ax, gpu_particles[i].m_ax, eps);
        EXPECT_NEAR(cpu_particles[i].m_ay, gpu_particles[i].m_ay, eps);
        EXPECT_NEAR(cpu_particles[i].m_az, gpu_particles[i].m_az, eps);
    }
    
}

TEST(ParticleEngineTest, euler_update)
{
    CpuParticleEngine cpu_engine;
    GpuParticleEngine1 gpu_engine;
    
    cpu_engine.initialize();
    gpu_engine.initialize();

    cpu_engine.compute_forces();
    gpu_engine.compute_forces();    
    
    cpu_engine.euler_update();
    gpu_engine.euler_update();

    std::array<Particle, Parameters::num_particles> cpu_particles = cpu_engine.getParticles();
    std::array<Particle, Parameters::num_particles> gpu_particles = gpu_engine.getParticles();

    const float eps = 1e-3;
    for(unsigned i = 0; i < Parameters::num_particles; i++)
    {
        EXPECT_NEAR(cpu_particles[i].m_x, gpu_particles[i].m_x, eps);
        EXPECT_NEAR(cpu_particles[i].m_y, gpu_particles[i].m_y, eps);
        EXPECT_NEAR(cpu_particles[i].m_z, gpu_particles[i].m_z, eps);
        
        EXPECT_NEAR(cpu_particles[i].m_vx, gpu_particles[i].m_vx, eps);
        EXPECT_NEAR(cpu_particles[i].m_vy, gpu_particles[i].m_vy, eps);
        EXPECT_NEAR(cpu_particles[i].m_vz, gpu_particles[i].m_vz, eps);
    }
    
}


TEST(ParticleEngineTest, draw_particles)
{
    CpuParticleEngine cpu_engine;
    GpuParticleEngine1 gpu_engine;
    
    cpu_engine.initialize();
    gpu_engine.initialize();
    
    float min_x = -10;
    float max_x = 10;
    float min_y = -10;
    float max_y = 10;
    
    cpu_engine.draw_particles(min_x, max_x,
                              min_y, max_y);
    gpu_engine.draw_particles(min_x, max_x,
                              min_y, max_y);

    constexpr unsigned num_pixels = Parameters::width * Parameters::height;
    std::array<unsigned int, num_pixels> cpu_pixel_buf;
    std::array<unsigned int, num_pixels> gpu_pixel_buf;
    
    cpu_pixel_buf = cpu_engine.getPixelBuf();
    gpu_pixel_buf = gpu_engine.getPixelBuf();

    for(unsigned i = 0; i < num_pixels; i++)
    {
        EXPECT_EQ(cpu_pixel_buf[i], gpu_pixel_buf[i]);
    }
}
