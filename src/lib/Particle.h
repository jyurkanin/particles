#pragma once


class Particle
{
public:
    Particle();
    ~Particle();
    
    float m_x{0.0};
    float m_y{0.0};
    float m_z{0.0};
    
    float m_vx{0.0};
    float m_vy{0.0};
    float m_vz{0.0};
    
    float m_mass{1.0};
};
