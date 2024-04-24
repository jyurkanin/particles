#pragma once

struct Parameters
{
    static constexpr int width{640};
    static constexpr int height{640};
    static constexpr int num_particles{static_cast<int>(1e4)};
    static constexpr int seed{420};
    
    static constexpr unsigned num_iterations{100};
    static constexpr unsigned num_blocks{1024};
    static constexpr unsigned blocksize{256};
    
};
