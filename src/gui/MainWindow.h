#pragma once

#include "Parameters.h"
#include "Particle.h"
#include "ParticleEngine.h"

#include <SDL2/SDL.h>

class MainWindow
{
public:
    MainWindow();
    ~MainWindow();

    void run();
    void loop();
    
private:
    bool m_ok{false};
    
    SDL_Window *m_window = NULL;
    SDL_Surface *m_window_surface = NULL;
    
    ParticleEngine m_engine;
};
