#pragma once

#include "Parameters.h"
#include "Particle.h"
#include "ParticleEngine.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>

#include <string>
#include <vector>

class MainWindow
{
public:
    MainWindow();
    ~MainWindow();

    void run(int cnt);
    void loop();

    void drawParticleText(const std::vector<std::string> &particle_text);
    std::pair<int,int> renderText(int x, int y, const std::string &text);
    
private:
    bool m_ok{false};
    
    TTF_Font *m_font = NULL;
    
    SDL_Window *m_window = NULL;
    SDL_Renderer *m_renderer = NULL;
    SDL_Surface *m_window_surface = NULL;
    
    ParticleEngine m_engine;

    static constexpr int m_particle_text_x{10};
    static constexpr int m_particle_text_y{10};
};
