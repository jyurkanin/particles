#include "MainWindow.h"
#include "Parameters.h"

#include <iostream>
#include <unistd.h>



MainWindow::MainWindow()
{
    if(SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cout << "SDL could not initialize! SDL_Error: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return;
    }
    
    SDL_DisplayMode dm;
    if(SDL_GetDesktopDisplayMode(0, &dm) != 0)
    {
        std::cout << "SDL_GetDesktopDIsplayMode failed: " << SDL_GetError() << "\n";
        return;
    }
    
    m_window = SDL_CreateWindow("Cool Particle Stuff", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, Parameters::m_width, Parameters::m_height, SDL_WINDOW_SHOWN);
    if(m_window == NULL)
    {
        std::cout << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return;
    }

    if(TTF_Init() == -1)
    {
        std::cout << "SDL_ttf could not initialize: " << TTF_GetError() << "\n";
    }
    
    //SDL_SetWindowFullscreen(m_window, SDL_WINDOW_FULLSCREEN)
    m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_SOFTWARE);
    m_window_surface = SDL_GetWindowSurface(m_window);
    m_font = TTF_OpenFont("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 18);

    if(m_font == NULL)
    {
        std::cout << "Failed to load: " << TTF_GetError() << "\n";
    }
    
    int count;
    int device;
    cudaGetDeviceCount(&count);
    cudaGetDevice(&device);

    std::cout << "Pixel Format: " << SDL_GetPixelFormatName(m_window_surface->format->format);
    std::cout << "Number of Devices: " << count << "\n";
    std::cout << "My device is: " << device << "\n";
    
    m_ok = true;

    m_engine.initialize();
}


MainWindow::~MainWindow()
{
    TTF_CloseFont(m_font);
    SDL_DestroyWindow(m_window);
    SDL_Quit();
}



void MainWindow::drawParticleText(const std::vector<std::string> &particle_text)
{
    int x = m_particle_text_x;
    int y = m_particle_text_y;
    
    for(int i = 0; i < particle_text.size(); i++)
    {
        std::pair<int,int> width_height = renderText(x, y, particle_text[i]);
        y += width_height.second;
    }
}

std::pair<int,int> MainWindow::renderText(int x, int y, const std::string &text)
{
    SDL_Color text_color;
    text_color.r = 0xFF;
    
    SDL_Surface* text_surface = TTF_RenderText_Solid( m_font, text.c_str(), text_color);
    SDL_Texture* texture = SDL_CreateTextureFromSurface(m_renderer, text_surface);

    int width;
    int height;
    SDL_QueryTexture(texture, NULL, NULL, &width, &height);
    
    SDL_Rect dst_rect{x, y, width, height};
    SDL_Rect src_rect{0, 0, width, height};
    
    SDL_RenderCopy(m_renderer, texture, &src_rect, &dst_rect);
    SDL_FreeSurface(text_surface);
    SDL_DestroyTexture(texture);

    return std::pair<int,int>(width, height);
}

void MainWindow::run(int cnt)
{
    m_engine.runIteration(cnt);
    m_engine.draw((unsigned int *) m_window_surface->pixels);
    
    // drawParticleText(m_engine.getParticleText());
    
    SDL_RenderPresent(m_renderer);
    
    //handle pending events
    SDL_Event event;
    while(SDL_PollEvent(&event))
    {
        if(event.type == SDL_KEYDOWN)
        {
            switch(event.key.keysym.sym)
            {
            case SDLK_ESCAPE:
                m_ok = false;
                break;
                
            default:
                break;
            }
        }
        else if(event.type == SDL_QUIT)
        {
            m_ok = false;
        }
        else if(event.type == SDL_WINDOWEVENT)
        {
            if(event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
            {
                // m_window_surface = SDL_GetWindowSurface(window);
                // m_width = m_window_surface->w;
                // m_height = m_window_surface->h;
            }
        }
        
    }

    SDL_UpdateWindowSurface(m_window);

    usleep(10000);
}

void MainWindow::loop()
{
    int i = 0;
    while(m_ok && i < 10000000)
    {
        run(i);
        i++;
    }
}
