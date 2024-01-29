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
    
    //SDL_SetWindowFullscreen(m_window, SDL_WINDOW_FULLSCREEN)
    
    m_window_surface = SDL_GetWindowSurface(m_window);
    
    std::cout << "Pixel Format: " << SDL_GetPixelFormatName(m_window_surface->format->format);
    
    int count;
    int device;
    cudaGetDeviceCount(&count);
    cudaGetDevice(&device);
    
    std::cout << "Number of Devices: " << count << "\n";
    std::cout << "My device is: " << device << "\n";
    
    m_ok = true;

    m_engine.initialize();
}


MainWindow::~MainWindow()
{
    SDL_DestroyWindow(m_window);
    SDL_Quit();
}

void MainWindow::run()
{
    m_engine.runIteration();
    m_engine.draw((unsigned char *) m_window_surface->pixels);

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
    while(m_ok && i < 10000)
    {
        run();
        i++;
    }
}
