#include "gui/MainWindow.h"

#include <fenv.h>

int main(int argc, char* argv[])
{
    feenableexcept(FE_INVALID | FE_OVERFLOW);
    
    
    MainWindow window;

    window.loop();
    
    return 1;
}


