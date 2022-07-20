## Mandelbrot renderer in CUDA and OpenMP

My implementation of the escape time algorithm for zooming into the mandelbrot set. Anti-aliasing is done by supersampling using median filter, by default it dumps the original rendered image as well as the anti-aliased image. The the script uses ffmpeg to convert them to a video. MPFR is used in the cpu implementation for arbitrary precision.

![Mandelbrot Set Zooming](mandelbrot_zoom.gif)

## Usage
For making the video
~~~~
mkdir frames
make gpu_render # or make cpu_render
make gpu_run  # or make cpu_run
./make_video.sh
~~~~
To generate the colour palette, make necessary changes in the gen_colour_palette.py and then run 
~~~~
python gen_colour_palette.py
~~~~
It generates the colour_palette.h file used by the cuda/c++ code.

## Machine Details

Tested on a machine with the following configuration:
- Ubuntu 18.04
- Cuda Driver Version 10.2
- GeForce GTX 1660 Ti - 6 gb

## References
1. https://www.kth.se/social/files/5504b42ff276543e4aa5f5a1/An_introduction_to_the_Mandelbrot_Set.pdf
2. http://linas.org/art-gallery/escape/escape.html