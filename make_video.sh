#!/bin/bash
ffmpeg -framerate 20 -start_number 0 -i frames/mandelbrot%d.png -pix_fmt yuv420p mandelbrot_zoom.mp4
