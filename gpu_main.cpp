#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "kernel.h"
#include "color_palette.h"

using namespace std;
using namespace cv;
cv::Vec3b get_median(int median_kernel_size, int window[][3])
{
    int i, j;
    int temp[3];
    int window_size = median_kernel_size * median_kernel_size;
    for(i = 0; i < window_size; i++)
    {
        temp[0] = window[i][0];
        temp[1] = window[i][1];
        temp[2] = window[i][2];
        for(j = i-1; j >= 0 && ((int)temp[0] + (int)temp[1] + (int)temp[2])/3 < ((int)window[j][0] + (int)window[j][1] + (int)window[j][2]/3); j--)
        {
            window[j+1][0] = window[j][0];
            window[j+1][1] = window[j][1];
            window[j+1][2] = window[j][2];
        }
        window[j+1][0] = temp[0];
        window[j+1][1] = temp[1];
        window[j+1][2] = temp[2];
    }

    
    if (window_size%2==0)
    {
    	cv::Vec3b ret; // = Vec3b((int)window[window_size/2][0],(int)window[window_size/2][1], (int)window[window_size/2][2]);
    	// // ret[0] = window[window_size/2][0];// + window[(window_size/2)-1])/2;
    	// // ret[1] = window[window_size/2][1];
    	// // ret[2] = window[window_size/2][2];
    	// // cout<<"window[0] : "<<window[window_size/2].val[0]<<endl;
    	ret.val[0] = (window[window_size/2][0] + window[(window_size/2)-1][0])/2;
    	ret.val[1] = (window[window_size/2][1] + window[(window_size/2)-1][1])/2;
    	ret.val[2] = (window[window_size/2][2] + window[(window_size/2)-1][2])/2;
    	return ret;
    }
    else
    {
    	// ret[0] = window[window_size/2][0];
    	// ret[1] = window[window_size/2][1];
    	// ret[2] = window[window_size/2][2];
    	cv::Vec3b ret = Vec3b(window[window_size/2][0], window[window_size/2][1], window[window_size/2][2]);
    	return ret;
    }
    
}	
int main()
{
	
	double center_x = -0.780536168332684210920007759750290234283493214962876942376096420835223334508289524505138869427047843265409475532745631198849878273904323577880859375000;
	double center_y = 0.133678216474162526562066770268974360465938997976421554155379838098560710705073590148016919520451382981633957651099176011655345064355060458183288574219;
	// double center_x = -1.41865900567399774220273850602674973716069151130625414225505664944648742675781250; // (-1.4186590056740714542390 + -1.4186590056739240302590)/2
	// double center_y = 0.00095101076821275047974399021664739569839085229878417049320660225930623710155487; // (0.0009510107681574665210 + 0.0009510107682680345062)/2
	printf("Centre : %lf + i %lf \n",center_x,center_y);
	fflush(stdout);
	int dim_x = 2048;
	int dim_y = 2048;
	int median_kernel_size = 2;
	int window[median_kernel_size*median_kernel_size][3];
	Mat img_mat(dim_y, dim_x, CV_8UC3, Scalar(0,0,255));
	Mat out_img(dim_y/median_kernel_size, dim_x/median_kernel_size, CV_8UC3, Scalar(0,0,255));
	size_t canvas_size =  dim_x*dim_y*sizeof(int);
	// if(dim_x%16!=0 || dim_y%16!=0)
	// {	

	// 	canvas_size = (16-(dim_x%16) + dim_x)*(16-(dim_y%16) + dim_y)*sizeof(int); 
	// }
	int it, init_len;
	double zoom;

	int *h_canvas = NULL;
	std::string filename = "frames/mandelbrot_gpu";
	h_canvas = (int *)malloc(canvas_size);
	double N=128;

	vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(1);
    int color_modulo=1000;
    // cv::MatIterator_<cv::Vec3b> it;
	for (it=0, zoom=1.5, init_len = 10.0; (zoom<=pow(10,40) && it<1000); zoom *= 1.09, it++)
	{
		N = sqrt(zoom);
		// N = N>1000?1000000:N*dim_y;
		if(N<=10)
		{
			N=250;
			color_modulo = 250;	
		}
		else if(N>10 && N<=100)
		{
			N=1000;
			color_modulo = 1000;	
		}
		else if(N>100 && N<=200)
		{
			N=2000;
			color_modulo = 1000;
		}
		else if(N>200 && N<=300)
		{
			N=3000;
			color_modulo = 2000;
		}
		else if(N>300 && N<=500)
		{
			N=4000;
			color_modulo = 3000;
		}
		else if(N>500 && N<=1000)
		{
			N=5000;
			color_modulo = 4000;
		}
		else if(N>1000)
		{
			N = 7500;
			color_modulo = 7500;
		}


		printf("it : %d, zoom : %lf, N: %lf\n", it, zoom, N);
		fflush(stdout);

		string it_s = to_string(it);

		render(h_canvas, zoom, center_x, center_y, init_len/zoom, dim_x, dim_y, N);

		// printf("Writing the rendered image\n");
		// fflush(stdout);
		// FILE* img = fopen((filename + it_s + string(".pgm")).c_str(), "wb");
		// (void )(void) fprintf(img, "P6\n%d %d\n255\n", dim_x, dim_y);
		// printf("Writing RGB values ...\n");
		// fflush(stdout);

		for(int i=0;i<dim_y;i++)
		{
			for(int j=0;j<dim_x;j++)
			{
				// unsigned char tmp[3];
				int color_idx = h_canvas[i*dim_x + j];

				if(color_idx>=0)
				{
					color_idx = color_idx % color_modulo;
					// Reading the colours in reverse because opencv has BGR format of coloring
					img_mat.at<cv::Vec3b>(i, j) = Vec3b(colors[color_idx][2], colors[color_idx][1], colors[color_idx][0]);
				}
				else
				{
					img_mat.at<cv::Vec3b>(i, j) = Vec3b(0, 0, 0);
				}
			}
		}
		//assumption that rows and cols are divisible by kernel size
		

		
		// Vec3b aat = img_mat.at<Vec3b>(23, 53);
  //      	cout<<"aat[0]: "<< (int)img_mat.at<Vec3b>(23, 53)[0] << endl;

		imwrite( filename + it_s + "_without_filter.png", img_mat, compression_params);


		for(int y = 0; y < img_mat.rows - median_kernel_size; y+=median_kernel_size)
		{
            for(int x = 0; x < img_mat.cols - median_kernel_size; x+=median_kernel_size)
            {
                // Pick up window element
                for(int y_it =0; y_it<median_kernel_size; y_it++ )
                {
                	for(int x_it=0; x_it<median_kernel_size; x_it++)
                	{
                		window[y_it*median_kernel_size + x_it][0] = (int)img_mat.at<cv::Vec3b>(y + y_it, x + x_it)[0];
                		window[y_it*median_kernel_size + x_it][1] = (int)img_mat.at<cv::Vec3b>(y + y_it, x + x_it)[1];
                		window[y_it*median_kernel_size + x_it][2] = (int)img_mat.at<cv::Vec3b>(y + y_it, x + x_it)[2];
                	}
                }
                // sort the window to find median
                cv::Vec3b median = get_median(median_kernel_size, window);

				out_img.at<cv::Vec3b>(y/median_kernel_size, x/median_kernel_size)[0] = median[0];
				out_img.at<cv::Vec3b>(y/median_kernel_size, x/median_kernel_size)[1] = median[1];
				out_img.at<cv::Vec3b>(y/median_kernel_size, x/median_kernel_size)[2] = median[2];
            }
        }

		// imwrite( filename + it_s + "_without_filter2.png", img_mat, compression_params);
		// medianBlur (img_mat, out_img, 3);
		// resize(out_img, out_img, Size(1024, 1024));
		imwrite( filename + it_s + ".png", out_img, compression_params);

	}
	free(h_canvas);
}