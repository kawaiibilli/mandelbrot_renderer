#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <mpfr.h>
#include <omp.h>
#include "color_palette.h"

// #include "kernel.h"
// #define N 1024

using namespace std;
using namespace cv;
double ln2_inv = 1.44269504088896340735992468100189213742664595415299;
double ln2 = 0.69314718055994530941723212145817656807550013436026;

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
void cpu_render(int *canvas, mpfr_t l_margin, mpfr_t d_margin, mpfr_t x_multiplier, mpfr_t y_multiplier, int dim_x, int dim_y, mpfr_t ITER_MAX)//, double l_margin_double, double r_margin_double, double u_margin_double, double d_margin_double)
{
	int idx = 0;
	double bailout = 128; // with a smaller value there are lines on magn=1
    double logLogBailout = log(log(bailout));

	#pragma omp parallel for schedule(dynamic) num_threads(32) private(idx)
	for(idx=0;idx<dim_x*dim_y;idx++)
	{
		
		mpfr_t rere, imim, z_n_x, z_n_y, c_x, c_y, x_pos, y_pos, tmp_x, tmp_y, modsq, t1;
		mpfr_init2(z_n_x, 150);
		mpfr_init2(z_n_y, 150);
		mpfr_init2(rere, 150);
		mpfr_init2(imim, 150);
		mpfr_init2(c_x, 150);
		mpfr_init2(c_y, 150);
		mpfr_init2(x_pos, 150);
		mpfr_init2(y_pos, 150);
		mpfr_init2(tmp_x, 150);
		mpfr_init2(tmp_y, 150);
		mpfr_init2(modsq, 150);
		mpfr_init2(t1, 150);


		bool bailed = false;
		double hx, hy, d, zxd, zyd;
		double eps = 1e-17;
		int pos_x = idx%dim_x;
		int pos_y = idx/dim_x;
		int check = 3;
		int whenupdate = 10;


		unsigned long escape_time = 0;

		mpfr_mul_ui(c_x, x_multiplier, pos_x, MPFR_RNDN);
		mpfr_add(c_x, c_x, l_margin, MPFR_RNDN);

		mpfr_mul_ui(c_y, y_multiplier, pos_y, MPFR_RNDN);
		mpfr_add(c_y, c_y, d_margin, MPFR_RNDN);

		// printf("idx : %d\n",idx);
		// mpfr_printf("c_x : %.50Rf, c_y : %.50Rf \n", c_x, c_y);

		mpfr_set(z_n_x, c_x, MPFR_RNDN);
		mpfr_set(z_n_y, c_y, MPFR_RNDN);


		canvas[idx] = 0;

		while(mpfr_cmp_ui(ITER_MAX, escape_time)>0) //z_n_x_double*z_n_x_double + z_n_y_double*z_n_y_double<4.0)
		{
			mpfr_mul(rere, z_n_x, z_n_x, MPFR_RNDN);
			mpfr_mul(imim, z_n_y, z_n_y, MPFR_RNDN);
			mpfr_add(modsq, rere, imim, MPFR_RNDN);

			if(mpfr_cmp_ld(modsq, bailout)>0)
			{
				// mpfr_printf("iteration no : %d, z_n_x : %.100Rf, z_n_y : %.100Rf \n", escape_time, z_n_x, z_n_y);
				
				bailed = true;
				break;
			}

			mpfr_mul(z_n_y, z_n_y, z_n_x, MPFR_RNDN);
			mpfr_mul_ui(z_n_y, z_n_y, 2, MPFR_RNDN);
			mpfr_add(z_n_y, z_n_y, c_y, MPFR_RNDN);


			mpfr_sub(z_n_x, rere, imim, MPFR_RNDN);
			mpfr_add(z_n_x, z_n_x, c_x, MPFR_RNDN);
			
			escape_time ++;

			// period checking
			zxd = mpfr_get_d(z_n_x, MPFR_RNDN);

			d = zxd - hx;
			if (d > 0.0 ? d < eps : d > -eps)
			{
				zyd = mpfr_get_d(z_n_y, MPFR_RNDN);
				d = zyd - hy;
				if (d > 0.0 ? d < eps : d > -eps)
				{
					break;
				}
			}
			if ((escape_time % check) == 0)
			{
				if (--whenupdate == 0)
				{
					whenupdate = 10;
					check <<= 1;
					check++;
				}
				// period = 0;
				zxd = mpfr_get_d(z_n_x, MPFR_RNDN);
				zyd = mpfr_get_d(z_n_y, MPFR_RNDN);
				hx = zxd;
				hy = zyd;
            }
		}
		// 	escape_time ++;
		// }
		if(!bailed)
		{
			canvas[idx] = -1.0;
		}
		else
		{
			// zxd = mpfr_get_d(z_n_x, MPFR_RNDN);
			// zyd = mpfr_get_d(z_n_y, MPFR_RNDN);

			// double r = sqrt(zxd*zxd + zyd*zyd);
			// printf("iter_end radius : %lf\n",r);
			// double c = escape_time - 1.28 + (logLogBailout - log(log(r))) * ln2_inv;
			// printf("c : %lf\n",c);
			// int color_idx = fmod((log(c/64+1)/ln2+0.45), 1)*GRADIENTLENGTH + 0.5;

			canvas[idx] = escape_time;

		}

		mpfr_clear(z_n_x);
		mpfr_clear(z_n_y);
		mpfr_clear(c_x);
		mpfr_clear(c_y);
		mpfr_clear(x_pos);
		mpfr_clear(y_pos);
		mpfr_clear(tmp_x);
		mpfr_clear(tmp_y);
		mpfr_clear(modsq);
		mpfr_clear(t1);
		mpfr_clear(rere);
		mpfr_clear(imim);

	}
	// mpfr_clear(z_n_x);
	// mpfr_clear(z_n_y);
	// mpfr_clear(c_x);
	// mpfr_clear(c_y);
	// mpfr_clear(x_pos);
	// mpfr_clear(y_pos);
	// mpfr_clear(tmp_x);
	// mpfr_clear(tmp_y);
	// mpfr_clear(modsq);
	// mpfr_clear(t1);

}
int main()
{

	// double control_points[] = {0.0, 0.16, 0.42, 0.6425, 0.8575};
	// vector <Vec3b> color_points;
	// color_points.push_back(Vec3b(0, 7, 100));
	// color_points.push_back(Vec3b(32, 107, 203));
	// color_points.push_back(Vec3b(237, 255, 255));
	// color_points.push_back(Vec3b(255, 170, 0));
	// color_points.push_back(Vec3b(0, 2, 0));
	// Vec3b color;

	int dim_x = 2048;
	int dim_y = 2048;
	int median_kernel_size = 2;
	int color_modulo = 1000;
	int window[median_kernel_size*median_kernel_size][3];
	Mat img_mat(dim_y, dim_x, CV_8UC3, Scalar(0,0,255));
	Mat out_img(dim_y/median_kernel_size, dim_x/median_kernel_size, CV_8UC3, Scalar(0,0,255));
	Mat ycrcb;

	mpfr_t ITER_MAX,temp1, len, half_len, l_margin, d_margin, x_multiplier, y_multiplier, center_x, center_y, zoom, zoom_limit, sqrt_zoom;

	mpfr_init2(len, 150);
	mpfr_init2(half_len, 150);
	mpfr_init2(l_margin, 150);
	mpfr_init2(x_multiplier, 150);
	mpfr_init2(d_margin, 150);
	mpfr_init2(y_multiplier, 150);
	mpfr_init2(center_x, 150);
	mpfr_init2(center_y, 150);
	mpfr_init2(zoom, 150);
	mpfr_init2(zoom_limit, 150);
	mpfr_init2(sqrt_zoom, 150);
	mpfr_init2(temp1, 150);
	mpfr_init2(ITER_MAX, 150);

	mpfr_set_str(center_x, "-0.780536168332684210920007759750290234283493214962876942376096420835223334508289524505138869427047843265409475532745631198849878273904323577880859375000", 10, MPFR_RNDN);
							
	mpfr_set_str(center_y, "0.133678216474162526562066770268974360465938997976421554155379838098560710705073590148016919520451382981633957651099176011655345064355060458183288574219", 10, MPFR_RNDN);
	mpfr_set_str(zoom, "1", 10, MPFR_RNDN);
	mpfr_set_str(zoom_limit, "1000000000000000000000000000000000", 10, MPFR_RNDN);
	mpfr_set_d(ITER_MAX, 1000, MPFR_RNDN);
	// mpfr_set_d(init_len, 4.0, MPFR_RNDN);

	vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(1);

	size_t canvas_size =  dim_x*dim_y*sizeof(int);
	int it=0;

	int *h_canvas = NULL;
	double zoom_multiplier = 1.1;
	int init_len = 4;
	h_canvas = (int *)malloc(canvas_size);

	std::string filename = "frames/mandelbrot_cpu";
	for (it=0; mpfr_cmp(zoom, zoom_limit)<0; it++)
	{


		mpfr_ui_div(len, init_len, zoom, MPFR_RNDN);

		mpfr_si_div(l_margin, -2, zoom, MPFR_RNDN);
    	mpfr_add(l_margin, l_margin, center_x, MPFR_RNDN);
		
    	mpfr_ui_div(d_margin, 2, zoom, MPFR_RNDN);
    	mpfr_set_ui(temp1, dim_y, MPFR_RNDN);
    	mpfr_div_ui(temp1, temp1, dim_x, MPFR_RNDN);
    	mpfr_mul(d_margin, d_margin, temp1, MPFR_RNDN);
    	mpfr_add(d_margin, d_margin, center_y, MPFR_RNDN);
		
		mpfr_ui_div(x_multiplier, 4, zoom, MPFR_RNDN);
    	mpfr_div_ui(x_multiplier, x_multiplier, dim_x, MPFR_RNDN);
		mpfr_si_div(y_multiplier, -4, zoom, MPFR_RNDN);
		mpfr_div_ui(y_multiplier, y_multiplier, dim_x, MPFR_RNDN);

		mpfr_sqrt(sqrt_zoom, zoom, MPFR_RNDN);
		mpfr_mul_ui(ITER_MAX, sqrt_zoom, dim_x/(8*median_kernel_size), MPFR_RNDN);
		color_modulo = (int)mpfr_get_d(ITER_MAX, MPFR_RNDN);
		
		if(mpfr_cmp_ui(ITER_MAX, 100000)>0 && mpfr_cmp_ui(ITER_MAX, 10000000)<=0)
		{
			mpfr_set_ui(ITER_MAX, 10000, MPFR_RNDN);
			color_modulo = 10000;	
		}
		else if(mpfr_cmp_ui(ITER_MAX, 10000000)>0 && mpfr_cmp_ui(ITER_MAX, 1000000000)<=0)
		{
			mpfr_set_ui(ITER_MAX, 20000, MPFR_RNDN);
			color_modulo = 20000;
		}
		else if(mpfr_cmp_ui(ITER_MAX, 1000000000)>0)
		{
			mpfr_set_ui(ITER_MAX, 25000, MPFR_RNDN);
			color_modulo = 25000;
		}


		if(it>520)
		{
			mpfr_set_ui(ITER_MAX, 30000, MPFR_RNDN);
			color_modulo = 25000;
		}
		else if(it>650)
		{
			mpfr_set_ui(ITER_MAX, 40000, MPFR_RNDN);
			color_modulo = 40000;
		}


		if(it==400)
		{
			zoom_multiplier=1.12;
		}
		if(it==415)
		{
			zoom_multiplier=1.15;
		}
		if(it==470)
		{
			zoom_multiplier=1.2;
		}


		printf("iteration : %d\n",it);
		mpfr_printf("zoom=%.50Rf \n",zoom);
		mpfr_printf("ITER_MAX=%.50Rf \n",ITER_MAX);

		// render(h_canvas, center_x, center_y, len, dim_x, dim_y);
		// cpu_render(h_canvas, l_margin_big, r_margin_big, u_margin_big, d_margin_big, dim_x, dim_y, l_margin, r_margin, u_margin, d_margin);
		
		mpfr_mul_d(zoom, zoom, zoom_multiplier, MPFR_RNDN);
		// if(it<480)
		// {
		// 	continue;
		// }
		cpu_render(h_canvas, l_margin, d_margin, x_multiplier, y_multiplier, dim_x, dim_y, ITER_MAX);//, l_margin_double, r_margin_double, u_margin_double, d_margin_double);

		// int max_val = 0;
		// int min_val = INT_MAX;
		// for(int i=0;i<dim_y;i++)
		// {
		// 	for(int j=0;j<dim_x;j++)
		// 	{
		// 		max_val = max(max_val, h_canvas[i*dim_x + j]);
		// 		min_val = min(min_val, h_canvas[i*dim_x + j]);
		// 	}
		// }

		// printf("max_val :%d\n", max_val);
		// printf("min_val :%d\n", min_val);

		
		string it_s = to_string(it);		

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
		// cvtColor(img_mat , ycrcb, CV_BGR2YCrCb);
		// vector<Mat> channels;
  //       split(ycrcb,channels);
  //       equalizeHist(channels[0], channels[0]);
        // merge(channels,ycrcb);
        // cvtColor(ycrcb, img_mat, CV_YCrCb2BGR);
	
	}

	mpfr_clear(half_len);
	mpfr_clear(l_margin);
	mpfr_clear(d_margin);
	mpfr_clear(center_x);
	mpfr_clear(center_y);
	mpfr_clear(x_multiplier);
	mpfr_clear(y_multiplier);
	mpfr_clear(zoom);
	mpfr_clear(zoom_limit);
	mpfr_clear(temp1);
	mpfr_free_cache();
}