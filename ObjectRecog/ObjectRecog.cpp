// ObjectRecog.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define _CRT_SECURE_NO_WARNINGS

#include "opencv/cvaux.h"
#include "opencv/highgui.h"
#include "opencv/cxcore.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat src, src_gray, last, drawing;
int thresh = 180;
int max_thresh = 255;
const int deviate = 35, tsize = 25;
bool first;
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);
void lightAndShadow();
int findThresh(Mat img);
int Undistort2(CvCapture* lol);
bool calibCheck();

int _tmain(int argc, _TCHAR* argv[])
{
	CvCapture* hehe = cvCreateCameraCapture( 0 );

	//while(1)
	//{
	//	p_cam>>src;

	//	lightAndShadow();

	//	escCheck = cvWaitKey(20);
	//	if(escCheck == 27)
	//	{
	//		p_cam.release();
	//		cvDestroyAllWindows();
	//		break;
	//	}
	//}

	Undistort2(hehe);

	return 0;
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY_INV );
  /// Find contours
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );

  for( int i = 0; i < contours.size(); i++ )
     { 
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
		boundRect[i] = boundingRect( Mat(contours_poly[i]) );
     }

  if(first) drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

  /// Draw polygonal contour + bonding rects + circles
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	   //drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	   if((boundRect[i].height > tsize && boundRect[i].width > tsize) && boundRect[i].height < src.size().height - 30) 
	   {
		   rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
	   }
     }

  /// Show in a window
  if(!first)
  {
	  addWeighted(src, 0.5, drawing, 0.5, 0.0, last);
	  //namedWindow( "Rocks", CV_WINDOW_AUTOSIZE );
	  imshow( "Rockyy", last );
	  waitKey(1);
	  //createTrackbar( "Threshold:", "Rock", &thresh, max_thresh, thresh_callback );
  }
}

void lightAndShadow()
{
	/// Convert image to gray and blur it
	cvtColor( src, src_gray, CV_BGR2GRAY );
	blur( src_gray, src_gray, Size(3,3) );

	thresh = findThresh(src_gray) + deviate;
	first = true;
	thresh_callback(0,0);

	thresh -= deviate*2;
	first = false;
	thresh_callback(0,0);
}

int findThresh(Mat img)
{
	int hval = 0, sum = 0, sumt = 0, mean = 0, meant = 0;
	int hei = img.size().height;
	int wid = img.size().width;
	for(int i = 0; i < hei; i++)
	{
		for(int j = 0; j < wid; j++)
		{
			int val = img.at<uchar>(i,j);
			sum += val;
		}
	}
	mean = sum / (hei*wid);
	return mean;
}

int Undistort2(CvCapture* lol)
{
	const int board_dt = 20;
	int board_w = 5; // Board width in squares
	int board_h = 8; // Board height 
	int n_boards = 5; // Number of boards
	int board_n = board_w * board_h;
	CvSize board_sz = cvSize( board_w, board_h );
	CvCapture* capture = lol;
	assert( capture );

	cvNamedWindow( "Calibration" );
	// Allocate Sotrage
	CvMat* image_points		= cvCreateMat( n_boards*board_n, 2, CV_32FC1 );
	CvMat* object_points		= cvCreateMat( n_boards*board_n, 3, CV_32FC1 );
	CvMat* point_counts			= cvCreateMat( n_boards, 1, CV_32SC1 );
	CvMat* intrinsic_matrix		= cvCreateMat( 3, 3, CV_32FC1 );
	CvMat* distortion_coeffs	= cvCreateMat( 5, 1, CV_32FC1 );

	CvPoint2D32f* corners = new CvPoint2D32f[ board_n ];
	int corner_count;
	int successes = 0;
	int step, frame = 0;

	IplImage *image = cvQueryFrame( capture );
	IplImage *gray_image = cvCreateImage( cvGetSize( image ), 8, 1 );

	// Capture Corner views loop until we've got n_boards
	// succesful captures (all corners on the board are found)

	if(calibCheck())
	{
		while( successes < n_boards ){
			// Skp every board_dt frames to allow user to move chessboard
			if( frame++ % board_dt == 0 ){
				// Find chessboard corners:
				int found = cvFindChessboardCorners( image, board_sz, corners,
					&corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );

				// Get subpixel accuracy on those corners
				cvCvtColor( image, gray_image, CV_BGR2GRAY );
				cvFindCornerSubPix( gray_image, corners, corner_count, cvSize( 11, 11 ), 
					cvSize( -1, -1 ), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

				// Draw it
				cvDrawChessboardCorners( image, board_sz, corners, corner_count, found );
				cvShowImage( "Calibration", image );

				// If we got a good board, add it to our data
				if( corner_count == board_n ){
					step = successes*board_n;
					for( int i=step, j=0; j < board_n; ++i, ++j ){
						CV_MAT_ELEM( *image_points, float, i, 0 ) = corners[j].x;
						CV_MAT_ELEM( *image_points, float, i, 1 ) = corners[j].y;
						CV_MAT_ELEM( *object_points, float, i, 0 ) = j/board_w;
						CV_MAT_ELEM( *object_points, float, i, 1 ) = j%board_w;
						CV_MAT_ELEM( *object_points, float, i, 2 ) = 0.0f;
					}
					CV_MAT_ELEM( *point_counts, int, successes, 0 ) = board_n;
					successes++;
				}
			} 

			// Handle pause/unpause and ESC
			int c = cvWaitKey( 15 );
			if( c == 'p' ){
				c = 0;
				while( c != 'p' && c != 27 ){
					c = cvWaitKey( 250 );
				}
			}
			if( c == 27 )
				return 0;
			image = cvQueryFrame( capture ); // Get next image
		} // End collection while loop

		// Allocate matrices according to how many chessboards found
		CvMat* object_points2 = cvCreateMat( successes*board_n, 3, CV_32FC1 );
		CvMat* image_points2 = cvCreateMat( successes*board_n, 2, CV_32FC1 );
		CvMat* point_counts2 = cvCreateMat( successes, 1, CV_32SC1 );

		// Transfer the points into the correct size matrices
		for( int i = 0; i < successes*board_n; ++i ){
			CV_MAT_ELEM( *image_points2, float, i, 0) = CV_MAT_ELEM( *image_points, float, i, 0 );
			CV_MAT_ELEM( *image_points2, float, i, 1) = CV_MAT_ELEM( *image_points, float, i, 1 );
			CV_MAT_ELEM( *object_points2, float, i, 0) = CV_MAT_ELEM( *object_points, float, i, 0 );
			CV_MAT_ELEM( *object_points2, float, i, 1) = CV_MAT_ELEM( *object_points, float, i, 1 );
			CV_MAT_ELEM( *object_points2, float, i, 2) = CV_MAT_ELEM( *object_points, float, i, 2 );
		}

		for( int i=0; i < successes; ++i ){
			CV_MAT_ELEM( *point_counts2, int, i, 0 ) = CV_MAT_ELEM( *point_counts, int, i, 0 );
		}
		cvReleaseMat( &object_points );
		cvReleaseMat( &image_points );
		cvReleaseMat( &point_counts );

		// At this point we have all the chessboard corners we need
		// Initiliazie the intrinsic matrix such that the two focal lengths
		// have a ratio of 1.0

		CV_MAT_ELEM( *intrinsic_matrix, float, 0, 0 ) = 1.0;
		CV_MAT_ELEM( *intrinsic_matrix, float, 1, 1 ) = 1.0;

		// Calibrate the camera
		cvCalibrateCamera2( object_points2, image_points2, point_counts2, cvGetSize( image ), 
			intrinsic_matrix, distortion_coeffs, NULL, NULL, CV_CALIB_FIX_ASPECT_RATIO ); 

		// Save the intrinsics and distortions
		cvSave( "Intrinsics.xml", intrinsic_matrix );
		cvSave( "Distortion.xml", distortion_coeffs );
	}

	// Example of loading these matrices back in
	CvMat *intrinsic = (CvMat*)cvLoad( "Intrinsics.xml" );
	CvMat *distortion = (CvMat*)cvLoad( "Distortion.xml" );

	// Build the undistort map that we will use for all subsequent frames
	IplImage* mapx = cvCreateImage( cvGetSize( image ), IPL_DEPTH_32F, 1 );
	IplImage* mapy = cvCreateImage( cvGetSize( image ), IPL_DEPTH_32F, 1 );
	cvInitUndistortMap( intrinsic, distortion, mapx, mapy );

	// Run the camera to the screen, now showing the raw and undistorted image
	cvNamedWindow( "Undistort" );

	while( image )
	{
		IplImage *t = cvCloneImage( image );
		cvShowImage( "Calibration", image ); // Show raw image
		cvRemap( t, image, mapx, mapy ); // undistort image
		cvReleaseImage( &t );
		cvShowImage( "Undistort", image ); // Show corrected image

		src = Mat(image);
		lightAndShadow();

		// Handle pause/unpause and esc
		int c = cvWaitKey( 15 );
		if( c == 'p' ){
			c = 0;
			while( c != 'p' && c != 27 ){
				c = cvWaitKey( 250 );
			}
		}
		if( c == 27 )
			break;
		image = cvQueryFrame( capture );
	}

	return 0;
}

bool calibCheck()
{
	again:
	string resp;
	cout << "Create new calibration? (Y/N) \n";
	getline(cin, resp);
	if(resp == "Y") return true;
	else if(resp == "N") return false;
	else goto again;
}

