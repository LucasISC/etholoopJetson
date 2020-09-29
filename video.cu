/*


                                        Ali NouriZonoz
                                        Huberlab
                                        Department of Basic Neuroscience
                                        University of Geneva
                                        Switzerland

                                        14 Dec 2018


                                        2D extraction of a single LED from XIMEA camera using Jetson TX2


*/


#include <atomic>
#include <netdb.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include <opencv2/photo/cuda.hpp>


#define ILOWV 20
#define WB_BLUE 2
#define WB_GREEN 1
#define WB_RED 1.3

/****************************************************************    Defining Global Variables        ***********************************************************/

using namespace cv;
using namespace std;

VideoCapture cap;
atomic<bool> getFrame;		// boolean telling when the camera frame is captured
Mat imageMorphSelected;
Mat imageRGB;
cuda::GpuMat    gpu_imageRAW,	// gpu matrix RAW image
gpu_imageRGB, 	// gpu matrix RGB image
gpu_imageHSV; 	// gpu matrix HSV image

bool running = true;


/****************************************************************************************************************************************************************/




/*************************************************************    CPU Thread for capturing images        ********************************************************/


/* gpu_getRGBimage
 * function returning RGB image from RAW captured image
 */
void gpu_getRGBimage(cuda::GpuMat gpu_imageRAW, cuda::GpuMat& gpu_imageRGB){

    // demosaicing RAW image
    cuda::demosaicing(gpu_imageRAW, gpu_imageRGB, COLOR_BayerRG2BGR);
    // multiplying with RGB scalar for the white balance
    cuda::multiply(gpu_imageRGB, Scalar(WB_BLUE, WB_GREEN, WB_RED), gpu_imageRGB);
}


/* gpu_getHSVimage
 * function returning HSV image from RGB image
 */
void gpu_getHSVimage(cuda::GpuMat gpu_imageRGB, cuda::GpuMat& gpu_imageHSV){

    // convert image from RGB to HSV
    cuda::cvtColor(gpu_imageRGB, gpu_imageHSV, COLOR_BGR2HSV);
}


/* getImage
 * Thread capturing the camera frame and storing it in imageRAW
 */

void *getImage(void *input)
{	
    // capture opencv variable
    VideoCapture *cap = (VideoCapture*) input;


    Mat imageRAW;			// frame of the camera// thresholded matrix

    // capture the frame until the user exists the program
    while(running){
        cap->grab();
        cap->retrieve(imageRAW);

        // if the captured frame is not empty, set getFrame flag to true
        if(!imageRAW.empty()){

            //store in gpu variable the RAW, RGB and HSV image
            gpu_imageRAW.upload(imageRAW);
            gpu_getRGBimage(gpu_imageRAW, gpu_imageRGB);
            gpu_getHSVimage(gpu_imageRGB, gpu_imageHSV);

            gpu_imageRGB.download(imageRGB);
            getFrame.store(true);
        }
    }

    pthread_exit(0);
}


/******************************************************    GPU function and kernel for thresholding image        *************************************************/

/* kernel_thresholdHSV
 * Function running in the GPU
 * kernel taking as input an HSV range (lowHSV and highHSV) to threshold and the source image.
 * the ouput is a mask where the  pixel value is either 255 if in the HSV range either 0 otherwise
 */
__global__ void kernel_thresholdHSV(const cuda::PtrStepSz<uchar3> src, cuda::PtrStepSzb dst,
                                    int lH, int hH, int lS, int hS, int lV, int hV) {
    // get (x,y) coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // if coordinates are out of the matrix size, return
    if (x >= src.cols || y >= src.rows) return;

    // get source pixel value
    uchar3 v = src(y, x);

    // if the pixel value is in the range then output is 255, else is 0
    if (v.x >= lH && v.x <= hH && v.y >= lS && v.y <= hS && v.z >= lV && v.z <= hV)
        dst(y, x) = 255;
    else
        dst(y, x) = 0;

}

/* gpu_thresholdHSV
 * Function launching the kernel for thresholding the image
 */
void gpu_thresholdHSV(cuda::GpuMat &src, Scalar &lowHSV, Scalar &highHSV,
                      cuda::GpuMat &dst) {

    // allocating memory for output matrix
    dst.create(src.size(), CV_8UC1);

    // number of threads created to parallelize the computation
    const int m = 32;
    int numRows = src.rows, numCols = src.cols;
    if (numRows == 0 || numCols == 0) return;

    const dim3 gridSize(ceil((float)numCols / m), ceil((float)numRows / m), 1);
    const dim3 blockSize(m, m, 1);

    kernel_thresholdHSV<<<gridSize, blockSize>>>(src, dst, lowHSV[0], highHSV[0], lowHSV[1], highHSV[1], lowHSV[2], highHSV[2]);
}

/**********************************************************************    CPU functions        ******************************************************************/




/* cameraInit
 * Function initializing camera parameters
 */
void cameraInit(VideoCapture& cap){

    // create a videocapture
    int open = cap.open(CV_CAP_XIAPI);
    if(!open){
        cout<<"Can't open camera"<<endl;
    }

    float gain=0.0;						// camera gain
    cap.set(CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE,1);		// put camera on Zero ROT mode.
    // See details at ximea.com
    cap.set(CV_CAP_PROP_XI_DOWNSAMPLING_TYPE,1);		// Seting downsampling type for skipping
    cap.set(CV_CAP_PROP_XI_DOWNSAMPLING,2);			// do the skiping in 2x2
    cap.set(CV_CAP_PROP_XI_DATA_FORMAT,5);			// set capturing mode for RAW 8
    cap.set(CV_CAP_PROP_XI_AEAG,0);				// no automatic adjusment of exposure and gain
    cap.set(CV_CAP_PROP_XI_EXPOSURE,800);			// set exposure (value in microseconds)
    cap.set(CV_CAP_PROP_XI_GAIN,gain);				// adjust gain
    cap.set(CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH,8);            // pixel size = 8 bits
    cap.set(CV_CAP_PROP_XI_AUTO_WB,0);				// no auto white background configurations

    // create thread for capturing the image
    pthread_t thread;
    pthread_create(&thread, NULL, getImage, (void *)& cap);
}

/* createSocket
 * function creating a socket to send the position values via UDP to the hostmachine
 */
int createSocket(char* hostMachine, char* port, int& sockfd, addrinfo* p ){

    // variables
    addrinfo hints, *servinfo, *p_temp;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;

    int rv = getaddrinfo(hostMachine, port, &hints, &servinfo);
    if (rv  != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        return 1;
    }


    // loop through all the results and make a socket
    for(p_temp = servinfo; p_temp != NULL; p_temp = p_temp->ai_next) {
        if ((sockfd = socket(p_temp->ai_family, p_temp->ai_socktype, p_temp->ai_protocol)) == -1) {
            perror("talker: socket");
            continue;
        }
        break;
    }
    if (p_temp == NULL) {
        fprintf(stderr, "talker: failed to create socket\n");
        return 2;
    }

    // copy p_temp into p
    memcpy(p, p_temp, sizeof *p_temp);

    return 0;
}

/* getCentroidImage
 * function returning the cendroid of a matrix
 */
void getCentroidImage(Mat image, Point& pos){

    // computes the moments of the image
    Moments oMoments = moments(image);

    double dM01 = oMoments.m01;
    double dM10 = oMoments.m10;
    double dArea = oMoments.m00;

    // mass center of the image
    pos.x = dM10 / dArea;
    pos.y = dM01 / dArea;
}

/* createControlInterface
 * function creating a user interface to control the HSV thresholding
 */
void createControlInterface(int& nColor, int nbColors, int& iLowH, int& iHighH, int& iLowS, int& iHighS, int& iLowV, int& iHighV, int& sizeMorph){

    // create opencv window
    const char* nWindow = "HSV Threshold result";
    namedWindow(nWindow, WINDOW_NORMAL);

    // create number target trackbar
    cvCreateTrackbar("Select target", nWindow, &nColor, nbColors-1);

    // create Hue trackbar
    cvCreateTrackbar("LowH", nWindow, &iLowH, 179);
    cvCreateTrackbar("HighH", nWindow, &iHighH, 179);

    // create Saturation trackbar
    cvCreateTrackbar("LowS", nWindow, &iLowS, 255);
    cvCreateTrackbar("HighS", nWindow, &iHighS, 255);

    // create value trackbar
    cvCreateTrackbar("LowV", nWindow, &iLowV, 255);
    cvCreateTrackbar("HighV", nWindow, &iHighV, 255);

    cvCreateTrackbar("Size Morph", nWindow, &sizeMorph, 15);
}



/* onMouse
 * function called when using mouse in opencv window
 */
static void onMouse( int event, int x, int y, int f, void* p){

    // if there is a click on the image
    if (event == CV_EVENT_LBUTTONDOWN){
        // convert the pixel value in HSV value
        Mat HSV;
        Mat RGB=imageRGB(Rect(x,y,1,1));
        cvtColor(RGB, HSV,CV_BGR2HSV);
        Vec3b hsv=HSV.at<Vec3b>(0,0);

        // print HSV value
        cout<<"HSV : "<<hsv<<"\t(x,y) : ("<<x<<","<<y<<")"<<endl;
    }
}


/* ColorStruct
 * structure storing information for the color detection
 */
struct ColorStruct{
    int numColor;		// color number
    Scalar lowHSV;
    Scalar highHSV;
    int sizeMorph;
    Scalar RGB;
    Point pos;			// coordinates of the color centroid
    atomic<bool> compute;	// boolean to know when to compute
    atomic<bool> send;          // boolean to know when to send the pos
    bool selected = false;
};


/* gpu_morphImage
 * function opening and closing the image with an element of size sizeMorph
 */
void gpu_morphImage(cuda::GpuMat gpu_imageThresh, cuda::GpuMat& gpu_imageMorph, int& sizeMorph){

    if(sizeMorph == 0)
        sizeMorph=1;

    // creating the element of size sizeMorph
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(sizeMorph, sizeMorph));
    //cuda::GpuMat gpu_element(element);

    //create the dilate and erode filter
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, gpu_imageThresh.type(), element);
    Ptr<cuda::Filter> erodeFilter = cuda::createMorphologyFilter(MORPH_ERODE, gpu_imageThresh.type(), element);

    // opening the image
    erodeFilter->apply(gpu_imageThresh, gpu_imageThresh);
    dilateFilter->apply(gpu_imageThresh, gpu_imageThresh);

    //closing the image
    dilateFilter->apply(gpu_imageThresh, gpu_imageThresh);
    erodeFilter->apply(gpu_imageThresh, gpu_imageMorph);
}


/* getPosColor
 * thread function to retrieve the coordinates of the centroid image thesholded depending of the HSV range
 */
void* getPosColor(void *input){

    // get the input structure which corresponds to a color to detect
    ColorStruct *color = static_cast<ColorStruct*>(input);

    Mat imageMorph;

    int sizeMorph =4;
    
    cuda::GpuMat    gpu_imageThresh,	// gpu matrix thresholded image
                    gpu_imageMorph;

    while(running){

        // if ready to compute
        if(color->compute){

            // set compute variable to false
            color->compute.store(false);

            // threshold image from selected HSV range
            gpu_thresholdHSV(gpu_imageHSV, color->lowHSV, color->highHSV, gpu_imageThresh);

            gpu_morphImage(gpu_imageThresh, gpu_imageMorph, sizeMorph);

            //store in CPU matrix the thresholded image
            gpu_imageMorph.download(imageMorph);

            // compute centroid of thresholded image
            getCentroidImage(imageMorph, color->pos);

            if(color->selected)
                imageMorphSelected = imageMorph;

            // ready to send pos
            color->send.store(true);
        }
    }

    pthread_exit(0);
}



/*************************************************************    Main function       ********************************************************/

/* main
 * there are 3 modes in the main function, each mode is called by writing it as the second argument when executing the program
 * - "test" mode : mode to get the HSV ranges for the color detection, with a trackbar, the clickable image to get the pixel HSV value and the thesholded mask
 * - "led" mode : mode for the calibration part, using the led. image thresholded on intensity
 * - "color" mode : mode taking as input the number of colors to detect and the HSV ranges and outputs the coordinates of each color detected.
 */
int main(int argc, char *argv[])
{

    // get the chosen mode, it's the second argument when calling the program
    string mode = string(argv[1]);

    time_t start, end;
    int count = 0;

    // if chosen mode is "test" ***************************************************
    if (mode == "test"){

        // get number of colors to detect
        int nbColors = atoi(argv[2]);

        // if the number of arguments is not exact return, has to be 6*nbColors + 4 arguments
        if(argc != 3*nbColors + 3){
            cout<<"not the correct number of arguments"<<endl;
            return 0;
        }

        // initialize camera settings
        cameraInit(cap);

        ColorStruct colors[nbColors];
        pthread_t threads[nbColors];

        for(int i=0; i<nbColors; i++){
            colors[i].numColor = i;
            colors[i].RGB = Scalar(atoi(argv[3*i + 3]), atoi(argv[3*i + 4]), atoi(argv[3*i + 5]));
            colors[i].compute.store(false);
            colors[i].pos = Point(0,0);
            colors[i].lowHSV = Scalar(0,0,0);
            colors[i].highHSV = Scalar(179,255,255);
            colors[i].sizeMorph = 1;
            // create the threads
            pthread_create(&threads[i], NULL, getPosColor, (void *)& colors[i]);
        }

        // HSV range variables
        int lowH = 0, highH = 179;
        int lowS = 0, highS = 255;
        int lowV = 0, highV = 255;
        int sizeMorph = 3;
        int nColor = 1;
        int nColor_temp = 1;

        // create user interface to control the HSV values
        createControlInterface(nColor, nbColors, lowH, highH, lowS, highS, lowV, highV, sizeMorph);

        while(1){

            // if got frame
            if(getFrame){

                // set value to false
                getFrame.store(false);

                if(nColor_temp!=nColor){
                    setTrackbarPos("LowH", "HSV Threshold result", colors[nColor].lowHSV[0]);
                    setTrackbarPos("LowS", "HSV Threshold result", colors[nColor].lowHSV[1]);
                    setTrackbarPos("LowV", "HSV Threshold result", colors[nColor].lowHSV[2]);

                    setTrackbarPos("HighH", "HSV Threshold result", colors[nColor].highHSV[0]);
                    setTrackbarPos("HighS", "HSV Threshold result", colors[nColor].highHSV[1]);
                    setTrackbarPos("HighV", "HSV Threshold result", colors[nColor].highHSV[2]);

                    setTrackbarPos("Size Morph", "HSV Threshold result", colors[nColor].sizeMorph);
                    nColor_temp = nColor;
                }
                else{
                    // create the two HSV limits
                    colors[nColor].lowHSV = Scalar(lowH, lowS, lowV);
                    colors[nColor].highHSV = Scalar(highH, highS, highV);

                    colors[nColor].sizeMorph = sizeMorph;
                }

                for (int i=0; i<nbColors; i++){

                    if(i == nColor)
                        colors[i].selected = true;
                    else
                        colors[i].selected = false;

                    colors[i].compute.store(true);

                    // create circle over RGB image to show the position of detected color
                    circle(imageRGB, colors[i].pos, 3, colors[i].RGB, 2 );
                }

                if(!imageMorphSelected.empty()){
                    // show thresholded image (black and white mask)
                    imshow("out", imageMorphSelected);
                    waitKey(2);
                }

                // show RGB image with the circle
                imshow("XIMEA camera", imageRGB);
                waitKey(1);

                // set the mouse callback function to get the HSV pixel value
                setMouseCallback( "XIMEA camera", onMouse, NULL);

            }
        }
    }

    // if chosen mode is "led" ***************************************************
    else if (mode == "led"){

        // initialize camera settings
        cameraInit(cap);

        // RGB CPU images
        Mat imageThresh;

        cuda::GpuMat 	gpu_imageThresh;	// gpu matrix thresholded image

        // variables for the udp socket
        int sockfd;
        char* hostMachine = argv[2];    // the third variable is the hostmachine name to send the udp messages
        addrinfo *p = new addrinfo();   // the fourth variable is the port
        char* port = argv[3];

        // create socket
        createSocket(hostMachine, port, sockfd, p);

        // position of the detected color
        Point pos;

        time(&start);

        while(count<100000){

            // if frame captures
            if(getFrame){
                // store false value
                getFrame.store(false);

                // threshold the intensity value
                cuda::threshold(gpu_imageRAW, gpu_imageThresh, ILOWV, 1, THRESH_BINARY);

                // store in CPU matrix the thresholded image
                gpu_imageThresh.download(imageThresh);

                // compute centroid
                getCentroidImage(imageThresh, pos);

                // send via UDP the position
                sendto(sockfd, &pos, sizeof(pos), 0, p->ai_addr, p->ai_addrlen);

                // print position value
                cout<<count<<"\t"<<pos.x<<"\t"<<pos.y<<endl;

                count++;
            }
        }

    }

    // if chosen mode is "color" ***************************************************
    else if (mode == "color"){

        // get number of colors to detect
        int nbColors = atoi(argv[4]);

        // if the number of arguments is not exact return, has to be 6*nbColors + 4 arguments
        if(argc != 6*nbColors + 5){
            cout<<"not the correct number of arguments"<<endl;
            return 0;
        }

        // initialize camera settings
        cameraInit(cap);

        // variables for the udp socket
        int sockfd;
        char* hostMachine = argv[2];    // the third variable is the hostmachine name to send the udp messages
        addrinfo *p = new addrinfo();
        char* port = argv[3];           // the fourth variable is the port

        // create socket
        createSocket(hostMachine, port, sockfd, p);

        // create the number of colors structures and threads
        ColorStruct colors[nbColors];
        pthread_t threads[nbColors];

        // initialize the structures
        for (int i=0; i<nbColors; i++){
            colors[i].numColor = i;				// color number
            colors[i].lowHSV = Scalar(atoi(argv[6*i + 5]), atoi(argv[6*i + 7]), atoi(argv[6*i + 9]));
            colors[i].highHSV = Scalar(atoi(argv[6*i + 6]), atoi(argv[6*i + 8]), atoi(argv[6*i + 10]));
            colors[i].pos = Point(0,0);				// position of detected color
            colors[i].compute.store(false);				// if ready to compute pos
            colors[i].send.store(false);				// if ready to send pos

            // create the threads
            pthread_create(&threads[i], NULL, getPosColor, (void *)& colors[i]);
        }

        time(&start);

        while(count<10000){

            // if frame captures
            if(getFrame){
                // store false value
                getFrame.store(false);

                // go through the colors
                for(int i=0; i<nbColors; i++){
                    // ready to compute
                    colors[i].compute.store(true);

                    // if ready to send pos values
                    if(colors[i].send){
                        // store false value
                        colors[i].send.store(false);

                        // create message to send : [color numver, pos.x, pos.y]
                        int message[3] = {colors[i].numColor, colors[i].pos.x, colors[i].pos.y};

                        // print message
                        cout<<count<<"\tmessage :"<<message[0]<<" "<<message[1]<<" "<<message[2]<<" "<<endl;

                        // send message via udp
                        sendto(sockfd, &message, sizeof(message), 0, p->ai_addr, p->ai_addrlen);

                        if(i==0)count ++;
                    }
                }
            }
        }
    }

    else
        cout<<"No mode chosen"<<endl;

    running = false;

    time(&end); // find the ending time

    // here the program prints your frame rate exits
    double seconds = difftime (end, start);
    cout << "Time taken : " << seconds << " seconds" << endl;

    double fps  = count / seconds;
    cout << "Estimated frames per second : " << fps << endl;
    
    //cap.release();

    return 0;
}





