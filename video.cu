/*
                                        Ali NouriZonoz
                                        Huberlab
                                        Department of Basic Neuroscience
                                        University of Geneva
                                        Switzerland

                                        Lucas Maigre
                                        Institut des Sciences Cognitives
                                        Bron
                                        France


                                        2D extraction of a single LED from XIMEA camera using Jetson TX2

*/


#include <atomic>
#include <netdb.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include <opencv2/photo/cuda.hpp>
#include <chrono>
#include <algorithm>
#include <sys/types.h> 
#include <sys/socket.h>

#define ILOWV 20

// White balance, fixed so the color filtering is always the same
#define WB_BLUE 2
#define WB_GREEN 1
#define WB_RED 1

/****************************************************************    Defining Global Variables        ***********************************************************/

using namespace cv;
using namespace std;
using namespace std::chrono;

VideoCapture cap;                   // gets the images with opencv
atomic<bool> getFrame;              // boolean telling when the camera frame is captured
Mat imageDenoiseSelected;           // denoised image
Mat imageRAW, imageRGB, imageSHOWN; // images matrices
cuda::GpuMat    gpu_imageRAW,       // gpu matrix RAW image
gpu_imageRGB_uncorrected,
gpu_imageRGB,                       // gpu matrix RGB image
gpu_imageHSV;                       // gpu matrix HSV image
cuda::GpuMat 	gpu_image[4], gpu_image_unoised[4];

bool pauseVideo = false;
bool startRecord = false;
bool okMsg = false;
bool enableJetsonTest = false;      // if we enable the mask when we record and keep the jetson camera
int rotateImg = 0;

// create thread for capturing the image
pthread_t thread_cam, thread_record, thread_udp;

string idCam;

bool stopRecording = false;

// variables for udp connection
int sockfd;
addrinfo *p = new addrinfo();

/* ColorStruct
 * structure storing information for the color detection
 */
struct ColorStruct{
    int numColor;		// color number
    Scalar lowHSV;              // low HSV filter
    Scalar highHSV;             // hign HSV filter
    int sizeMask;
    Scalar RGB;                 // rgb value chosen, just for humans
    Point pos;			// coordinates of the color centroid
    Point posOld;               // coordinates of the previous color centroid
    int count;
};

int numberColors;
ColorStruct colors[10];




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

// get gpu matrices (RGB, HSV) from the cpu matrix
void getImageRGB(){

    //store in gpu variable the RAW, RGB and HSV image
    gpu_imageRAW.upload(imageRAW);
    gpu_getRGBimage(gpu_imageRAW, gpu_imageRGB_uncorrected);

    cuda::gammaCorrection(gpu_imageRGB_uncorrected, gpu_imageRGB);

    gpu_getHSVimage(gpu_imageRGB, gpu_imageHSV);
}

/****************************************************************************************************************************************************************/




/*************************************************************    CPU Thread for capturing images        ********************************************************/

/* udpSocket
 * Thread for the communication between jetson and the hostmachine
 */
void * udpSocket(void *input){

    char buf[10];
    int len;
    int n;

    memset(buf, '\0', sizeof(buf));

    while(!stopRecording){

        // receives buffer message in "buf"
        n = recvfrom(sockfd, buf, sizeof(buf), MSG_WAITALL, p->ai_addr, (socklen_t*)&len);
        cout<<"buf:"<<buf<<endl;

        if(n>0 && strstr(buf, "OK"))
            okMsg = true;

        else if(n>0 && strstr(buf, "STOP"))
            stopRecording = true;
    }
    return NULL;
}


/* getImage
 * Thread capturing the camera frame and storing it in imageRAW
 */

void * getImage(void *input)
{
    // capture opencv variable
    VideoCapture *cap = (VideoCapture*) input;

    // capture the frame until the user exists the program
    while(!stopRecording){
        int a = cap->grab();
        cap->retrieve(imageRAW);

        // if the captured frame is not empty, set getFrame flag to true
        if(!imageRAW.empty())
            getFrame.store(true);

        if(!a)
            return NULL;
    }

    return NULL;
}

void * recordCam(void *input){

    // Get resolution and framerate from capture
    unsigned int width = cap.get (cv::CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get (cv::CAP_PROP_FRAME_HEIGHT);
    unsigned int fps = 30;
    float intervalFrameMicrosec = 1000000/fps;

    cout<<idCam<<endl;

    cout<<"h:"<<height<<"w:"<<width<<endl;

    // video directory
    string outDir = "/home/nvidia/Desktop/JetsonData/";

    // gpu image of mask
    cuda::GpuMat gpu_image_convert(height, width , 16);

    // save image mask
    if(enableJetsonTest)
        width = width*2;

    // Create the writer with gstreamer pipeline encoding into H264, muxing into mkv container and saving to file
    VideoWriter gst_nvh264_writer("appsrc ! queue ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc bitrate=8000000 insert-sps-pps=true ! h264parse ! qtmux ! filesink location=" + outDir + idCam + ".mp4", CAP_GSTREAMER,  0, fps, cv::Size (width, height));

    if (!gst_nvh264_writer.isOpened ()) {
        std::cout << "Failed to open gst_nvh264 writer." << std::endl;
        return (NULL);
    }

    // used for having corredct frame rate
    long int time1 = 0, time2 = 0;

    // image RGB + image mask
    cuda::GpuMat gpu_imageConcat(height, width, 16);
    Mat imageConcat;

    while(!stopRecording){

        if(!startRecord)
            continue;

        // interval for having the good frame rate
        time2 = duration_cast< microseconds >( system_clock::now().time_since_epoch() ).count();

        if(time2 - time1 >= intervalFrameMicrosec ){

            time1 = duration_cast< microseconds >( system_clock::now().time_since_epoch() ).count();

            // if we save mask
            if(enableJetsonTest){

                gpu_image_convert.setTo(Scalar(0,0,0));

                for(int i=0; i<numberColors; i++)
                    gpu_image_convert.setTo(Scalar(255,255,255), gpu_image[i]);

                gpu_imageRGB.copyTo(gpu_imageConcat(Rect(0,0,  gpu_imageRGB.cols,gpu_imageRGB.rows)));
                gpu_image_convert.copyTo(gpu_imageConcat(Rect(gpu_image_convert.cols,0,   gpu_image_convert.cols, gpu_image_convert.rows)));

                // gpu mat to cpu mat
                gpu_imageConcat.download(imageConcat);

                // writes cpu mat to file
                gst_nvh264_writer.write(imageConcat);
            }

            // if we don't save mask
            else{
                // gpu mat to cpu mat
                gpu_imageRGB.download(imageRGB);
                // writes cpu mat to file
                gst_nvh264_writer.write(imageRGB);
            }
        }
    }

    gst_nvh264_writer.release();
    cout<<"video writer out"<<endl;

    return NULL;
}




/******************************************************    GPU function and kernel for thresholding image        *************************************************/

/* kernel_thresholdHSV
 * Function running in the GPU
 * kernel taking as input an HSV range (lowHSV and highHSV) to threshold, and the source image.
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

    // if the pixel value is in the range then output is 255 (or 1), else is 0
    if (lH<hH && v.x >= lH && v.x <= hH && v.y >= lS && v.y <= hS && v.z >= lV && v.z <= hV)
        dst(y, x) = 1;
    else if (lH>hH && (v.x <= hH || v.x >= lH) && v.y >= lS && v.y <= hS && v.z >= lV && v.z <= hV)
        dst(y, x) = 1;
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

/* kernel_noiseReduction
 * Function running in the GPU
 * kernel taking as input the mask matrix (probably noised) to denoised it with a mask of size "sizeMask"
 */
__global__ void kernel_noiseReduction(const cuda::PtrStepSzb src, cuda::PtrStepSzb dst, int sizeMask) {
    // get (x,y) coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // if coordinates are out of the matrix size, return
    if (x >= src.cols || y >= src.rows) return;

    // get source pixel value
    int v = src(y, x);

    float value = 0;
    int iStart = x-sizeMask/2;
    int jStart = y-sizeMask/2;

    if(v==0)
        dst(y, x) = 0;
    else{
        for(int i=0; i<sizeMask; i++){
            if(iStart+i>=0 && iStart+i<src.cols){
                for(int j=0; j<sizeMask; j++){
                    if(jStart+j>=0 && jStart+j<src.rows)
                        value+=src(jStart+j, iStart+i);
                }
            }
        }

        value/=sizeMask*sizeMask;

        if(value>0.5)
            dst(y, x) = 1;
        else
            dst(y, x) = 0;
    }
}

/* gpu_noiseReduction
 * Function launching the kernel for noise reduction
 */
void gpu_noiseReduction(cuda::GpuMat &src, cuda::GpuMat &dst, int sizeMask) {

    // allocating memory for output matrix
    dst.create(src.size(), CV_8UC1);

    // number of threads created to parallelize the computation
    const int m = 32;
    int numRows = src.rows, numCols = src.cols;
    if (numRows == 0 || numCols == 0) return;

    const dim3 gridSize(ceil((float)numCols / m), ceil((float)numRows / m), 1);
    const dim3 blockSize(m, m, 1);

    kernel_noiseReduction<<<gridSize, blockSize>>>(src, dst, sizeMask);
}


/* kernel_centroid
 * Function running in the GPU
 * kernel taking as input the mask matrix to get the centroid of detected target
 */
__global__ void kernel_centroid(const cuda::PtrStepSzb src, int* centroid) {
    // get (x,y) coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // if coordinates are out of the matrix size, return
    if (x >= src.cols || y >= src.rows) return;

    // get source pixel value
    int v = src(y, x);

    if(v){
        atomicAdd(&centroid[0], x);
        atomicAdd(&centroid[1], y);
        atomicAdd(&centroid[2], 1);
    }

}

/* gpu_getCentroidImage
 * Function launching the kernel for the target centroid
 */
void gpu_getCentroidImage(cuda::GpuMat &src, Point &centroid) {

    // number of threads created to parallelize the computation
    const int m = 32;
    int numRows = src.rows, numCols = src.cols;
    if (numRows == 0 || numCols == 0) return;

    const dim3 gridSize(ceil((float)numCols / m), ceil((float)numRows / m), 1);
    const dim3 blockSize(m, m, 1);

    int n =3;
    int moment[n] = {0,0,0};
    int *dmoment;

    cudaMalloc((void**)&dmoment,sizeof(int)*n);
    cudaMemcpy(dmoment,&moment,sizeof(int)*n,cudaMemcpyHostToDevice);

    kernel_centroid<<<gridSize, blockSize>>>(src, dmoment);

    cudaMemcpy(&moment,dmoment,sizeof(int)*n,cudaMemcpyDeviceToHost);

    int xcenter = (moment[0]/moment[2]);
    int ycenter = (moment[1]/moment[2]);

    centroid = Point(xcenter, ycenter);
}

/**********************************************************************    CPU functions        ******************************************************************/




/* cameraInit
 * Function initializing camera parameters
 */
int cameraInit(VideoCapture& cap){

    // create a videocapture
    int open = cap.open(CV_CAP_XIAPI);
    if(!open){
        cout<<"Can't open camera"<<endl;
        return 0;
    }

    float gain=0.0;						// camera gain
    cap.set(CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE,1);		// put camera on Zero ROT mode.
    // See details at ximea.com
    cap.set(CV_CAP_PROP_XI_DATA_FORMAT,5);			// set capturing mode for RAW 8
    cap.set(CV_CAP_PROP_XI_AEAG,0);				// no automatic adjusment of exposure and gain
    cap.set(CV_CAP_PROP_XI_EXPOSURE,4000);			// set exposure (value in microseconds)
    cap.set(CV_CAP_PROP_XI_GAIN,gain);				// adjust gain
    cap.set(CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH,8);            // pixel size = 8 bits
    cap.set(CV_CAP_PROP_XI_AUTO_WB,0);				// no auto white background configurations
    //cap.set(CAP_PROP_XI_ACQ_TIMING_MODE,1);
    //cap.set(CV_CAP_PROP_XI_FRAMERATE, 60);


    // creates thread for capturing images from cam
    pthread_create(&thread_cam, NULL, getImage, (void *)& cap);

    return 1;
}

/* recordInit
 * Function creating thread for the record of the jetson camera
 */
void recordInit(){
    pthread_create(&thread_record, NULL, recordCam,  NULL);
}

/* createSocket
 * function creating a socket to send the position values via UDP to the hostmachine
 */
int createSocket(char* hostMachine, char* port){

    // variables
    addrinfo hints, *servinfo, *p_temp;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;

    int rv = getaddrinfo(hostMachine, port, &hints, &servinfo);
    if (rv  != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        return 1;
    }

    // loop through all the results and make a socket
    for(p_temp = servinfo; p_temp != NULL; p_temp = p_temp->ai_next) {
        if ((sockfd = socket(p_temp->ai_family, p_temp->ai_socktype, 0)) == -1) {
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

    pthread_create(&thread_udp, NULL, udpSocket,  NULL);

    return 0;
}

// DELETE
//void callbBackButton(int state, void*){}


/* createControlInterface
 * function creating a user interface to control the HSV thresholding
 */
void createControlInterface(int& nColor, int nbColors, int& iLowH, int& iHighH, int& iLowS, int& iHighS, int& iLowV, int& iHighV, int& sizeMask){

    // create opencv window
    const char* nWindow = "XIMEA camera";
    namedWindow(nWindow, WINDOW_GUI_NORMAL);
    resizeWindow("XIMEA camera", 640,682);

    namedWindow("out", WINDOW_GUI_NORMAL);
    resizeWindow("out", 640,512);

    if(nbColors>1){
        // create number target trackbar
        cvCreateTrackbar("Select target", nWindow, &nColor, nbColors-1);
    }

    // create Hue trackbar
    cvCreateTrackbar("LowH", nWindow, &iLowH, 179);
    cvCreateTrackbar("HighH", nWindow, &iHighH, 179);

    // create Saturation trackbar
    cvCreateTrackbar("LowS", nWindow, &iLowS, 255);
    cvCreateTrackbar("HighS", nWindow, &iHighS, 255);

    // create value trackbar
    cvCreateTrackbar("LowV", nWindow, &iLowV, 255);
    cvCreateTrackbar("HighV", nWindow, &iHighV, 255);

    cvCreateTrackbar("Size Mask", nWindow, &sizeMask, 15);
}



/* onMouse
 * function called when using mouse in opencv window
 * returns (x,y) and pixel value (HSV)
 */
static void onMouse( int event, int x, int y, int f, void* input){

    // if there is a click on the image
    if (event == CV_EVENT_LBUTTONDOWN){

        // convert the pixel value in HSV value
        Mat pixRGB, pixHSV, imageRGBrotate;
        
        // if images is rotated, make sure to rotate matrix to get correct value
        if(rotateImg == 1){
            rotate(imageRGB, imageRGBrotate, cv::ROTATE_90_CLOCKWISE);
        }
        else if(rotateImg == 2){
            rotate(imageRGB, imageRGBrotate, cv::ROTATE_180);

        }
        else if(rotateImg == 3){
            rotate(imageRGB, imageRGBrotate, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
        else
            imageRGBrotate = imageRGB;
        
        pixRGB=imageRGBrotate(Rect(x,y,1,1));
        cvtColor(pixRGB, pixHSV,CV_BGR2HSV);
        Vec3b hsv=pixHSV.at<Vec3b>(0,0);

        string statusBarText = "H: "+ to_string(hsv[0]) + ", S: "+ to_string(hsv[1]) + ", V: "+ to_string(hsv[2]) + " / x: " + to_string(x) + "y: " + to_string(y);
        displayOverlay("XIMEA camera", statusBarText, 0);

        // print HSV value
        cout<<"HSV : "<<hsv<<"\t(x,y) : ("<<x<<","<<y<<")"<<endl;
    }

    // if there is a click on the image
    if (event == CV_EVENT_MBUTTONDOWN){
        pauseVideo = !pauseVideo;
    }
}




/*************************************************************    Main function       ********************************************************/

/* main
 * there are 3 modes in the main function, each mode is called by writing it as the second argument when executing the program
 * - "test" mode : mode to get the HSV ranges for the color detection, with a trackbar, the clickable image to get the pixel HSV value and the thesholded mask, used for color calibration
 * - "led" mode : mode for the calibration part, using the led. image thresholded on intensity,  not used for ISC setup because of glass pannels reflecting led light
 * - "color" mode : mode taking as input the number of colors to detect and the HSV ranges and outputs the coordinates of each color detected.
 * - "colorecord" mode : same as "color" mode but saves jetson camera images also (sent to server) in a video.
 * - "colorecordtest" mode : same as "colorecord" mode but the mask is also saved in video
 * - "record" mode : only records jetson cameras (no position tracking). Used for neural network image capture
 */
int main(int argc, char *argv[])
{

    // get the chosen mode, it's the second argument when calling the program
    string mode = string(argv[1]);

    long int start, end;
    long int elapsedStart, elapsedTemp;

    long int count = 0;

    // if chosen mode is "test" ***************************************************
    if (mode == "test" && argc>2){

        // get number of colors to detect
        numberColors = atoi(argv[2]);

        bool isHSV;

        // if the number of arguments is not exact return, has to be 6*numberColors + 4 arguments
        if(argc == 3*numberColors + 3){
            isHSV =false;
        }
        else if (argc == 10*numberColors + 3){
            isHSV =true;
        }
        else{
            cout<<"not the correct number of arguments"<<endl;
            return 0;
        }

        // initialize camera settings
        int ret = cameraInit(cap);
        if(!ret)
            return 0;

        // puts the argument in the color structure
        for(int i=0; i<numberColors; i++){
            colors[i].numColor = i;
            colors[i].RGB = Scalar(atoi(argv[10*i + 5]), atoi(argv[10*i + 4]), atoi(argv[10*i + 3]));
            colors[i].pos = Point(0,0);
            colors[i].lowHSV = Scalar(0,0,0);
            colors[i].highHSV = Scalar(179,255,255);
            colors[i].sizeMask = 3;
            if(isHSV){
                puts("ok");
                colors[i].lowHSV = Scalar(atoi(argv[10*i + 6]),atoi(argv[10*i + 8]),atoi(argv[10*i + 10]));
                colors[i].highHSV = Scalar(atoi(argv[10*i + 7]),atoi(argv[10*i + 9]),atoi(argv[10*i + 11]));
                colors[i].sizeMask = atoi(argv[10*i + 12]);
            }
        }

        // HSV range variables
        int lowH = 0, highH = 179;
        int lowS = 0, highS = 255;
        int lowV = 0, highV = 255;
        int sizeMask = 3;

        if(isHSV){
            lowH = atoi(argv[6]); highH = atoi(argv[7]);
            lowS = atoi(argv[8]); highS = atoi(argv[9]);
            lowV = atoi(argv[10]); highV = atoi(argv[11]);
            sizeMask = atoi(argv[12]);
        }

        int nColor = 0;
        int nColor_temp = 0;

        // create user interface to control the HSV values
        createControlInterface(nColor, numberColors, lowH, highH, lowS, highS, lowV, highV, sizeMask);

        bool recording = true;

        // looop to show image, mask and update the control interface
        while(recording){

            // if got frame
            if(getFrame){

                // set value to false
                getFrame.store(false);

                getImageRGB();

                // go through the colors
                for(int i=0; i<numberColors; i++){

                    gpu_thresholdHSV(gpu_imageHSV, colors[i].lowHSV, colors[i].highHSV, gpu_image[i]);

                    gpu_noiseReduction(gpu_image[i], gpu_image[i], colors[i].sizeMask);

                    gpu_getCentroidImage(gpu_image[i], colors[i].pos);
                }


                if(nColor_temp!=nColor){

                    setTrackbarPos("LowH", "XIMEA camera", colors[nColor].lowHSV[0]);
                    setTrackbarPos("LowS", "XIMEA camera", colors[nColor].lowHSV[1]);
                    setTrackbarPos("LowV", "XIMEA camera", colors[nColor].lowHSV[2]);

                    setTrackbarPos("HighH", "XIMEA camera", colors[nColor].highHSV[0]);
                    setTrackbarPos("HighS", "XIMEA camera", colors[nColor].highHSV[1]);
                    setTrackbarPos("HighV", "XIMEA camera", colors[nColor].highHSV[2]);

                    setTrackbarPos("Size Mask", "XIMEA camera", colors[nColor].sizeMask);
                    nColor_temp = nColor;
                }
                else{
                    // create the two HSV limits
                    colors[nColor].lowHSV = Scalar(lowH, lowS, lowV);
                    colors[nColor].highHSV = Scalar(highH, highS, highV);

                    colors[nColor].sizeMask = sizeMask;
                }

                // if middle button clicked, pause video
                if(!pauseVideo){

                    gpu_imageRGB.download(imageRGB);
                    gpu_imageRGB.download(imageSHOWN);

                    for (int i=0; i<numberColors; i++){
                        // create circle over RGB image to show the position of detected color
                        circle(imageSHOWN, colors[i].pos, 3, colors[i].RGB, 2 );
                    }
                    
                    gpu_image[nColor].download(imageDenoiseSelected);
                }
                
                
                // if "f" pressed, rotate image
                if(rotateImg == 1){
                    rotate(imageSHOWN, imageSHOWN, cv::ROTATE_90_CLOCKWISE);
                    rotate(imageDenoiseSelected, imageDenoiseSelected, cv::ROTATE_90_CLOCKWISE);
                }
                else if(rotateImg == 2){
                    rotate(imageSHOWN, imageSHOWN, cv::ROTATE_180);
                    rotate(imageDenoiseSelected, imageDenoiseSelected, cv::ROTATE_180);

                }
                else if(rotateImg == 3){
                    rotate(imageSHOWN, imageSHOWN, cv::ROTATE_90_COUNTERCLOCKWISE);
                    rotate(imageDenoiseSelected, imageDenoiseSelected, cv::ROTATE_90_COUNTERCLOCKWISE);

                }
                
                //show mask image
                imshow("out", 255*imageDenoiseSelected);

                // show RGB image with the circle
                imshow("XIMEA camera", imageSHOWN);
                
                
                // set the mouse callback function to get the HSV pixel value
                setMouseCallback( "XIMEA camera", onMouse, NULL);
                setMouseCallback( "out", onMouse, NULL);
                
                int key = waitKey(1);

                // if "esc" or "q" pressed, quit
                if(key == 27 || key == 113){
                    for(int i=0; i<numberColors; i++)
                        cout<<"nHSVm:"<<i<<","<<colors[i].lowHSV[0]<<" "<<colors[i].highHSV[0]<<" "<<colors[i].lowHSV[1]<<" "<<colors[i].highHSV[1]<<" "<<colors[i].lowHSV[2]<<" "<<colors[i].highHSV[2]<<" "<<colors[i].sizeMask<<endl;
                    
                    destroyAllWindows();
                    recording = false;
                    stopRecording = true;
                }
                // if "f" pressed, rotate image
                else if(key == 102){
                    rotateImg += 1;
                    if(rotateImg == 4)
                        rotateImg = 0;

                }
            }
        }
    }

    // if chosen mode is "led" ***************************************************
    else if (mode == "led" && argc>2){

        // initialize camera settings
        int ret = cameraInit(cap);
        if(!ret)
            return 0;

        cuda::GpuMat 	gpu_imageThresh;	// gpu matrix thresholded image

        // variables for the udp socket
        char* hostMachine = argv[2];    // the third variable is the hostmachine name to send the udp messages
        // the fourth variable is the port
        char* port = argv[3];

        // create socket
        createSocket(hostMachine, port);

        // position of the detected color
        Point pos;

        start = duration_cast< milliseconds >( system_clock::now().time_since_epoch() ).count();

        while(count<100000){

            // if frame captures
            if(getFrame){
                // store false value
                getFrame.store(false);

                //store in gpu variable the RAW image
                gpu_imageRAW.upload(imageRAW);

                // threshold the intensity value
                cuda::threshold(gpu_imageRAW, gpu_imageThresh, ILOWV, 1, THRESH_BINARY);

                // compute centroid
                gpu_getCentroidImage(gpu_imageThresh, pos);

                // send via UDP the position
                sendto(sockfd, &pos, sizeof(pos), 0, p->ai_addr, p->ai_addrlen);

                // print position value
                //cout<<count<<"\t"<<pos.x<<"\t"<<pos.y<<endl;

                count++;
            }
        }
    }

    // if chosen mode is "color" ***************************************************
    else if (mode == "color" && argc>2){

        // get number of colors to detect
        numberColors = atoi(argv[4]);

        // if the number of arguments is not exact return, has to be 6*nbColors + 4 arguments
        if(argc != 7*numberColors + 5){
            cout<<"not the correct number of arguments"<<endl;
            return 0;
        }

        // initialize camera settings
        int ret = cameraInit(cap);
        if(!ret)
            return 0;

        // variables for the udp socket
        char* hostMachine = argv[2];    // the third variable is the hostmachine name to send the udp messages
        
        char* port = argv[3];           // the fourth variable is the port

        // create socket
        createSocket(hostMachine, port);

        // initialize the structures
        for (int i=0; i<numberColors; i++){
            colors[i].numColor = i;				// color number
            Scalar lowHSV = Scalar(atoi(argv[7*i + 5]), atoi(argv[7*i + 7]), atoi(argv[7*i + 9]));
            colors[i].lowHSV = lowHSV;
            Scalar highHSV = Scalar(atoi(argv[7*i + 6]), atoi(argv[7*i + 8]), atoi(argv[7*i + 10]));
            colors[i].highHSV = highHSV;
            colors[i].pos = Point(0,0);				// position of detected color
            colors[i].posOld = Point(0,0);
            colors[i].sizeMask = atoi(argv[7*i + 11]);
            colors[i].count = 0;
        }

        int message[3];

        start = duration_cast< milliseconds >( system_clock::now().time_since_epoch() ).count();


        while(!stopRecording){

            // if frame captures
            if(getFrame){
                // store false value
                getFrame.store(false);

                getImageRGB();

                // go through the colors
                for(int i=0; i<numberColors; i++){

                    gpu_thresholdHSV(gpu_imageHSV, colors[i].lowHSV, colors[i].highHSV, gpu_image[i]);

                    gpu_noiseReduction(gpu_image[i], gpu_image[i], colors[i].sizeMask);

                    gpu_getCentroidImage(gpu_image[i], colors[i].pos);

                    // stores in message variable the (x,y) pos of the color. message will be sent to host
                    if(colors[i].pos != Point(0,0) && colors[i].posOld != Point(0,0)){
                        message[0] = colors[i].numColor;
                        message[1] = colors[i].posOld.x;
                        message[2] = colors[i].posOld.y;
                    }
                    else{
                        message[0] = colors[i].numColor;
                        message[1] = 0;
                        message[2] = 0;
                    }

                    // print message
                    cout<<count<<"\tmessage :"<<message[0]<<" "<<message[1]<<" "<<message[2]<<" "<<endl;

                    // send message via udp
                    sendto(sockfd, &message, sizeof(message), 0, p->ai_addr, p->ai_addrlen);

                    if(i==0)count ++;
                    colors[i].posOld = colors[i].pos;
                }
            }
        }
    }

    // if chosen mode is "colorecord" or colorecordtest ***************************************************
    else if ((mode == "colorecord" || mode =="colorecordtest") && argc>2){

        if(mode == "colorecord")
            enableJetsonTest = false;
        else
            enableJetsonTest = true;

        // get number of colors to detect
        numberColors = atoi(argv[4]);

        // if the number of arguments is not exact return, has to be 6*nbColors + 4 arguments
        if(argc != 7*numberColors + 5){
            cout<<"not the correct number of arguments"<<endl;
            return 0;
        }

        char hostname[5];
        gethostname(hostname, HOST_NAME_MAX);
        idCam = hostname;
        transform(idCam.begin(), idCam.end(), idCam.begin(), ::toupper);


        // initialize camera settings
        int ret = cameraInit(cap);
        if(!ret)
            return 0;

        recordInit();

        // variables for the udp socket
        char* hostMachine = argv[2];    // the third variable is the hostmachine name to send the udp messages
        
        char* port = argv[3];           // the fourth variable is the port

        // create socket
        createSocket(hostMachine, port);


        // initialize the structures
        for (int i=0; i<numberColors; i++){
            colors[i].numColor = i;				// color number
            Scalar lowHSV = Scalar(atoi(argv[7*i + 5]), atoi(argv[7*i + 7]), atoi(argv[7*i + 9]));
            colors[i].lowHSV = lowHSV;
            Scalar highHSV = Scalar(atoi(argv[7*i + 6]), atoi(argv[7*i + 8]), atoi(argv[7*i + 10]));
            colors[i].highHSV = highHSV;
            colors[i].pos = Point(0,0);				// position of detected color
            colors[i].posOld = Point(0,0);
            colors[i].sizeMask = atoi(argv[7*i + 11]);
            colors[i].count = 0;
        }

        while(!getFrame){}

        getImageRGB();
        gpu_imageRGB.download(imageRGB);

        char gomsg[3] = "GO";
        // send via UDP the position
        sendto(sockfd, &gomsg, sizeof(gomsg), 0, p->ai_addr, p->ai_addrlen);
        
        cout<<"waiting"<<endl;
        
        while(! okMsg){usleep(100);}

        cout<<"working"<<endl;

        // send via UDP the position
        sendto(sockfd, &hostname, sizeof(hostname), 0, p->ai_addr, p->ai_addrlen);
        
        int message[3];

        start = duration_cast< milliseconds >( system_clock::now().time_since_epoch() ).count();


        while(!stopRecording){

            // if frame captures
            if(getFrame){
                // store false value
                getFrame.store(false);

                getImageRGB();

                // go through the colors
                for(int i=0; i<numberColors; i++){

                    gpu_thresholdHSV(gpu_imageHSV, colors[i].lowHSV, colors[i].highHSV, gpu_image_unoised[i]);

                    gpu_noiseReduction(gpu_image_unoised[i], gpu_image[i], colors[i].sizeMask);

                    gpu_getCentroidImage(gpu_image[i], colors[i].pos);

                    startRecord = true;


                    if(colors[i].pos != Point(0,0) && colors[i].posOld != Point(0,0)){
                        message[0] = colors[i].numColor;
                        message[1] = colors[i].posOld.x;
                        message[2] = colors[i].posOld.y;
                    }
                    else{
                        message[0] = colors[i].numColor;
                        message[1] = 0;
                        message[2] = 0;
                    }

                    // print message
                    cout<<count<<"\tmessage :"<<message[0]<<" "<<message[1]<<" "<<message[2]<<" "<<endl;

                    // send message via udp
                    sendto(sockfd, &message, sizeof(message), 0, p->ai_addr, p->ai_addrlen);

                    if(i==0)count ++;
                    colors[i].posOld = colors[i].pos;
                }
            }
        }
    }

    else if(mode == "record"){
        // initialize camera settings
        int ret = cameraInit(cap);
        if(!ret)
            return 0;

        string outDir = "/home/nvidia/Desktop/JetsonData/";

        char hostname[5];
        gethostname(hostname, HOST_NAME_MAX);
        idCam = hostname;
        transform(idCam.begin(), idCam.end(), idCam.begin(), ::toupper);

        // variables for the udp socket
        char* hostMachine = argv[2];    // the third variable is the hostmachine name to send the udp messages
        
        char* port = argv[3];           // the fourth variable is the port

        // create socket
        createSocket(hostMachine, port);

        // Get resolution and framerate from capture
        unsigned int width = cap.get (cv::CAP_PROP_FRAME_WIDTH);
        unsigned int height = cap.get (cv::CAP_PROP_FRAME_HEIGHT);
        unsigned int fps = 30;
        float intervalFrameMicrosec = 1000000/fps;

        cout<<fps<<endl;


        VideoWriter gst_nvh264_writer("appsrc ! queue ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc bitrate=16000000 insert-sps-pps=true ! h264parse ! qtmux ! filesink location=" + outDir + idCam + ".mp4", CAP_GSTREAMER,  0, fps, cv::Size (width, height));
        
        if (!gst_nvh264_writer.isOpened ()) {
            std::cout << "Failed to open gst_nvh264 writer." << std::endl;
            return (-6);
        }

        getImageRGB();
        gpu_imageRGB.download(imageRGB);

        char gomsg[3] = "GO";
        // send via UDP the position
        sendto(sockfd, &gomsg, sizeof(gomsg), 0, p->ai_addr, p->ai_addrlen);
        
        while(! okMsg){}

        cout<<"working"<<endl;

        // send via UDP the position
        sendto(sockfd, &hostname, sizeof(hostname), 0, p->ai_addr, p->ai_addrlen);

        start = duration_cast< milliseconds >( system_clock::now().time_since_epoch() ).count();

        while(!stopRecording){
            // if frame captures
            elapsedTemp = duration_cast< microseconds >( system_clock::now().time_since_epoch() ).count();

            if(getFrame && (elapsedTemp - elapsedStart) >= intervalFrameMicrosec ){
                //cout<<elapsedTemp-elapsedStart<<endl;

                elapsedStart = duration_cast< microseconds >( system_clock::now().time_since_epoch() ).count();

                // store false value
                getFrame.store(false);

                getImageRGB();

                gpu_imageRGB.download(imageRGB);

                gst_nvh264_writer.write(imageRGB);

                count ++;

            }
        }

        gst_nvh264_writer.release();
    }

    else{
        cout<<"No mode chosen"<<endl;
        return 0;
    }


    // find the ending time
    end = duration_cast< milliseconds >( system_clock::now().time_since_epoch() ).count();

    cout<<"avant fin"<<endl;

    // waits for the thread to be closed
    // the variable "stopRecording" closes those threads
    pthread_join(thread_cam, NULL);
    pthread_join(thread_record, NULL);
    pthread_join(thread_udp, NULL);

    cap.release();

    // here the program prints your frame rate exits
    long int millisecs = end - start;
    cout << "Time taken : " << millisecs << " milliseconds" << endl;

    return 0;
}





