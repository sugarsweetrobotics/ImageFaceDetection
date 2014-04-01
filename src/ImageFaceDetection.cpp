// -*- C++ -*-
/*!
 * @file  ImageFaceDetection.cpp
 * @brief Face Detection OpenCV Image Processing Component
 * @date $Date$
 *
 * $Id$
 */

#include "ImageFaceDetection.h"

enum IMAGE_FORMAT {
  FMT_GRAY,
  FMT_RGB,
  FMT_JPEG,
  FMT_PNG,
};

bool convertCvMatToImg(const cv::Mat& srcImage, Img::CameraImage& dstImage, const IMAGE_FORMAT outFormat, const int compression_ratio=75) {
  int width = srcImage.cols;
  int height = srcImage.rows;
  int depth = srcImage.depth();
  int inChannels = srcImage.channels();
  int outChannels = outFormat == FMT_GRAY ? 1 : 3;

  cv::Mat procImage;
  if (outChannels > inChannels) {
    cv::cvtColor(srcImage, procImage, CV_GRAY2RGB);
  } else if (outChannels < inChannels) {
    cv::cvtColor(srcImage, procImage, CV_RGB2GRAY);
  } else {
    procImage = srcImage;
  }

  dstImage.image.width = width;
  dstImage.image.height = height;

  switch(outFormat) {
  case FMT_RGB:
    dstImage.image.format = Img::CF_RGB;
    dstImage.image.raw_data.length( width * height * outChannels );
    for(int i = 0;i < height;i++) {
      memcpy(&(dstImage.image.raw_data[i * width * outChannels]),
	     &(procImage.data[i * procImage.step]),
	     width * outChannels);
    }
    break;
  case FMT_JPEG:
    {
      dstImage.image.format = Img::CF_JPEG;
      //Jpeg encoding using OpenCV image compression function
      std::vector<int> compression_param = std::vector<int>(2); 
      compression_param[0] = CV_IMWRITE_JPEG_QUALITY;
      compression_param[1] = compression_ratio;
      //Encode raw image data to jpeg data
      std::vector<uchar> compressed_image;
      cv::imencode(".jpg", procImage, compressed_image, compression_param);
      //Copy encoded jpeg data to Outport Buffer
      dstImage.image.raw_data.length(compressed_image.size());
      memcpy(&dstImage.image.raw_data[0], &compressed_image[0], sizeof(unsigned char) * compressed_image.size());
    }
    break;
  case FMT_PNG:
    {
      dstImage.image.format = Img::CF_PNG;
      //Jpeg encoding using OpenCV image compression function
      std::vector<int> compression_param = std::vector<int>(2); 
      compression_param[0] = CV_IMWRITE_PNG_COMPRESSION;
      compression_param[1] = (int)((double)compression_ratio/10.0);
      if(compression_param[1] == 10)
	compression_param[1] = 9;
      //Encode raw image data to jpeg data
      std::vector<uchar> compressed_image;
      cv::imencode(".png", procImage, compressed_image, compression_param);
      //Copy encoded jpeg data to Outport Buffer
      dstImage.image.raw_data.length(compressed_image.size());
      memcpy(&dstImage.image.raw_data[0], &compressed_image[0], sizeof(unsigned char) * compressed_image.size());
    }
    break;
  case FMT_GRAY:
    {
      dstImage.image.format = Img::CF_GRAY;
      dstImage.image.raw_data.length( width * height * outChannels);
      for(int i(0); i< height; ++i) {
	memcpy(&(dstImage.image.raw_data[i * width * outChannels]),
	       &(procImage.data[i * procImage.step]),
	       width * outChannels);
      }
    }
    break;
  default:
    return false;
  }
  return  true;
}

bool convertImgToCvMat(const Img::CameraImage& srcImage, cv::Mat& dstImage) {
  int channels = 1;
  int width = srcImage.image.width;
  int height = srcImage.image.height;
  int format = srcImage.image.format;
  int data_length = srcImage.image.raw_data.length();
  int image_size = width * height * channels;
  
  switch(format) {
  case Img::CF_GRAY:
    channels = 1;
    break;
  case Img::CF_RGB:
  case Img::CF_PNG:
  case Img::CF_JPEG:
    channels = 3;
    break;
  default:
    channels = (srcImage.image.raw_data.length()/width/height);
  }
  
  if (channels == 3) {
    dstImage.create(height, width, CV_8UC3);
  } else {
    dstImage.create(height, width, CV_8UC1);
  }
  
  switch(format) {
  case Img::CF_RGB:
    {
      for(int i = 0; i < height;i++) {
	memcpy(&dstImage.data[i*dstImage.step], 
	       &srcImage.image.raw_data[i*width*channels],
	       sizeof(unsigned char)*width*channels);
      }
      if(channels == 3) {
	//cv::cvtColor(dstImage, dstImage, CV_RGB2BGR);
      }
    }
    break;
  case Img::CF_JPEG:
  case Img::CF_PNG:
    {
      std::vector<uchar> compressed_image(data_length);
      memcpy(&compressed_image[0], &srcImage.image.raw_data[0], sizeof(unsigned char) * data_length);
      
      //Decode received compressed image
      cv::Mat decoded_image;
      if(channels == 3) {
	decoded_image = cv::imdecode(cv::Mat(compressed_image), CV_LOAD_IMAGE_COLOR);
	//cv::cvtColor(decoded_image, dstImage, CV_RGB2BGR);
      }
      else {
	decoded_image = cv::imdecode(cv::Mat(compressed_image), CV_LOAD_IMAGE_GRAYSCALE);
	dstImage = decoded_image;
      }
    }
    break;
  default:
    return false;
  }
  return true;
}


// Module specification
// <rtc-template block="module_spec">
static const char* imagefacedetection_spec[] =
  {
    "implementation_id", "ImageFaceDetection",
    "type_name",         "ImageFaceDetection",
    "description",       "Face Detection OpenCV Image Processing Component",
    "version",           "1.0.0",
    "vendor",            "Sugar Sweet Robotics",
    "category",          "Experimental",
    "activity_type",     "PERIODIC",
    "kind",              "DataFlowComponent",
    "max_instance",      "1",
    "language",          "C++",
    "lang_type",         "compile",
    // Configuration variables
    "conf.default.data_dir", "/usr/local/share/OpenCV/haarcascades",
    // Widget
    "conf.__widget__.data_dir", "text",
    // Constraints
    ""
  };
// </rtc-template>

/*!
 * @brief constructor
 * @param manager Maneger Object
 */
ImageFaceDetection::ImageFaceDetection(RTC::Manager* manager)
    // <rtc-template block="initializer">
  : RTC::DataFlowComponentBase(manager),
    m_inIn("in", m_in),
    m_outOut("out", m_out),
    m_positionOut("position", m_position)

    // </rtc-template>
{
}

/*!
 * @brief destructor
 */
ImageFaceDetection::~ImageFaceDetection()
{
}



RTC::ReturnCode_t ImageFaceDetection::onInitialize()
{
  // Registration: InPort/OutPort/Service
  // <rtc-template block="registration">
  // Set InPort buffers
  addInPort("in", m_inIn);
  
  // Set OutPort buffer
  addOutPort("out", m_outOut);
  addOutPort("position", m_positionOut);
  
  // Set service provider to Ports
  
  // Set service consumers to Ports
  
  // Set CORBA Service Ports
  
  // </rtc-template>

  // <rtc-template block="bind_config">
  // Bind variables and configuration variable
  bindParameter("data_dir", m_data_dir, "/usr/local/share/OpenCV/haarcascades");
  // </rtc-template>
  
  return RTC::RTC_OK;
}

/*
RTC::ReturnCode_t ImageFaceDetection::onFinalize()
{
  return RTC::RTC_OK;
}
*/

/*
RTC::ReturnCode_t ImageFaceDetection::onStartup(RTC::UniqueId ec_id)
{
  return RTC::RTC_OK;
}
*/

/*
RTC::ReturnCode_t ImageFaceDetection::onShutdown(RTC::UniqueId ec_id)
{
  return RTC::RTC_OK;
}
*/


RTC::ReturnCode_t ImageFaceDetection::onActivated(RTC::UniqueId ec_id)
{
  if (!m_FaceCascade.load(m_data_dir + "/haarcascade_frontalface_alt.xml")) {
    RTC_ERROR(("Load FaceCascade Failed(%s/%s)", m_data_dir.c_str(), "haarcascade_frontalface_alt.xml"));
    return RTC::RTC_ERROR;
  }
  if (!m_EyeCascade.load(m_data_dir + "/haarcascade_eye_tree_eyeglasses.xml")) {
    RTC_ERROR(("Load Eye Cascade Failed."));
    return RTC::RTC_ERROR;
  }
  return RTC::RTC_OK;
}



RTC::ReturnCode_t ImageFaceDetection::onDeactivated(RTC::UniqueId ec_id)
{
  return RTC::RTC_OK;
}



RTC::ReturnCode_t ImageFaceDetection::onExecute(RTC::UniqueId ec_id)
{
  if (m_inIn.isNew()) {
    m_inIn.read();
    if (convertImgToCvMat(m_in.data, m_srcImage)) {
      std::vector<cv::Rect> faces;

      cv::cvtColor(m_srcImage, m_grayImage, CV_BGR2GRAY );
      cv::equalizeHist(m_grayImage, m_grayImage);

      //-- Detect faces
      m_FaceCascade.detectMultiScale(m_grayImage,
				     faces,
				     1.1,
				     2,
				     0|CV_HAAR_SCALE_IMAGE,
				     cv::Size(30, 30));

      for(size_t i = 0;i < faces.size();i++) {
	cv::Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
	cv::ellipse(m_srcImage, center, cv::Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );
      }
      convertCvMatToImg(m_srcImage, m_out.data, FMT_RGB);

      setTimestamp(m_out);
      m_out.data.captured_time = m_out.tm;
      m_outOut.write();


	/**
	Mat faceROI = m_grayImage( faces[i] );
	std::vector<cv::Rect> eyes;
	eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

	for( size_t j = 0; j < eyes.size(); j++ ) {
	Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
	int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
	circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
	}
	}
	//-- Show what you got
	imshow( window_name, frame );
	**/

    }
  }
  return RTC::RTC_OK;
}

/*
RTC::ReturnCode_t ImageFaceDetection::onAborting(RTC::UniqueId ec_id)
{
  return RTC::RTC_OK;
}
*/

/*
RTC::ReturnCode_t ImageFaceDetection::onError(RTC::UniqueId ec_id)
{
  return RTC::RTC_OK;
}
*/

/*
RTC::ReturnCode_t ImageFaceDetection::onReset(RTC::UniqueId ec_id)
{
  return RTC::RTC_OK;
}
*/

/*
RTC::ReturnCode_t ImageFaceDetection::onStateUpdate(RTC::UniqueId ec_id)
{
  return RTC::RTC_OK;
}
*/

/*
RTC::ReturnCode_t ImageFaceDetection::onRateChanged(RTC::UniqueId ec_id)
{
  return RTC::RTC_OK;
}
*/



extern "C"
{
 
  void ImageFaceDetectionInit(RTC::Manager* manager)
  {
    coil::Properties profile(imagefacedetection_spec);
    manager->registerFactory(profile,
                             RTC::Create<ImageFaceDetection>,
                             RTC::Delete<ImageFaceDetection>);
  }
  
};


