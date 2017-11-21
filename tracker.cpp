class CTracker
{
public:
  CTracker():p_parameters(0),p_model(0)
  {}
  ~CTracker()
  {
    Close();
  }

  bool Init();
  bool DetectLandmarks(cv::Mat_<uchar> &grayscale_image);
  bool GetPose(double &s, double &tx, double &ty, double &e1, double &e2, double &e3);
  bool GetLandmarks3D(vector<double> &landmarks_3d);
  bool GetLandmarks2D(vector<double> &landmakrs_2d);
  double GetScore();
  bool Close();
public:
  LandmarkDetector::FaceModelParameters *p_parameters;
  LandmarkDetector::CLNF *p_model;
  double m_score;
};


bool CTracker::Init()
{
  p_parameters=new LandmarkDetector::FaceModelParameters();
  p_parameters->track_gaze=true;
  p_model=new LandmarkDetector::CLNF(p_parameters->model_localtion);
  return true;
}

bool Close()
{
  if(p_parameters) delete p_parameters;
  if(p_model) delete p_model;

  p_parameters=0;
  p_model=0;
  return true;
}

bool CTracker::DetectLandmarks(cv::Mat_<uchar> &grayscale_image)
{
  cv::Mat_<uchar> depth_image;
  return LandmarkDetector::DetectLandmarksInVideo(grayscale_image,depth_image,*p_model,*p_parameters);
}

double CTracker::GetScore()
{
  return p->model->detection_certainty;
}

bool CTracker::GetPose(double &s, double &tx, double &ty, double &e1, double &e2, double &e3)
{
  auto &g=p_model->params_global;
  s=g[0];
  e1=g[1];
  e2=g[2];
  e3=g[3];
  tx=g[4];
  ty=g[5];
  return true;
}

bool CTracker::GetLandmarks3D(vector<double> &landmarks_3d)
{
  int n=p_model->detected_landmarks.rows/2;
  cv::Mat_<double> shape3d(n*3,1);
  p_model->pdm.CalcShape3D(shape3d, p_model->params_local);

  //x
  for(int i=0; i<n; i++)
  {
    double x=shape3d.at<double>(i)+1;
    landmarks_3d.push_back(x);
  }

  //y
  for(int i=0; i<n; i++)
  {
    double y=shape3d.at<double>(i+n)+1;
    landmarks_3d.push_back(y);
  }

  //z
  for(int i=0; i<n; i++)
  {
    double z=shape3d.at<double>(i+2*n)+1;
    landmarks_3d.push_back(z);
  }

  return true;
}

bool CTracker::GetLandmarks2D(vector<double> &landmarks_2d)
{
  auto &marks=p_model->detected_landmarks;

  int n=marks.rows/2;

  //x
  for(int i=0; i<n; i++)
  {
    double x=marks.at<double>(i)+1;
    landmarks_2d.push_back(x);
  }

  //y
  for(int i=0; i<n; i++)
  {
    double y=marks.at<double>(i+n)+1;
    landmarks_2d.push_back(y);
  }

  return true;
}
