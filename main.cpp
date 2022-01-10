#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
//容器vector的头文件，vector是可以存放任意类型的动态数组
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include <thread>
//用于显示3D视觉图像
#include <pangolin/pangolin.h>
//Linux系统服务头文件
#include <unistd.h>
#include <cmath>
#include <mutex>
#define PI acos(-1)


int main(/*int argc, char* argv[]*/)
{  
  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main",640,480);
  
  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
    pangolin::ModelViewLookAt(-0,0.5,-3, 0,0,0, pangolin::AxisY)
  );

  // Choose a sensible left UI Panel width based on the width of 20
  // charectors from the default font.
  const int UI_WIDTH = 400;

  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, 640.0f/480.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  // Add named Panel and bind to variables beginning 'ui'
  // A Panel is just a View with a default layout and input handling
  pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

  // Safe and efficient binding of named variables.
  // Specialisations mean no conversions take place for exact types
  // and conversions between scalar types are cheap.
  pangolin::Var<bool> a_button("ui.A_Button",false,false);
  pangolin::Var<double> a_double("ui.A_Double",3,0,5);
  pangolin::Var<int> an_int("ui.An_Int",2,0,5);
  pangolin::Var<double> a_double_log("ui.Log_scale",3,1,1E4, true);
  pangolin::Var<bool> a_checkbox("ui.A_Checkbox",false,true);
  pangolin::Var<int> an_int_no_input("ui.An_Int_No_Input",2);
  pangolin::Var<std::string> a_string("ui.A_String", "Edit ME!");

  // std::function objects can be used for Var's too. These work great with C++11 closures.
  pangolin::Var<std::function<void(void)>> save_window("ui.Save_Window", [](){
      pangolin::SaveWindowOnRender("window");
  });

  pangolin::Var<std::function<void(void)>> save_cube_view("ui.Save_Cube", [&d_cam](){
      pangolin::SaveWindowOnRender("cube", d_cam.v);
  });

  // Demonstration of how we can register a keyboard hook to alter a Var
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'b', [&](){
      a_double = 3.5;
  });

  // Default hooks for exiting (Esc) and fullscreen (tab).
  while( !pangolin::ShouldQuit() )
  {
    // Clear entire screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    

    if( pangolin::Pushed(a_button) )
      std::cout << "You Pushed a button!" << std::endl;

    // Overloading of Var<T> operators allows us to treat them like
    // their wrapped types, eg:
    if( a_checkbox )
      an_int = (int)a_double;

    an_int_no_input = an_int;

    if(d_cam.IsShown()) {
        // Activate efficiently by object
        d_cam.Activate(s_cam);

        // Render some stuff
        glColor3f(1.0,1.0,1.0);
        pangolin::glDrawColouredCube();
    }

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }

  return 0;
}


// int main( int /*argc*/, char** /*argv*/ )
// {
//     pangolin::CreateWindowAndBind("Main",1280,960);
//     glEnable(GL_DEPTH_TEST);

//     // Define Projection and initial ModelView matrix
//     pangolin::OpenGlRenderState s_cam(
//         pangolin::ProjectionMatrix(320,240,420,420,320,240,0.2,100),
//         pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
//     );

//     pangolin::OpenGlRenderState s_cam2(
//         pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
//         pangolin::ModelViewLookAt(0,0,5, 0,0,0, pangolin::AxisY)
//     );

//     // Create Interactive View in window
//     pangolin::View& d_cam = pangolin::CreateDisplay()
//             .SetBounds(0.0, 0.5, 0.0, 0.5, -640.0f/480.0f)
//             .SetHandler(new pangolin::Handler3D(s_cam));

//     pangolin::View& d_cam2 = pangolin::CreateDisplay()
//             .SetBounds(0.5, 1, 0.5, 1, -640.0f/480.0f)
//             .SetHandler(new pangolin::Handler3D(s_cam2));



//     while( !pangolin::ShouldQuit() )
//     {
//         // Clear screen and activate view to render into
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//         // d_cam.Activate(s_cam);

//         glClearColor(1.0,1.0,1.0,1.0);

//         d_cam2.Activate(s_cam2);
//         glColor3f(1,0,0);
//         glBegin(GL_LINE_STRIP);
//         glVertex2d(0.0,0.0);
//         glVertex2d(0.0,1.0);
//         glVertex2d(0.1,0.1);
//         glVertex2d(0.2,0.2);
//         glVertex2d(0.3,0.3);
//         glEnd();
//         // glVertex2d(1,0);
//         // glVertex2d(-0.5,-0.5);
//         d_cam.Activate(s_cam);
//         // Render OpenGL Cube
//         // glColor3f(0,0,1);
//         pangolin::glDrawColouredCube();

        
//         // Swap frames and Process Events
//         pangolin::FinishFrame();
//     }

//     return 0;
// }

// class A{
// public:
//     A(){
//         k=3;
//     }
//     int& f(){
//         return k;
//     }
//     void print(){
//         std::cout<<k<<std::endl;
//     }
// private:
//     int k;

// };


// int main(){
//     std::cout<<2/3.0<<std::endl;

//     return 0;
// }

// class RandomVector{

// public:
//     RandomVector(){
//         hat = Eigen::Vector3d::Zero(3);
//         P = Eigen::Matrix3d::Zero(3,3);
//     }

//     RandomVector(Eigen::VectorXd hat_, Eigen::MatrixXd P_):hat(hat_),P(P_){}

//     void Print(){
//         std::cout<<"hat:"<<hat<<std::endl<<"P: "<<P<<std::endl;
//         return ;
//     }
    
//     RandomVector compound(RandomVector b){
//         Eigen::VectorXd ij = this->hat;
//         Eigen::VectorXd jk = b.hat;
//         this->hat = Eigen::Vector3d(jk(0)*cos(ij(2))-jk(1)*sin(ij(2))+ij(0),
//                             jk(0)*sin(ij(2))+jk(1)*cos(ij(2))+ij(1),
//                             ij(2)+jk(2));
//         Eigen::VectorXd ik = this->hat;
//         Eigen::MatrixXd J1(3,3),J2(3,3);
//         J1 << 1,0,-(ik(1)-ij(1)),
//                 0,1,(ik(0)-ij(0)),
//                 0,0,1;
//         J2 << cos(ij(2)), -sin(ij(2)),0,
//                 sin(ij(2)), cos(ij(2)),0,
//                 0,0,1;
//         this->P = J1*this->P*J1.transpose()+J2*b.P*J2.transpose();
//         return *this;
//     }
//     RandomVector compoundP(RandomVector b){
//         Eigen::VectorXd oi = this->hat;
//         Eigen::VectorXd ip = b.hat;
//         this->hat = Eigen::Vector2d(oi(0)+ip(0)*cos(oi(2))-ip(1)*sin(oi(2)),
//                             oi(1)+ip(0)*sin(oi(2))+ip(1)*cos(oi(2)));
//         Eigen::MatrixXd J1(2,3),J2(2,2);
//         J1 << 1,0,-ip(1)*sin(oi(2))-ip(1)*cos(oi(2)),
//                 0,1,ip(0)*cos(oi(2))-ip(1)*sin(oi(2));
//         J2 << cos(oi(2)), -sin(oi(2)),
//                 sin(oi(2)), cos(oi(2));
//         // std::cout<<"compoundP \n"<<this->P;
//         this->P = J1*this->P*J1.transpose()+J2*b.P*J2.transpose();
//         return *this;
//     }
//     static RandomVector CompoundP(RandomVector q, RandomVector b){
//         Eigen::VectorXd oi = q.hat;
//         Eigen::VectorXd ip = b.hat;
//         Eigen::VectorXd hat = Eigen::Vector2d(oi(0)+ip(0)*cos(oi(2))-ip(1)*sin(oi(2)),
//                             oi(1)+ip(0)*sin(oi(2))+ip(1)*cos(oi(2)));
//         Eigen::MatrixXd J1(2,3),J2(2,2);
//         J1 << 1,0,-ip(1)*sin(oi(2))-ip(1)*cos(oi(2)),
//                 0,1,ip(0)*cos(oi(2))-ip(1)*sin(oi(2));
//         J2 << cos(oi(2)), -sin(oi(2)),
//                 sin(oi(2)), cos(oi(2));
//         // std::cout<<"compoundP \n"<<this->P;
//         Eigen::Matrix<double, 2, 2> P = J1*q.P*J1.transpose()+J2*b.P*J2.transpose();
//         std::cout<<"what's wrong"<<std::endl;
//         RandomVector ret(hat,P);
//         return ret;
//     }
//     RandomVector rinverse(){
//         Eigen::VectorXd x = this->hat;
//         this->hat << -x(0)*cos(x(2))-x(1)*sin(x(2)),x(0)*sin(x(2))-x(1)*cos(x(2)),-x(2);
//         Eigen::MatrixXd J(3,3);
//         J<< -cos(x(2)),-sin(x(2)),x(1),
//             sin(x(2)),-cos(x(2)),-x(0),
//             0,0,-1;
//         this-> P = J*this->P*J.transpose();
//         // std::cout<<"reverse OK"<<std::endl;
//         return *this;
//     }

//     RandomVector tail2tail(RandomVector b){
//         return (this->rinverse()).compound(b);
//     }
//     bool operator ==(const RandomVector& b){
//         return (this->hat).isApprox(b.hat, 1e-5) && (this->P).isApprox(b.P);
//     }
//     Eigen::VectorXd hat;
//     Eigen::MatrixXd P;
// };

// typedef RandomVector point, pose, motion, error;

// std::vector<point>sonarPoints;
// std::vector<point>distortedPoints;
// std::vector<pose>poses;
// cv::Mat cameraImg;


// void initSonarPoints(){
//     sonarPoints.resize(100);
//     int i = 0;
//     for(auto&p: sonarPoints){
//         p.hat(0)=50.0*rand()/RAND_MAX;
//         p.hat(1)=0.03*(i++);
//         p.hat(2)=rand()*128;
//     }
// }

// int main(){
//     initSonarPoints();

//     pangolin::CreateWindowAndBind("Main",1024,768);
//     glEnable(GL_DEPTH_TEST);
//     glEnable(GL_BLEND);
//     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//     // pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(320,240,200,200,160,120,0.1,1000);
//     pangolin::OpenGlRenderState s_cam1(
//             pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
//             pangolin::ModelViewLookAt(0, 0, 10, 0, 0, 0, pangolin::AxisY)
//     );

//     pangolin::OpenGlRenderState s_cam2(
//             pangolin::ProjectionMatrix(640,480,160,160,320,480,0.1,1000),
//             pangolin::ModelViewLookAt(0, 0, 10, 0, 0, 0, pangolin::AxisY)
//     );

//     pangolin::View& d_cam1 = pangolin::Display("cam1")
//         .SetAspect(640.0f/480.0f)
//         .SetHandler(new pangolin::Handler3D(s_cam1));

//     pangolin::View& d_cam2 = pangolin::Display("img1")
//         .SetAspect(640.0f/480.0f);

//     pangolin::View& d_img2 = pangolin::Display("img2")
//         .SetAspect(640.0f/480.0f);

//     pangolin::View& d_img3 = pangolin::Display("img3")
//         .SetAspect(640.0f/480.0f);

//     // pangolin::View& d_img4 = pangolin::Display("img4")
//     //     .SetAspect(640.0f/480.0f);

//     pangolin::Display("multi")
//         .SetBounds(0.0, 1.0, 0.0, 1.0)
//         .SetLayout(pangolin::LayoutEqual)
//         .AddDisplay(d_cam1)
//         .AddDisplay(d_cam2)
//         .AddDisplay(d_img2)
//         .AddDisplay(d_img3);
//         // .AddDisplay(d_img4);
    
//     const int width =  64;
//     const int height = 48;


//     while( !pangolin::ShouldQuit() )
//     {
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//         // d_img4.Activate();
//         // 画图像处理后的数据
//         //

//         // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
//         d_cam1.Activate(s_cam1);
//         glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
//         glColor3f(0.29f, 0.71f, 1.0f);
//         //画位姿
//         pangolin::glDrawColouredCube();

//         d_cam2.Activate(s_cam2);
//         //画声纳源数据
//         glBegin(GL_LINES);
//         for(point &p:sonarPoints){
//             double r=p.hat(0),theta=p.hat(1);
//             glColor3f(0.4f,0.4f,0.2f);
//             glPointSize(4);
//             glVertex2d(r*cos(theta),r*sin(theta));
//             // std::cout<<r<<","<<theta<<","<<r*cos(theta)<<","<<r*sin(theta)<<std::endl;
//         }
//         glEnd();


//         // d_img2.Activate();
//         //画声纳处理后的数据
//         //
//         d_img3.Activate();
//         glClearColor(1.0f,1.0f,1.0f,1.0f);
//         //画图像源数据
//         cameraImg = cv::imread("../1.jpg");
//         pangolin::GlTexture cameraImgTexture(260/* d */, 260, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);
//         cameraImgTexture.Upload(cameraImg.data, GL_BGR, GL_UNSIGNED_BYTE);
//         glColor3f(1.0f, 1.0f, 1.0f);
//         cameraImgTexture.RenderToViewportFlipY();

//         pangolin::FinishFrame();
//     }

//     return 0;
// }

// class A{
// public:
//     A(){
//         X.resize(3);
//         X[0] = 0;
//         X[1] = 1;
//         X[2] = 2;
//         return ;
//     }
//     std::vector<int> getX(){
//         return X;
//     }

// private:
//     std::vector<int> X;
// };


// int main(){
//     A a;
//     std::vector<int> c = a.getX();
//     for(int i=0;i<c.size();i++){
//         std::cout<<c[i]<<std::endl;
//     }
//     return 0;
// }

// class point{
// public:
//     point(int a,int b){
//         x=a;
//         y=b;
//     }
//     int Getx(){
//         return x;
//     }
//     int Gety(){
//         return y;
//     }
//     int x;
//     int y;

// };

// class test{
// public:
//     test(point a):(c.x(a.Getx())),(c.y(a.Gety())){}
//     void print(){
//         std::cout<<a1<<":"<<a2<<std::endl;
//         return ;
//     }
// private:
//     point c;
//     int a1;
//     int a2;
// };


// int main(){
//     point pp(1,2);
//      test t(pp);
//     t.print();
//     return 0;
// }


// double MahDistance(const Eigen::Vector3d &ri, const Eigen::Matrix3d &Sigma, const Eigen::Vector3d &ni){
// /*
// 输入：向量ri,协方差矩阵，向量ni
// 输出：ri和ni的马氏距离
// */
//     std::cout<<ri<<"*"<<Sigma<<"*"<<ni<<std::endl;
//     return sqrt((ri - ni).transpose() * Sigma.inverse() * (ri - ni));
// }

// int main(){
//     Eigen::Vector3d a(1,2,3);
//     Eigen::Vector3d b(3,2,1);
//     Eigen::Matrix3d sigma = Eigen::MatrixXd::Identity(3,3);
//     std::cout<<MahDistance(a,sigma,b)<<std::endl;
// }

// int * k;

// void f1(){
//     int a = 2;
//     k=&a;
//     return ;
// }


// void f2(){
//     int a =3;
//     return ;
// }


// int main(){
//     std::vector<int> a;
//     a.push_back(1);
//     a.push_back(3);
//     a.push_back(4);
//     a.erase(1);
    
// }

// int main(){
//     f1();
//     f2();
//     Eigen::VectorXd a;
//     a.resize(8);
//     a<< -1.18542e-06,-5.59947e-07,3.44496e-06,8.13833e-07,-1.30002e-05,-6.14076e-06, 2.50595e-05, 6.71983e-06;

//     Eigen::VectorXd x_stmp = a;
//     Eigen::Vector3d x_shat = Eigen::Vector3d(x_stmp(0),x_stmp(1),x_stmp(3));

//     std::cout<<"from "<<a<<std::endl<<"extract "<<x_shat<<std::endl;
//     // std::cout<<a<<std::endl;
//     std::cout<<(double)0/1e9<<std::endl;
// }

// class A{
// public:
//     A(){
//         scan.resize(0);
//     }
//     void show(){
//         std::cout<<"hello"<<std::endl;
//     }
//     void Append(Eigen::VectorXd a){
//         scan.push_back(a);
//     }
// private:
//     std::vector<Eigen::VectorXd>scan;
// };

// int main(){
//     A a;
//     Eigen::VectorXd val,val2;
//     val.resize(12);
//     val << 1,1,2,2,3,4,4,5,5,6,6,7;
//     val2.resize(4);
//     val2<< 2,3,3,1;
//     a.Append(val);
//     a.Append(val2);
//     a.Clasd();
//     std::cout<<"No problem"<<std::endl;
//     return 0;
// }


// //B的private对象A能用指针访问吗？可以
// class B{
// public:
//     B(){
//         a=3;
//     }
//     int* get(){
//         return &a;
//     }

// private:
//     int a;
// };


// class A{
// public:
//     A(){
//         mpB = new B();
//         mpBthread = new std::thread(&B::loopCall,mpB);
//         //mpBthread->join();
//         //mpBthread->detach();
//         while(true){
//             std::cout<<"111"<<std::endl;
//         }
//     }
// private:
//     B* mpB;
//     std::thread* mpBthread;
// };

// int* p;

// class A{
// public:
//     A(){
//         val = 3;
//         p = &val;
//     }

// private:
//     int val;
// };

// class B{
// public:
//     B(){
//         std::cout<<*p<<std::endl;
//     }
// private:
//     int cc;
// };

// void f1(){
//     A a;
//     return;
// }

// void f2(){
//     p = new int(10);
// }

// int main(){
//     f1();
//     f2();
//     B b;
//     return 0;
// }

// int main(){
//     YAML::Node conf = YAML::LoadFile("../config.yaml");
//     B b;
//     std::cout<<*(b.get())<<std::endl;
//     return 0;
// }

// class B{
// public:
//     B(){
//     }

//     void loopCall(){
//         while(true){
//             std::cout<<"hello"<<std::endl;
//             usleep(1000);
//         }
//     }
// };


// class A{
// public:
//     A(){
//         mpB = new B();
//         mpBthread = new std::thread(&B::loopCall,mpB);
//         //mpBthread->join();
//         //mpBthread->detach();
//         while(true){
//             std::cout<<"111"<<std::endl;
//         }
//     }
// private:
//     B* mpB;
//     std::thread* mpBthread;
// };




// int main(){
//     A a;
//     return 0;
// }


// Eigen::Vector3d state(0,0,0);
// cv::Mat cameraMatrix=(cv::Mat_<double>(3,3)<< 707.0912,0,601.8873,0,707.0912,183.1104,0,0,1);


// bool isRotationMatrix(cv::Mat &R)
// {
//     cv::Mat Rt;
//     cv::transpose(R, Rt);
//     cv::Mat shouldBeIdentity = Rt * R;
//     cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    
//     return  norm(I, shouldBeIdentity) < 1e-6;
// }

// void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud);

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
// cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
// {

//     assert(isRotationMatrix(R));
//     float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
//     bool singular = sy < 1e-6;
//     float x, y, z;
//     if (!singular)
//     {
//         x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
//         y = atan2(-R.at<double>(2,0), sy);
//         z = atan2(R.at<double>(1,0), R.at<double>(0,0));
//     }
//     else
//     {
//         x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
//         y = atan2(-R.at<double>(2,0), sy);
//         z = 0;
//     }
//     return cv::Vec3f(x, y, z);
// }


// bool Track(){
//     if(last_frame_){
//         current_frame_->SetPose(relative_motion * last_frame_->Pose);
//     }
//     int num_track_left = TrackLastFrame();
//     tracking_inliners_ = EstimateCurrentPose();
//     if(tracking_inliers_ > num_features_tracking_){
//         std::cout<<"Traking Good";
//     }else if(tracking_inliers_ > num_features_tracking_bad_){
//         std::cout<<"Tracking Bad";
//     }else{
//         std::cout<<"Lost";
//     }
//     relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();
//     return true;
// }

// input: img1,img2
// output: 帧间运动
//

// int CalcMotion(std::vector<cv::Point2f>& pts1,std::vector<cv::Point2f>& pts2, cv::Mat& R, cv::Mat& t){
//     // int ptCount = pts1.size();
// //     cv::Mat *p1 = cv::cvCreateMat(ptCount, 2, CV_32F);
// //     cv::Mat p2(ptCount, 2, CV_32F);
 
// // // 把Keypoint转换为Mat
// //     cv::Point2f pt;
// //     for (int i=0; i<ptCount; i++)
// //     {
// //         pt = pts1[i];
// //         p1.at<float>(i, 0) = pt.x;
// //         p1.at<float>(i, 1) = pt.y;
    
// //         pt = pts2[i];
// //         p2.at<float>(i, 0) = pt.x;
// //         p2.at<float>(i, 1) = pt.y;
// //     }
//     std::cout<<pts1.size()<<"|"<<pts2.size()<<std::endl;
//     //cv::Mat fundamental_matrix;
//     //fundamental_matrix = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3, 0.99);
//     //std::cout<<"fundamental_matrix is"<<std::endl<<fundamental_matrix<<std::endl;
//     //cv::Mat cameraMatrix=(cv::Mat_<double>(3,3)<< 405.6385,0,189.9054,0,405.5883,139.9150,0,0,1);
//     cv::Mat essential_matrix;
//     essential_matrix = cv::findEssentialMat(pts1,pts2,cameraMatrix);
//     std::cout<<"have essential matrix"<<std::endl;
//     cv::recoverPose(essential_matrix, pts1, pts2, cameraMatrix,R,t);
//     cv::Vec3f euler= rotationMatrixToEulerAngles(R);
//     std::cout<<euler<<std::endl;
//     return 1;
// }

// int Track(cv::Mat& img1, cv::Mat& img2, cv::Mat& R, cv::Mat& t){
//     int num_good_pts=0;
//     std::vector<cv::Point2f> pts1, pts2;
//     // cv::Mat desc1,desc2;
//     // cv::Mat dest1;
//     // cv::Ptr<cv::ORB> detector = cv::ORB::create();
//     // detector->detect(img1,pts1,cv::Mat());
//     // cv::drawKeypoints(img1, pts1, dest1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
//     // cv::imshow("output", dest1);
//     std::vector<uchar> status;
//     std::vector<float> err;
//     cv::Mat old_frame,cur_frame;
//     cv::cvtColor(img1, old_frame, cv::COLOR_BGR2GRAY);
//     cv::goodFeaturesToTrack(old_frame, pts1, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
//     cv::cvtColor(img2, cur_frame, cv::COLOR_BGR2GRAY);
//     //goodFeaturesToTrack(cur_frame, pts2, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);

//     cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
//     cv::calcOpticalFlowPyrLK(old_frame, cur_frame, pts1, pts2, status, err, cv::Size(15,15), 2, criteria);
//     cv::Mat mask = cv::Mat::zeros(img1.size(), img1.type());
//     std::vector<cv::Point2f> good_new;
//     for(uint i = 0; i < pts1.size(); i++)
//     {
//         // std::cout<<"good "<<i<<std::endl;
//         // Select good points
//         if(status[i] == 1) {
//             good_new.push_back(pts2[i]);
//             // draw the tracks
//             line(mask,pts2[i], pts1[i], cv::Scalar(255,239,213), 5);
//             circle(img2, pts2[i], 1, cv::Scalar(0.01+0.03*i,0.01+0.03*i,0.01+0.03*i), -1);
//             num_good_pts++;
//         }
//     }
//     cv::Mat img;
//     cv::add(img2, mask, img);
//     cv::imshow("input", img);
//     //pts1和pts2为img1和img2的对应匹配点
//     CalcMotion(pts1,pts2,R,t);
//     std::cout<<R<<std::endl;
//     std::cout<<t<<std::endl;
//     return num_good_pts;
// }

// int main(int argc, char ** argv){
//     state << 0,0,0;//x,y,z,psi,theta,phi
//     std::string pngpath = "../images";
//     std::vector<std::string> image_files;
//     cv::glob(pngpath, image_files);

//     // cv::Mat img1 = cv::imread("1.png");
//     // cv::Mat img2 = cv::imread("2.png");
//     std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> poses;
//     cv::Mat lastimg,curimg;
//     for(unsigned int frame = 0; frame<image_files.size();frame++){
//         if(frame == 0){
//             curimg = cv::imread(image_files[frame]);
//             continue;
//         }
//         else{
//             lastimg = curimg;
//             curimg = cv::imread(image_files[frame]);
//         }
//         cv::Mat R,t;
//         std::cout<<Track(lastimg, curimg, R, t);//计算两帧间相对运动
//         Eigen::MatrixXd Re(3,3),te(3,1);
//         cv::cv2eigen(R,Re);
//         cv::cv2eigen(t,te);

//         state = Re*state+te;
//         Eigen::Vector4d c;
//         c<<state[0],state[1],state[2],0.4;
//         std::cout<<c<<std::endl;
//         poses.push_back(c);
//     }
//     // cv::Mat img1 = cv::imread(image_files[0]);
//     showPointCloud(poses);
//     // std::vector<cv::Point2f> kps_last, kps_cur;

//     // cv::namedWindow("input", cv::WINDOW_AUTOSIZE);
//     // cv::imshow("input", img1);
//     // cv::imshow("input", img2);

//     // cv::Mat R,t;
//     // std::cout<<Track(img1, img2,R, t);//计算两帧间相对运动

//     //cv::waitKey(0);
//     return 0;
// }




// using namespace std;
// using namespace Eigen;

// std::mutex pm;


// // 函数声明，在pangolin中画图，已写好，无需调整
// void changeSome(std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud){
//     while(1){
//         usleep(1000);
//         pm.lock();
//         for (auto &p: pointcloud) {
// 	    p[0]+=((float)rand()/RAND_MAX)*3*pow(-1,rand()&1);
// 	    p[1]+=((float)rand()/RAND_MAX)*3*pow(-1,rand()&1);
//         }
//         pm.unlock();
//     }
// }

// //使用pangolin绘制点云图函数
// void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud) {

//     //如果点云是空的，不绘制
//     if (pointcloud.empty()) {
//         std::cerr << "Point cloud is empty!" << std::endl;
//         return;
//     }

//     //创建并初始化显示窗口，定义窗口名称，宽度，高度
//     pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
//     glEnable(GL_DEPTH_TEST);
//     glEnable(GL_BLEND);
//     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//     pangolin::OpenGlRenderState s_cam(
//             pangolin::ProjectionMatrix(1024, 768, 20, 20, 512, 389, 0.1, 1000),
//             pangolin::ModelViewLookAt(0, 0, 500, 0, 0, 0, pangolin::AxisY)
//     );

//     pangolin::View &d_cam = pangolin::CreateDisplay()
//             .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
//             .SetHandler(new pangolin::Handler3D(s_cam));

//     while (pangolin::ShouldQuit() == false) {
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//         d_cam.Activate(s_cam);
//         glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

//         glPointSize(2);
//         glBegin(GL_LINE_STRIP);
//         //如果想绘制其他图像，将这里的pointcloud改成对应的容器就可以
//         //或者将pointcloud对应到其他图像
//         for (auto &p: pointcloud) {
//             glColor3f(p[3], p[3], p[3]);
//             glVertex3d(p[0], p[1], p[2]);
//         }
//         glEnd();
//         //usleep(5000);   // sleep 5 ms
//         pangolin::FinishFrame();
//     }
//     return;

// }

// int main(int argc, char **argv) {

//     // 内参
//     double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
//     // 基线
//     double b = 0.573;


//     std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud;

//     // 遍历元素
//     for (int v = 0; v < 200; v++){
//         for (int u = 0; u < 200; u++) {
//             //创建一个四维向量用于存储一个三维点的信息，前三维为xyz
//             // 第四维为颜色，这里采用原始图像颜色作为参考，显示效果更好，并对颜色做了归一化处理
//             Eigen::Vector4d point(0, 0, 0, rand()%254 / 255.0);
//             // 根据双目模型计算 point 的三维空间位置
//             //double x = (u - cx) / fx;
//             //double y = (v - cy) / fy;
//             //double depth = fx * b / (disparity.at<float>(v, u));
            
//             //point[0] = x * depth;
//             //point[1] = y * depth;
//             //point[2] = depth;
//             double r=((float)rand()/RAND_MAX)*50;
//             double t=((float)rand()/RAND_MAX)*2*PI;
//             point[0] = r*cos(t);
//             point[1] = r*sin(t);
//             point[2] = 1;
//             //将计算得到的三维点从尾部添加到点云容器point中
//             pointcloud.push_back(point);
//         }
//     }
//     //显示视差图，对视差图中所有像素灰度值除以96，从而将像素值归一化到0-1，符合CV_32F格式条件，并显示
//     //cv::imshow("disparity", disparity / 96.0);
//     //cv::waitKey(0);
//     // 画出点云.
//     std::thread* cThread = new std::thread(changeSome,std::ref(pointcloud));
//     cThread->detach();
//     std::thread* sThread = new std::thread(showPointCloud,std::ref(pointcloud));
//     sThread->join();
//     return 0;
// }

// int main(int argc, char** argv)
// {
// 	//使用系统时间做随机数种子
// 	srand(time(NULL));
// 	//创建一个PointXYZ类型点云指针
// 	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
 
// 	//初始化点云数据
// 	cloud->width = 1000;//宽为1000
// 	cloud->height = 1;//高为1，说明为无序点云
// 	cloud->points.resize(cloud->width * cloud->height);
// 	//使用随机数填充数据
 
// 	for (size_t i = 0; i < cloud->size(); ++i)
// 	{
// 		//PointCloud类中对[]操作符进行了重载，返回的是对points的引用
// 		// (*cloud)[i].x 等同于 cloud->points[i].x
// 		(*cloud)[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
// 		cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);//推进写法
// 		cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);//推进写法
// 	}
 
// 	//创建kd树对象
// 	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
 
// 	//设置点云输入,将在cloud中搜索
// 	kdtree.setInputCloud(cloud);
 
// 	//设置被搜索点,用随机数填充
// 	pcl::PointXYZ searchPoint;
// 	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
// 	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
// 	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
 
// 	//开始k最近邻搜索
// 	int K = 10;
// 	//使用两个vector存储搜索结果
// 	vector<int> pointIdxNKNSearch(K);//保存下标
// 	vector<float> pointNKNSquaredDistance(K);//保存距离的平方
 
// 	cout << "K nearest neighbor search at (" << searchPoint.x
// 		<< " " << searchPoint.y
// 		<< " " << searchPoint.z
// 		<< ") with K = " << K << endl;
// 	/**
// 	 * 假设我们的KdTree返回超过0个最近的邻居，
// 	 * 然后它打印出所有10个离随机searchPoint最近的邻居的位置，
// 	 * 这些都存储在我们之前创建的vector中。
// 	 */
// 	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
// 	{
// 		for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
// 		{
// 			cout << "    " << cloud->points[pointIdxNKNSearch[i]].x
// 				<< " " << cloud->points[pointIdxNKNSearch[i]].x
// 				<< " " << cloud->points[pointIdxNKNSearch[i]].z
// 				<< "( squared distance: " << pointNKNSquaredDistance[i] << " )" << endl;
// 		}
// 	}
 
// 	//基于半径的邻域搜索
// 	//搜索结果保存在两个数组中，一个是下标，一个是距离
// 	vector<int> pointIdxRadiusSearch;
// 	vector<float> pointRadiusSquaredDistance;
 
// 	//设置搜索半径，随机值
// 	float radius = 256.0f* rand() / (RAND_MAX + 1.0f);
 
// 	cout << "Neighbors within radius search at (" << searchPoint.x
// 		<< " " << searchPoint.y
// 		<< " " << searchPoint.z
// 		<< ") with radius=" << radius << endl;
 
// 	/**
// 	 * 如果我们的KdTree在指定的半径内返回超过0个邻居，它将打印出这些存储在向量中的点的坐标。
// 	 */
// 	if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
// 	{
// 		for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
// 		{
// 			cout << "    " << cloud->points[pointIdxRadiusSearch[i]].x
// 				<< " " << cloud->points[pointIdxRadiusSearch[i]].x
// 				<< " " << cloud->points[pointIdxRadiusSearch[i]].z
// 				<< "( squared distance: " << pointRadiusSquaredDistance[i] << " )" << endl;
// 		}
// 	}
// 	return 0;
// }

// #include <Eigen/Core>
// #include <vector>
// #include <iostream>

// int main(){
//     std::vector<Eigen::VectorXd> a;
//     Eigen::Vector3d x1(2,3,4);
//     Eigen::VectorXd x2(2000);
//     for(int i=0;i<2000;i++)x2(i)=i;
//     Eigen::Vector2d x3(3,2);
//     a.push_back(x1);
//     a.push_back(x2);
//     a.push_back(x3);
//     for(int i=0;i<a.size();i++){
//         std::cout<<a[i]<<std::endl;
//     }
//     return 0;
// }

// #include <pangolin/pangolin.h>
// #include <stdlib.h>
// #include <thread>
// #include <vector>
// #include <mutex>
// #include <unistd.h>


// static const std::string window_name = "HelloPangolinThreads";

// struct point{
//     point(double xx, double yy):x(xx),y(yy){}
//     double x;
//     double y;
// };

// std::vector<point>a;
// std::mutex mx;

// int k1=0,k2=0,k3=1;
// void setup() {
//     // create a window and bind its context to the main thread
//     pangolin::CreateWindowAndBind(window_name, 640, 480);

//     // enable depth
//     glEnable(GL_DEPTH_TEST);

//     // unset the current context from the main thread
//     pangolin::GetBoundWindow()->RemoveCurrent();
// }

// void run() {
//     sleep(2);
//     // fetch the context and bind it to this thread
//     pangolin::BindToContext(window_name);

//     // we manually need to restore the properties of the context
//     glEnable(GL_DEPTH_TEST);

//     // Define Projection and initial ModelView matrix
//     pangolin::OpenGlRenderState s_cam(
//         pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
//         pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
//     );

//     // Create Interactive View in window
//     pangolin::Handler3D handler(s_cam);
//     pangolin::View& d_cam = pangolin::CreateDisplay()
//             .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
//             .SetHandler(&handler);

//     while( !pangolin::ShouldQuit() )
//     {
//         // Clear screen and activate view to render into
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//         d_cam.Activate(s_cam);

//         // Render OpenGL Cube
//         if(!rand()){
//           pangolin::glDrawColouredCube();
//         }else{
//            glBegin(GL_LINES);
//            glColor3f(1,0,0);//修改颜色
//            mx.lock();
//            for( int i=0;i<a.size();i++){
//               glVertex3f(a[i].x,a[i].y,0);
//               //glVertex3f(0,1,k3);
//             //glVertex3f(0,0,0);
//             //glVertex3f(k3,0,k3);
//             std::cout<<"画了"<<i<<":"<<a[i].x<<a[i].y<<std::endl;
//            }
//            mx.unlock();
// 	        glEnd();
//         }

//         // Swap frames and Process Events
//         pangolin::FinishFrame();
//     }

//     // unset the current context from the main thread
//     pangolin::GetBoundWindow()->RemoveCurrent();
// }

// void run2() {
//     while(true){
//         k3=rand()%3;
//     }
//     return ;
// }

// void well(){
//     //A a;
//     //a.good();
//     std::thread render_loop(run);
//     //render_loop = std::thread(run);
//     render_loop.detach();
//     int cnt = 1;
//     while(1){
//         mx.lock();
//         switch(cnt%4){
//             case 0:
//                 a.push_back(point(cnt,0));
//                 break;
//             case 1:
//                 a.push_back(point(0,cnt));
//                 break;
//             case 2:
//                 a.push_back(point(-cnt,0));
//                 break;
//             case 3:
//                 a.push_back(point(0,-cnt));
//                 break;
//         }
//         cnt++;
//         mx.unlock();
//         usleep(500000);
//     }
//     //std::cout<<"well done"<<std::endl;
// }


// int main( int /*argc*/, char** /*argv*/ )
// {
//     // create window and context in the main thread
//     setup();

//     // use the context in a separate rendering thread
//     std::thread my_loop;
//     well();
//     k3=3;
    
//     k3=1;
//     my_loop = std::thread(run2);
//     my_loop.join();
//     std::cout<<"love"<<std::endl;
//     return 0;
// }
