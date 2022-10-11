
#include <yarp/cv/Cv.h>
#include <yarp/os/all.h>
#include <yarp/sig/Image.h>
#include <event-driven/all.h>
#include <hpe-core/utility.h>
#include <hpe-core/motion_estimation.h>
#include <hpe-core/openpose_detector.h>
#include <hpe-core/fusion.h>
#include <opencv2/opencv.hpp>
#include "yarp/rosmsg/output.h"


#include <vector>
#include <string>

using namespace yarp::os;
using namespace yarp::sig;
using std::vector;
using namespace ev;

class openposethread {
private:

    std::thread th;
    hpecore::OpenPoseDetector detop;
    hpecore::stampedPose pose{0.0, -1.0, 0.0};
    cv::Mat image;

    bool stop{false};
    bool data_ready{true};
    std::mutex m;

    void run()
    {
        while(true) {
            m.lock();
            if(stop) return;
            pose.pose = detop.detect(image);
            data_ready = true;
        }
    }

public:

    bool init(std::string model_path, std::string model_name)
    {
        //initialise open pose
        if(!detop.init(model_path, model_name, "256"))
            return false;
        
        //make sure the thread won't start until an image is provided
        m.lock();

        //make sure that providing an image will start things for the first go
        data_ready = true;

        //start the thread
        th = std::thread( [this]{this->run();} );
        
        return true;
    }

    void close()
    {
        stop = true;
        m.try_lock();
        m.unlock();
    }

    bool update(cv::Mat next_image, double image_timestamp, hpecore::stampedPose &previous_result)
    {
        //if no data is ready (still processing) do nothing
        if(!data_ready)
            return false;
        
        //else set the result to the provided stampedPose
        previous_result = pose;
        
        //set the timestamp
        pose.timestamp = image_timestamp;

        //and the image for the next detection
        static cv::Mat img_u8, img_float;
        next_image.copyTo(img_float);
        double min_val, max_val;
        cv::minMaxLoc(img_float, &min_val, &max_val);
        max_val = std::max(fabs(max_val), fabs(min_val));
        img_float /= (2 * max_val);
        img_float += 0.5; 
        img_float.convertTo(img_u8, CV_8U, 255, 0);
        cv::cvtColor(img_u8, image, cv::COLOR_GRAY2BGR);

        //and unlock the procesing thread
        m.try_lock();
        m.unlock();
        data_ready = false;
        return true;
    }

    // hpecore::OpenPoseDetector detop;

    // hpecore::skeleton13 detect(cv::Mat pimmer3)
    // {
    //     hpecore::skeleton13 pose = this->detop.detect(pimmer3);
    //     return pose;
    // }

};

class isaacHPE : public RFModule, public Thread {

private:

    vReadPort<vector<AE> > input_events;
    BufferedPort<Bottle> input_gt;
    BufferedPort<Bottle> input_mn;
    BufferedPort<ImageOf<PixelMono>> input_grey;

    hpecore::skeleton13 skeleton_gt{0};
    hpecore::skeleton13 skeleton_detection{0};

    openposethread opt;
    hpecore::jointName my_joint;
    // hpecore::queuedVelocity velocity_estimator;
    hpecore::surfacedVelocity velocity_estimator;

    // hpecore::kfEstimator state;
    hpecore::stateEstimator state;
    hpecore::PIM pim;
    std::mutex m;

    cv::Size image_size;
    cv::Mat vis_image;
    cv::Mat grey_frame;

    hpecore::writer skelwriter;

    bool use_gt{false};
    int roiSize;
    int detF;
    bool csv = false, circle = false, err = false, noVelEst = false;
    double scaler;
    cv::VideoWriter output_video;
    bool rec = false;
    int noise;
    double procU, measU;
    bool movenet = false;

    yarp::os::Node* ros_node{nullptr};
    yarp::os::Publisher<yarp::rosmsg::output> ros_publisher;
    yarp::rosmsg::output ros_output;

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {
        if (!yarp::os::Network::checkNetwork(2.0)) {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }
        //set the module name used to name ports
        setName((rf.check("name", Value("/vojext-hpe")).asString()).c_str());

        //open io ports
        if(!input_events.open(getName("/AE:i"))) {
            yError() << "Could not open events input port";
            return false;
        }

        if(!input_gt.open(getName("/gt:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        if(!input_mn.open(getName("/movenet:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        // Network::connect("/file/ch0dvs:o", getName("/AE:i"), "fast_tcp");
        // // Network::connect("/atis3/AE:o", getName("/AE:i"), "fast_tcp");
        // Network::connect("/file/ch2GT50Hzskeleton:o", getName("/gt:i"), "fast_tcp");
        // Network::connect("/movenet/sklt:o", getName("/movenet:i"), "fast_tcp");
         Network::connect("/zynqGrabber/AE:o", getName("/AE:i"), "fast_tcp");

        use_gt = rf.check("use_gt") && rf.check("use_gt", Value(true)).asBool();
        movenet = rf.check("movenet") && rf.check("movenet", Value(true)).asBool();
      
        image_size = cv::Size(rf.check("w" , Value(640)).asInt32(),
                              rf.check("h", Value(480)).asInt32());
        vis_image = cv::Mat(image_size, CV_8UC3, cv::Vec3b(0, 0, 0));
        grey_frame = cv::Mat(image_size, CV_8UC1, cv::Vec3b(0, 0, 0));
        cv::namedWindow("isaac-hpe", cv::WINDOW_NORMAL);

        // cv::namedWindow("Representation Visualisation", cv::WINDOW_NORMAL);

        my_joint = hpecore::str2enum("elbowR");
        // velocity_estimator.setParameters(40, 3, 5, 1000);


        roiSize = rf.check("roi", Value(32)).asInt32();
        velocity_estimator.setParameters(roiSize, 2, 5, 800, image_size);
        detF = rf.check("detF", Value(10)).asInt32();
        if(rf.check("c"))
            circle = true;
        if(rf.check("err"))
            err = true;
        if(rf.check("nve"))
            noVelEst = true;
        scaler = rf.check("sc", Value(2.0)).asFloat64();
        noise = rf.check("n", Value(0)).asInt32();

        if(rf.check("filepath")) {
            std::string filepath = rf.find("filepath").asString();
            if(skelwriter.open(filepath))
                yInfo() << "saving data to:" << filepath;
        }


        procU = rf.check("pu", Value(1.0)).asFloat64();
        measU = rf.check("mu", Value(1e-3)).asFloat64();
        
        state.initialise({procU, measU});
        pim.init(image_size.width, image_size.height);

        std::string models_path = rf.check("models_path", Value("/openpose/models")).asString();
        std::string pose_model = rf.check("pose_model", Value("COCO")).asString();
        if(!opt.init(models_path, pose_model))
            return false;

        if(rf.check("v")) {
            std::string videopath = rf.find("v").asString();
            if(!output_video.open(videopath + "/outputvideo.mp4",
                          cv::VideoWriter::fourcc('H','2','6','4'), 30,
                          cv::Size(image_size.width, image_size.height)))
            {
                yError() << "Could not open video writer!!";
                return false;
            }
        }
        
        // set-up ROS interface
        ros_node = new yarp::os::Node("/VOJEXT");
        if(!ros_publisher.topic(getName("/output2ros"))) {
            yError() << "Could not open ROS output publisher";
            return false;
        }


        return Thread::start();
    }

    double getPeriod() override
    {
        //run the module as fast as possible. Only as fast as new images are
        //available and then limited by how fast OpenPose takes to run
        return 0.05; 
    }

    bool interruptModule() override
    {
        //if the module is asked to stop ask the asynchronous thread to stop
        return Thread::stop();
    }

    void onStop() override
    {
        opt.close();
        input_events.close();
        input_grey.close();
        input_gt.close();
        input_mn.close();
        skelwriter.close();
        output_video.release();
    }

    bool close() override
    {
        //when the asynchronous thread is asked to stop, close ports and do other clean up
        return true;
    }

    //synchronous thread
    virtual bool updateModule() 
    {
        // yInfo() << input_events.queryDelayT();

        //perform open-pose
        cv::Mat floater;
        pim.getSurface().copyTo(floater);

        //cv::GaussianBlur(floater, floater, cv::Size(5, 5), 0);
        double min_val, max_val;
        cv::minMaxLoc(floater, &min_val, &max_val);
        max_val = std::max(fabs(max_val), fabs(min_val));
        floater /= (2*max_val);
        floater += 0.5;

        cv::Mat pimmer, pimmer3;
        floater.convertTo(pimmer, CV_8U, 255, 0);
        //pimmer = grey_frame;
        cv::cvtColor(pimmer, pimmer3, cv::COLOR_GRAY2BGR);
    
        // double tic = yarp::os::Time::now();
        // hpecore::skeleton13 pose = opt.detop.detect(pimmer3);
        // hpecore::skeleton13 pose = opt.detect(pimmer3);
        // yInfo() << "open-pose:" << (int)(1000 * (Time::now() - tic)) << "ms";
        // hpecore::drawSkeleton(pimmer3, pose);


        cv::Size compiled_size(std::max(grey_frame.size().width, vis_image.size().width),
                            std::max(grey_frame.size().height, vis_image.size().height));
        cv::Mat compiled(compiled_size, CV_8UC3, cv::Vec3b(0, 0, 0));
        cv::Mat rgb; cv::cvtColor(grey_frame, rgb, cv::COLOR_GRAY2BGR);
        rgb.copyTo(compiled(cv::Rect(0, 0, rgb.cols, rgb.rows)));
        vis_image.copyTo(compiled, vis_image);
        // pimmer3.copyTo(compiled, pimmer3);
        
        //hpecore::print_skeleton<hpecore::skeleton13>(skeleton_gt);
        //hpecore::drawSkeleton(compiled, state.query(), {128, 128, 0});
        // cv::rectangle(compiled, cv::Rect(state.query(my_joint).u-20, state.query(my_joint).v-20, 40, 40), CV_RGB(255, 0, 0));
        vis_image.setTo(cv::Vec3b(0, 0, 0));
        // pimmer3.setTo(cv::Vec3b(0, 0, 0));
        



        
        //cv::normalize(floater, floater, 0, 1, cv::NORM_MINMAX);
        //cv::namedWindow("Representation Visualisation", cv::WINDOW_NORMAL);
        hpecore::drawSkeleton(compiled, state.query(), {0, 0, 255}, 3);
        hpecore::drawSkeleton(compiled, skeleton_detection
, {255, 0, 0});
        cv::imshow("isaac-hpe", compiled);
        // cv::imshow("Representation Visualisation", pimmer3);
        cv::waitKey(1);

        // output_video.write(compiled);
        if (rec)
            output_video << compiled;

        return true;
    }

    //asynchronous thread
    void run() override
    {
        Stamp ystamp;
        const vector<AE> *q;
        double  t0 = Time::now(),   // initial time
                tnow=t0,            // current time
                tD=t0;              // time between detections
        bool initTimer = false;
        std::chrono::high_resolution_clock::duration tMax;
        hpecore::stampedPose detected_pose;
        double dt;
        bool detection_available;
        double t_write;
        double randNoise;
        hpecore::skeleton13_vel jv;

        while (!Thread::isStopping())
        {
            auto start = std::chrono::high_resolution_clock::now();
            tnow = Time::now() - t0;
            static double tic = yarp::os::Time::now();

            // ---------- DETECTIONS ----------
            if (use_gt) // use ground-truth, no detector
            {
                Bottle *gt_container = input_gt.read(false);
                if (gt_container && 1 / (tnow - tD) <= detF)
                {
                    skeleton_detection = hpecore::extractSkeletonFromYARP<Bottle>(*gt_container);
                    std::cout << "\033c";
                    yInfo() << 1 / (tnow - tD);
                    if (noise) // add noise to ground-truth
                    {
                        for (int i = 0; i < 13; i++)
                        {
                            randNoise = rand() * (2.0 * noise / RAND_MAX) - noise / 2;
                            skeleton_detection[i].u += randNoise;
                            randNoise = rand() * (2.0 * noise / RAND_MAX) - noise / 2;
                            skeleton_detection[i].v += randNoise;
                        }
                    }
                    state.updateFromPosition(skeleton_detection, 0.0);
                    dt = tnow - tD;
                    tD = Time::now() - t0;
                    detection_available = (gt_container != nullptr);
                }
                else
                    detection_available = false;
            }
            else
            {
                if (movenet) // use Movenet
                {
                    Bottle *mn_container = input_mn.read(false);
                    if (mn_container)
                    {
                        skeleton_detection = hpecore::extractSkeletonFromYARP<Bottle>(*mn_container);
                        std::cout << "\033c";
                        yInfo() << 1 / (tnow - tD);
                        dt = tnow - tD;
                        tD = Time::now() - t0;
                        state.updateFromPosition(skeleton_detection, dt);
                        detection_available = (mn_container != nullptr);
                    }
                    else
                        detection_available = false;
                }
                else // use OpenPose
                {
                    detection_available = opt.update(pim.getSurface(), Time::now(), detected_pose);
                    if (detection_available && hpecore::poseNonZero(detected_pose.pose))
                    {
                        skeleton_detection = detected_pose.pose;
                        std::cout << "\033c";
                        yInfo() << 1 / (tnow - tD);
                        dt = tnow - tD;
                        tD = Time::now() - t0;
                        
                        // format skeleton to ros output
                        std:vector<double> sklt_out, vel_out;
                        for (int j = 0; j < 13; j++)
                        {
                            sklt_out.push_back(skeleton_detection[j].u);
                            sklt_out.push_back(skeleton_detection[j].v);
                            vel_out.push_back(jv[j].u);
                            vel_out.push_back(jv[j].v);
                        }
                        // for(auto t : sklt_out)
                        //     std::cout << t << ", ";
                        // std::cout << std::endl;
                        // for(auto t : vel_out)
                        //     std::cout << t << ", ";
                        // std::cout << std::endl;
                        ros_output.timestamp = tnow;
                        ros_output.pose = sklt_out;
                        ros_output.velocity = vel_out;
                        // publish data
                        ros_publisher.prepare() = ros_output;
                        ros_publisher.write();
                    }
                }
            }

            // ---------- EVENT PROCESSING ----------
            int nqs = input_events.queryunprocessed();
            for (auto i = 0; i < nqs; i++)
            {
                if (!initTimer)
                {
                    initTimer = true;
                    rec = true;
                    t0 = Time::now();
                }
                q = input_events.read(ystamp);
                if (!q)
                    return;
                for (auto &v : *q)
                {
                    if (v.polarity)
                        vis_image.at<cv::Vec3b>(v.y, v.x) = cv::Vec3b(64, 150, 90);
                    else
                        vis_image.at<cv::Vec3b>(v.y, v.x) = cv::Vec3b(32, 82, 50);
                    pim.update(v.x, v.y, 0.0, v.polarity);
                }

                if (!noVelEst)
                {
                    // ---------- VELOCITY ESTIMATION ----------
                    // hpecore::skeleton13_vel jv;

                    if (err)
                    {
                        // calculate error to previous velocity
                        if (circle)
                            velocity_estimator.errorToCircle<vector<AE>>(*q, state.query(), state.queryVelocity(), state.queryError());
                        else
                            velocity_estimator.errorToVel<vector<AE>>(*q, state.query(), state.queryVelocity(), state.queryError());
                    }
                    else
                    {
                        // estimate velocities without error calculation
                        jv = velocity_estimator.update<vector<AE>>(*q, state.query());
                    }

                    // update velocity based on error
                    if (err)
                        jv = velocity_estimator.updateOnError(state.queryVelocity(), state.queryError());
                    state.setVelocity(jv);
                    for (int j = 0; j < 13; j++) // (F) overload * to skeleton13
                    {
                        jv[j] = jv[j] * vtsHelper::vtsscaler * scaler;
                    }

                    // update state with velocity
                    state.updateFromVelocity(jv, ev::vtsHelper::deltaS(q->back().stamp, q->front().stamp));
                }

                // WRITE CSV OUTPUT: velocity based update
                t_write = q->back().stamp * vtsHelper::vtsscaler * 1e-13;
                if (state.poseIsInitialised() && ystamp.getTime() && hpecore::poseNonZero(state.query()))
                    skelwriter.write({tnow, input_events.queryDelayT(), state.query()});
            }

            if (use_gt) // use ground-truth, no detector
            {
                if (!state.poseIsInitialised() && detection_available)
                {
                    state.set(skeleton_detection);
                }
                else if (state.poseIsInitialised() && detection_available)
                {
                    state.updateFromPosition(skeleton_detection, dt);
                }
            }
            else
            {
                if (movenet) // use Movenet
                {
                    if (!state.poseIsInitialised() && detection_available)
                    {
                        state.set(skeleton_detection);
                    }
                    else if (state.poseIsInitialised() && detection_available)
                    {
                        state.updateFromPosition(skeleton_detection, dt);
                    }
                }
                else
                {
                    if (!state.poseIsInitialised() && detection_available && detected_pose.timestamp > 0.0)
                    {
                        state.set(skeleton_detection);
                    }

                    else if (state.poseIsInitialised() && detection_available && hpecore::poseNonZero(skeleton_detection))
                    {
                        state.updateFromPosition(skeleton_detection, dt);
                    }
                }
            }
            pim.temporalDecay(Time::now(), 1.0);

            // WRITE CSV OUTPUT: detection based update
            if (state.poseIsInitialised() && detection_available && initTimer && nqs)
                skelwriter.write({tnow, input_events.queryDelayT(), state.query()});
        }
    }
};

int main(int argc, char * argv[])
{
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose( false );
    rf.configure( argc, argv );

    /* create the module */
    isaacHPE instance;
    return instance.runModule(rf);
}
