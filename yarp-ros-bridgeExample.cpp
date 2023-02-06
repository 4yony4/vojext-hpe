#include <yarp/os/all.h>
#include <event-driven/all.h>
#include "yarp/rosmsg/Vjxoutput.h"
#include <vector>

using namespace yarp::os;
using namespace yarp::sig;
using std::vector;
using namespace ev;

class yarpRosBridge : public RFModule, public Thread
{

private:
    int detF;        // frequency of Vjxoutput
    double t0, tnow; // timestamp variables

    // yarp-ros bridge
    yarp::os::Node *ros_node{nullptr};
    yarp::os::Publisher<yarp::rosmsg::Vjxoutput> ros_publisher;
    yarp::rosmsg::Vjxoutput ros_output;

public:
    bool configure(yarp::os::ResourceFinder &rf) override
    {
        if (!yarp::os::Network::checkNetwork(2.0))
        {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }
        // set the module name used to name ports
        // setName((rf.check("name", Value("/vojext-hpe")).asString()).c_str());

        detF = rf.check("detF", Value(10)).asInt32();

        // set-up ROS interface
        ros_node = new yarp::os::Node("/VOJEXT");
        if (!ros_publisher.topic(getName("/output2ros")))
        {
            yError() << "Could not open ROS output publisher";
            return false;
        }

        return Thread::start();
    }

    double getPeriod() override
    {
        // run the module as fast as possible. Only as fast as new images are
        // available and then limited by how fast OpenPose takes to run
        return 1.0 / detF;
    }

    bool interruptModule() override
    {
        // if the module is asked to stop ask the asynchronous thread to stop
        return Thread::stop();
    }

    void onStop() override
    {
    }

    bool close() override
    {
        // when the asynchronous thread is asked to stop, close ports and do other clean up
        return true;
    }

    // synchronous thread
    virtual bool updateModule()
    {
    // format skeleton to ros output
    std:
        vector<double> sklt_out, vel_out;
        for (int j = 0; j < 13; j++)
        {
            sklt_out.push_back(rand() * 150.0 / RAND_MAX);
            sklt_out.push_back(rand() * 150.0 / RAND_MAX);
            vel_out.push_back(rand() * (300.0 / RAND_MAX) - 150);
            vel_out.push_back(rand() * (300.0 / RAND_MAX) - 150);
        }
        // // print to check what data is created - used to debug
        // std::cout << tnow << std::endl;
        // for(auto t : sklt_out)
        //     std::cout << t << ", ";
        // std::cout << std::endl;
        // for(auto t : vel_out)
        //     std::cout << t << ", ";
        // std::cout << std::endl;
        // std::cout << std::endl;

        // put data in ros output structure
        ros_output.timestamp = tnow;
        ros_output.pose = sklt_out;
        ros_output.velocity = vel_out;
        // publish data
        ros_publisher.prepare() = ros_output;
        ros_publisher.write();

        return true;
    }

    // asynchronous thread
    void run() override
    {
        t0 = Time::now(); // initial time

        while (!Thread::isStopping())
        {
            tnow = Time::now() - t0;
        }
    }
};

int main(int argc, char *argv[])
{
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    /* create the module */
    yarpRosBridge instance;
    return instance.runModule(rf);
}
