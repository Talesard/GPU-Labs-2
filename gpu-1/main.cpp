#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

void print_info() {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    for (int p_id = 0; p_id < platforms.size(); p_id++) {
        std::cout << "Platform #" << p_id << ": " << platforms[p_id].get_info<sycl::info::platform::name>() << std::endl;
        std::vector<sycl::device> devices = platforms[p_id].get_devices();
        for (int d_id = 0; d_id < devices.size(); d_id++) {
            std::cout << "-- Device #" << d_id << ": " << devices[d_id].get_info<sycl::info::device::name>() << std::endl;
        }
    }
    std::cout << std::endl;
}

void hello_world() {
   std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
   for (int p_id = 0; p_id < platforms.size(); p_id++) {
       std::vector<sycl::device> devices = platforms[p_id].get_devices();
       for (int d_id = 0; d_id < devices.size(); d_id++) {
           sycl::queue queue((devices[d_id]));
           std::cout << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
           queue.submit([&](sycl::handler& cgh) {
               sycl::stream out(1024, 80, cgh);
               cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(4), sycl::range<1>(1)), [=](sycl::nd_item<1> item) {
                   out << "[" << item.get_global_id(0) << "] Hello from platform " << p_id << " and device " << d_id << sycl::endl;
               });
            });
            queue.wait();
            std::cout << std::endl;
       }
   }
}

int main(int argc, char* argv[]) {
    std::cout << std::endl;
    print_info();
    hello_world();
    return 0;
}