#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

void print_info() {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    for (int p_id = 0; p_id < platforms.size(); p_id++) {
        std::cout << "Platform " << p_id << ": " << platforms[p_id].get_info<sycl::info::platform::name>() << std::endl;
        std::vector<sycl::device> devices = platforms[p_id].get_devices();
        for (int d_id = 0; d_id < devices.size(); d_id++) {
            std::cout << "-- Device " << d_id << ": " << devices[d_id].get_info<sycl::info::device::name>() << std::endl;
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
           {
                sycl::buffer<int> buf_p_id(&p_id, 1);
                sycl::buffer<int> buf_d_id(&d_id, 1);

                queue.submit([&](sycl::handler& cgh) {
                    sycl::stream out(1024, 80, cgh);

                    auto acc_p_id = buf_p_id.get_access<sycl::access::mode::read>(cgh);
                    auto acc_d_id = buf_d_id.get_access<sycl::access::mode::read>(cgh);

                    // 4 work items, 4 groups, 1 work item in each group
                    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(4), sycl::range<1>(1)), [=](sycl::nd_item<1> item) {
                        out << "[" << item.get_global_id(0) << "] Hello from platform " << acc_p_id[0] << " and device " << acc_d_id[0] << sycl::endl;
                    });
                    });
                    queue.wait();
                    std::cout << std::endl;
           }
       }
   }
}

int main(int argc, char* argv[]) {
    std::cout << std::endl;
    print_info();
    hello_world();
    return 0;
}