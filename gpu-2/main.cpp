#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>

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

std::pair<int, std::string> parse_args(int argc, char* argv[]) {
    int N;
    std::string device;
    try {
        if (argc < 2) throw -1;
        N = atoi(argv[1]);
        if (N == 0) throw -1;
        device = argv[2];
        if (device != "cpu" && device != "gpu") {
            throw -1;
        }
    } catch (...) {
        std::cout << "Args error" << std::endl;
        exit(-1);
    }
    return std::pair<int, std::string>{N, device};
}

void integral(int N, std::string device_type) {
    sycl::queue queue;
    if (device_type == "cpu") {
        queue = sycl::queue(sycl::cpu_selector{});
    } else if (device_type == "gpu") {
        queue = sycl::queue(sycl::gpu_selector{});
    } else {std::cout << "Selector error" << std::endl; exit(-1);}

    std::cout << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    int N = args.first;
    std::string device_type = args.second;

    integral(N, device_type);

    return 0;
}