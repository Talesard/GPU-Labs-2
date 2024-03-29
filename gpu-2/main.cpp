#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <numeric>

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
        if (argc < 3) throw -1;
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

float analytical_solution(){
    return 2 * sin(0.5) * sin(0.5) * sin(1);
}

void integral(int N, std::string device_type) {
    const int group_count = 8;
    const int group_size = 8;

    sycl::queue queue;
    if (device_type == "cpu") {
        queue = sycl::queue(sycl::cpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else if (device_type == "gpu") {
        queue = sycl::queue(sycl::gpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else {
        std::cout << "Selector error" << std::endl;
        exit(-1);
    }

    std::vector<float> group_results(group_count * group_count);
    std::fill(group_results.begin(), group_results.end(), 0.0f);
    const int interval_per_work_item = (float)(N / (group_count * group_size));
    const float step = 1.0f / N;
    uint64_t start_time, end_time = 0;
    {
        sycl::buffer<float> group_results_buff(group_results.data(), group_results.size());
        try {
            sycl::event event = queue.submit([&](sycl::handler& cgh) {
                auto group_results_acc = group_results_buff.get_access<sycl::access::mode::write>(cgh);
                cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(group_count*group_size, group_count*group_size), sycl::range<2>(group_size, group_size)), [=](sycl::nd_item<2> item) {

                float begin_x = (float)(interval_per_work_item * item.get_global_id(0)) / (float)N;
                float begin_y = (float)(interval_per_work_item * item.get_global_id(1)) / (float)N;

                float end_x = begin_x + (float)interval_per_work_item * step;
                float end_y = begin_y + (float)interval_per_work_item * step;

                float work_item_res = 0.0f;
                for (float x = begin_x; x <= end_x; x += step) {
                    for (float y = begin_y; y <= end_y;  y += step) {
                        work_item_res += sycl::sin((x + x + step) / 2.0f) * sycl::cos((y + y + step) / 2.0f) * step * step;
                    }
                }



                float group_sum = sycl::reduce_over_group(item.get_group(), work_item_res, std::plus<float>());
                if (item.get_local_id() == 0) {
                    group_results_acc[item.get_group(0) + item.get_group(1)*group_count] = group_sum;
                }
                    
                });
            });
            queue.wait_and_throw();
            start_time = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            end_time = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        } catch (sycl::exception &e) {
            std::cout << e.what() << std::endl;
        }
    }

    float result = std::accumulate(group_results.begin(), group_results.end(), 0.0f);

    std::cout << "Number of rectangles:\t" << N << " x " << N << std::endl;
    std::cout << "Target device:\t\t" << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Kernel time:\t\t" << (end_time - start_time) / 1000000 << " ms" << std::endl;
    std::cout << "Expected:\t\t" << analytical_solution() << std::endl;
    std::cout << "Computed:\t\t" << result << std::endl;
    std::cout << "Difference:\t\t" << fabs(result - analytical_solution()) << std::endl;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    integral(args.first, args.second);
    return 0;
}