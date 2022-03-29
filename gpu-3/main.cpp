#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <numeric>
#include <random>

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

// ---------- WTF ----------
static float vectorLength(const float *x, size_t n) {
    float s = 0;
    for (size_t i = 0; i < n; i++) {
        s += x[i] * x[i];
    }
    return std::sqrt(s);
}

static float normAbs(const float *x0, const float *x1, size_t n) {
    float s = 0;
    for (size_t i = 0; i < n; i++) {
        s += (x0[i] - x1[i]) * (x0[i] - x1[i]);
    }
    return std::sqrt(s);
}

static float normRel(const float *x0, const float *x1, size_t n) {
    return normAbs(x0, x1, n) / vectorLength(x0, n);
}

float norm(const std::vector<float> &x0, const std::vector<float> &x1) {
    return normRel(x0.data(), x1.data(), x0.size());
}

static float deviationAbs(const float *a, const float *b, const float *x, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) {
            s += a[j * n + i] * x[j];
        }
        s -= b[i];
        norm += s * s;
    }
    return sqrt(norm);
}

// ---------- WTF ----------

std::vector<float> random_vector(int size, float min, float max) {
    std::mt19937 gen(time(0));
    std::vector<float> result(size);
    std::uniform_real_distribution<float> distr(min, max);
    for (int i = 0; i < size; i++) {
        result[i] = distr(gen);
    }
    return result;
}

std::pair<std::vector<float>, std::vector<float>> get_random_system(int N) {
    std::vector<float> A = random_vector(N * N, 1.0f, 3.0f);
    std::vector<float> b = random_vector(N, 1.0f, 3.0f);
    std::vector<float> diag = random_vector(N, N * 5.0f, N * 5.0f + 2.0f);
    for (int i = 0; i < N; i++) {
        A[i * N + i] = diag[i];
    }
    return std::pair<std::vector<float>, std::vector<float>>{A, b};
}

std::vector<float> jacobi_accessors(int N, float target_accuracy, int max_iters, std::string device_type, std::vector<float> A, std::vector<float> b) {
    sycl::queue queue;
    if (device_type == "cpu") {
        queue = sycl::queue(sycl::cpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else if (device_type == "gpu") {
        queue = sycl::queue(sycl::gpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else {
        std::cout << "Selector error" << std::endl;
        exit(-1);
    }

    sycl::buffer<float> A_buff(A.data(), A.size());
    sycl::buffer<float> b_buff(b.data(), b.size());

    std::vector<float> x0;
    std::vector<float> x1 = b;

    int iter_counter = 0;
    float accuracy = 0.0f;

    auto start_time = std::chrono::steady_clock::now();

    do {
        iter_counter++;
        x0 = x1;
        {
            sycl::buffer<float> x0_buff(x0.data(), x0.size());
            sycl::buffer<float> x1_buff(x1.data(), x1.size());
            sycl::event event = queue.submit([&](sycl::handler &cgh) {
                auto A_acc = A_buff.get_access<sycl::access::mode::read>(cgh);
                auto b_acc = b_buff.get_access<sycl::access::mode::read>(cgh);
                auto x0_acc = x0_buff.get_access<sycl::access::mode::read>(cgh);
                auto x1_acc = x1_buff.get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for(sycl::range<1>(b.size()), [=](sycl::item<1> item) {
                    int i = item.get_id(0);
                    int n = item.get_range(0);
                    float s = 0.0f;
                    for (int j = 0; j < n; j++) {
                        s += i != j ? A_acc[j*n+i] * x0_acc[j] : 0;
                    }
                    x1_acc[i] = (b_acc[i]-s) / A_acc[i*n+i];
                });
            });
            queue.wait();
        }
        accuracy = norm(x0, x1);
    } while (iter_counter < max_iters && accuracy > target_accuracy);

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    float final_accuracy = deviationAbs(A.data(), b.data(), x1.data(), N);

    std::cout << "[Accessors]" << std::endl;
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Time: " << elapsed_ms.count() << " ms" << std::endl;
    std::cout << "Iters: " << iter_counter << " / " << max_iters << std::endl;
    std::cout << "Accuracy " << final_accuracy << " / " << target_accuracy << std::endl;

    return x1;
}

void jacobi_shared_mem(int N, float target_accuracy, int max_iters, std::string device_type, std::vector<float> A, std::vector<float> b) {
    sycl::queue queue;
    if (device_type == "cpu") {
        queue = sycl::queue(sycl::cpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else if (device_type == "gpu") {
        queue = sycl::queue(sycl::gpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else {
        std::cout << "Selector error" << std::endl;
        exit(-1);
    }

}

void jacobi_device_mem(int N, float target_accuracy, int max_iters, std::string device_type, std::vector<float> A, std::vector<float> b) {
    sycl::queue queue;
    if (device_type == "cpu") {
        queue = sycl::queue(sycl::cpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else if (device_type == "gpu") {
        queue = sycl::queue(sycl::gpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else {
        std::cout << "Selector error" << std::endl;
        exit(-1);
    }

}

int main(int argc, char* argv[]) {
    if (argc != 5) {std::cout << "Args error. Expected: N, accuracy, maxiters, device" << std::endl; exit(-1);}
    int N = atoi(argv[1]);
    float target_accuracy = std::stof(argv[2]);
    int max_iters = atoi(argv[3]);
    std::string device = argv[4];

    auto system = get_random_system(N);
    auto res = jacobi_accessors(N, target_accuracy, max_iters, device, system.first, system.second);

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         std::cout << system.first[i * N + j] << " ";
    //     }
    //     std::cout << " | " << system.second[i] << std::endl;
    // }

    // for (auto v : res) std::cout << v << " ";

    return 0;
}