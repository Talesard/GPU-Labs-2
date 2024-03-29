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

float base_accuracy(std::vector<float> xk, std::vector<float> xk1) {
    float sum = 0.0f;
    for (int i = 0; i < xk.size(); i++) {
        float tmp = xk[i] - xk1[i];
        sum += tmp * tmp;
    }
    return std::sqrt(sum);
}

float relative_accuracy(std::vector<float> xk, std::vector<float> xk1) {
    float top = 0.0f;
    float bot = 0.0f;
    for (int i = 0; i < xk.size(); i++) {
        float tmp = xk[i] - xk1[i];
        top += tmp * tmp;
        bot += xk[i] * xk[i];
    }
    return std::sqrt(top) / std::sqrt(bot);
}

float achived_accuracy(std::vector<float> A, std::vector<float> b, std::vector<float> x) {
    float result = 0.0f;
    for (int i = 0; i < b.size(); i++) {
        float tmp = 0.0f - b[i];
        for (int j = 0; j < b.size(); j++) {
            tmp += A[j * b.size() + i] * x[j];
        }
        result += tmp * tmp;
    }
    return std::sqrt(result);
}

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

void print_results(std::string version, long long time, float accuracy, float accuracy_while, int iters, int max_iters) {
    std::cout << "[" << version << "] " << "Time: " << time << " ms Accuracy: " << accuracy << " (" << accuracy_while << ")" << " Iters: " << iters << " / " << max_iters << std::endl;
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

    std::vector<float> xk;
    std::vector<float> xk1 = b;

    int iter_counter = 0;
    float accuracy = 0.0f;

    auto start_time = std::chrono::steady_clock::now();

    do {
        iter_counter++;
        xk = xk1;
        {
            sycl::buffer<float> xk_buff(xk.data(), xk.size());
            sycl::buffer<float> xk1_buff(xk1.data(), xk1.size());
            sycl::event event = queue.submit([&](sycl::handler &cgh) {
                auto A_acc = A_buff.get_access<sycl::access::mode::read>(cgh);
                auto b_acc = b_buff.get_access<sycl::access::mode::read>(cgh);
                auto xk_acc = xk_buff.get_access<sycl::access::mode::read>(cgh);
                auto xk1_acc = xk1_buff.get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for(sycl::range<1>(b.size()), [=](sycl::item<1> item) {
                    int i = item.get_id(0);
                    int n = item.get_range(0);
                    float sum = 0.0f;
                    for (int j = 0; j < n; j++) {
                        if (i != j) {
                            sum += A_acc[i + j * n] * xk_acc[j];
                        }
                    }
                    xk1_acc[i] = (b_acc[i] - sum) / A_acc[i + i * n];
                });
            });
            queue.wait();
        }
        accuracy = relative_accuracy(xk, xk1);
    } while (iter_counter < max_iters && accuracy > target_accuracy);

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    float final_accuracy = achived_accuracy(A, b, xk1);

    std::cout << "Target device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    print_results("Accessors", elapsed_ms.count(), final_accuracy, accuracy, iter_counter, max_iters);

    return xk1;
}

std::vector<float> jacobi_shared_mem(int N, float target_accuracy, int max_iters, std::string device_type, std::vector<float> A, std::vector<float> b) {
    sycl::queue queue;
    if (device_type == "cpu") {
        queue = sycl::queue(sycl::cpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else if (device_type == "gpu") {
        queue = sycl::queue(sycl::gpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else {
        std::cout << "Selector error" << std::endl;
        exit(-1);
    }

    float* A_shared = sycl::malloc_shared<float>(A.size(), queue);
    float* b_shared = sycl::malloc_shared<float>(b.size(), queue);
    std::vector<float> xk(b.size());
    float* xk_shared = sycl::malloc_shared<float>(xk.size(), queue);
    std::vector<float> xk1 = b;
    float* xk1_shared = sycl::malloc_shared<float>(xk1.size(), queue);

    queue.memcpy(A_shared, A.data(), A.size()*sizeof(float)).wait();
    queue.memcpy(b_shared, b.data(), b.size()*sizeof(float)).wait();
    queue.memcpy(xk_shared, xk.data(), xk.size()*sizeof(float)).wait();
    queue.memcpy(xk1_shared, xk1.data(), xk1.size()*sizeof(float)).wait();
    


    int iter_counter = 0;
    float accuracy = 0.0f;

    auto start_time = std::chrono::steady_clock::now();

    do {
        iter_counter++;
        queue.memcpy(xk_shared, xk1_shared, xk.size()*sizeof(float)).wait();
        queue.parallel_for(sycl::range<1>(b.size()), [=](sycl::item<1> item) {
            int i = item.get_id(0);
            int n = item.get_range(0);
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum += A_shared[i + j * n] * xk_shared[j];
                }
            }
            xk1_shared[i] = (b_shared[i] - sum) / A_shared[i + i * n];
        }).wait();
        queue.memcpy(xk.data(), xk_shared, xk.size()*sizeof(float)).wait();
        queue.memcpy(xk1.data(), xk1_shared, xk1.size()*sizeof(float)).wait();
        accuracy = relative_accuracy(xk, xk1);
    } while (iter_counter < max_iters && accuracy > target_accuracy);

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    float final_accuracy = achived_accuracy(A, b, xk1);

    sycl::free(A_shared, queue);
    sycl::free(b_shared, queue);
    sycl::free(xk_shared, queue);
    sycl::free(xk1_shared, queue);

    print_results("  Shared ", elapsed_ms.count(), final_accuracy, accuracy, iter_counter, max_iters);

    return xk1;

}

std::vector<float> jacobi_device_mem(int N, float target_accuracy, int max_iters, std::string device_type, std::vector<float> A, std::vector<float> b) {
    sycl::queue queue;
    if (device_type == "cpu") {
        queue = sycl::queue(sycl::cpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else if (device_type == "gpu") {
        queue = sycl::queue(sycl::gpu_selector{}, {sycl::property::queue::enable_profiling()});
    } else {
        std::cout << "Selector error" << std::endl;
        exit(-1);
    }

    float* A_device = sycl::malloc_device<float>(A.size(), queue);
    float* b_device = sycl::malloc_device<float>(b.size(), queue);
    std::vector<float> xk(b.size());
    float* xk_device = sycl::malloc_device<float>(xk.size(), queue);
    std::vector<float> xk1 = b;
    float* xk1_device = sycl::malloc_device<float>(xk1.size(), queue);

    queue.memcpy(A_device, A.data(), A.size()*sizeof(float)).wait();
    queue.memcpy(b_device, b.data(), b.size()*sizeof(float)).wait();
    queue.memcpy(xk_device, xk.data(), xk.size()*sizeof(float)).wait();
    queue.memcpy(xk1_device, xk1.data(), xk1.size()*sizeof(float)).wait();
    


    int iter_counter = 0;
    float accuracy = 0.0f;

    auto start_time = std::chrono::steady_clock::now();

    do {
        iter_counter++;
        queue.memcpy(xk_device, xk1_device, xk.size()*sizeof(float)).wait();
        queue.parallel_for(sycl::range<1>(b.size()), [=](sycl::item<1> item) {
            int i = item.get_id(0);
            int n = item.get_range(0);
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum += A_device[i + j * n] * xk_device[j];
                }
            }
            xk1_device[i] = (b_device[i] - sum) / A_device[i + i * n];
        }).wait();
        queue.memcpy(xk.data(), xk_device, xk.size()*sizeof(float)).wait();
        queue.memcpy(xk1.data(), xk1_device, xk1.size()*sizeof(float)).wait();
        accuracy = relative_accuracy(xk, xk1);
    } while (iter_counter < max_iters && accuracy > target_accuracy);

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    float final_accuracy = achived_accuracy(A, b, xk1);

    sycl::free(A_device, queue);
    sycl::free(b_device, queue);
    sycl::free(xk_device, queue);
    sycl::free(xk1_device, queue);

    print_results("  Device ", elapsed_ms.count(), final_accuracy, accuracy, iter_counter, max_iters);

    return xk1;
}

float* jacobi_seq(float* A, float* b, int n, float targetAcc, int maxIters) {
    float accuracy = 0.0f;
    int iter_counter = 0;
    float* xk = new float[n];
    float* xk1 = new float[n];
    for (int i = 0; i < n; i++) {
        xk1[i] = b[i];
        xk[i] = 0.0f;
    }
    auto start_time = std::chrono::steady_clock::now();
    do {
        std::swap(xk, xk1);
        iter_counter++;
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum += A[j * n + i] * xk[j];
                }  
            }
            xk1[i] = (1.0f / A[i * n + i]) * (b[i] - sum);
        }
        accuracy = relative_accuracy(std::vector(xk, xk+n), std::vector(xk1, xk1+n));
    } while (iter_counter < maxIters && accuracy > targetAcc);
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    float final_accuracy = achived_accuracy(std::vector<float>(A, A+n*n), std::vector<float>(b, b+n), std::vector<float>(xk1, xk1+n));
    print_results(" Seq cpu ", elapsed_ms.count(), final_accuracy, accuracy, iter_counter, maxIters);
    return xk1;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {std::cout << "Args error. Expected: N, accuracy, maxiters, device" << std::endl; exit(-1);}
    int N = atoi(argv[1]);
    float target_accuracy = std::stof(argv[2]);
    int max_iters = atoi(argv[3]);
    std::string device = argv[4];

    auto system = get_random_system(N);
    auto res_accessors = jacobi_accessors(N, target_accuracy, max_iters, device, system.first, system.second);
    auto res_shared_mem = jacobi_shared_mem(N, target_accuracy, max_iters, device, system.first, system.second);
    auto res_device_mem = jacobi_device_mem(N, target_accuracy, max_iters, device, system.first, system.second);
    //auto res_seq = jacobi_seq(system.first.data(), system.second.data(), N, target_accuracy, max_iters);
    assert(res_accessors == res_shared_mem);
    assert(res_accessors == res_device_mem);

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         std::cout << system.first[i * N + j] << " ";
    //     }
    //     std::cout << " | " << system.second[i] << std::endl;
    // }

    // for (auto v : res_accessors) std::cout << v << " ";

    return 0;
}