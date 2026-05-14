/*
 * CUDA加速版本 - 拉普拉斯本征值计算
 * 
 * 注意：需要安装CUDA Toolkit
 * 编译：nvcc -c cuda_laplacian.cu -o cuda_laplacian.o
 * 链接：gcc ... cuda_laplacian.o -lcudart
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_NODES 1000
#define MAX_EIGENVECTORS 50

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

__global__ void apply_laplacian_kernel(
    const int num_nodes,
    const int num_edges,
    const int *__restrict__ edge_from,
    const int *__restrict__ edge_to,
    const int *__restrict__ in_degree,
    const int *__restrict__ out_degree,
    const double *__restrict__ v_in,
    double *__restrict__ v_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_nodes) {
        double degree_i = in_degree[i] + out_degree[i];
        v_out[i] = degree_i * v_in[i];
    }
}

__global__ void apply_laplacian_edges_kernel(
    const int num_edges,
    const int *__restrict__ edge_from,
    const int *__restrict__ edge_to,
    const double *__restrict__ v_in,
    double *__restrict__ v_out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (e < num_edges) {
        int u = edge_from[e];
        int v = edge_to[e];
        
        atomicAdd(&v_out[u], -v_in[v]);
        atomicAdd(&v_out[v], -v_in[u]);
    }
}

__global__ void normalize_kernel(
    const int n,
    double *__restrict__ v,
    double *__restrict__ norm_result
) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = 0.0;
    if (i < n) {
        sdata[tid] = v[i] * v[i];
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        norm_result[blockIdx.x] = sdata[0];
    }
}

__global__ void scale_kernel(
    const int n,
    double *__restrict__ v,
    const double scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        v[i] *= scale;
    }
}

__global__ void dot_product_kernel(
    const int n,
    const double *__restrict__ v1,
    const double *__restrict__ v2,
    double *__restrict__ result
) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = 0.0;
    if (i < n) {
        sdata[tid] = v1[i] * v2[i];
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

extern "C" void compute_laplacian_spectrum_cuda(
    const int num_nodes,
    const int num_edges,
    const int *edge_from,
    const int *edge_to,
    const int *in_degree,
    const int *out_degree,
    double *eigenvalues,
    double *eigenvectors,
    int num_eigenvalues
) {
    int *d_edge_from, *d_edge_to, *d_in_degree, *d_out_degree;
    double *d_v, *d_Lv, *d_temp, *d_norm_result;
    
    size_t nodes_size = num_nodes * sizeof(int);
    size_t edges_size = num_edges * sizeof(int);
    size_t double_nodes_size = num_nodes * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_edge_from, edges_size));
    CUDA_CHECK(cudaMalloc(&d_edge_to, edges_size));
    CUDA_CHECK(cudaMalloc(&d_in_degree, nodes_size));
    CUDA_CHECK(cudaMalloc(&d_out_degree, nodes_size));
    CUDA_CHECK(cudaMalloc(&d_v, double_nodes_size));
    CUDA_CHECK(cudaMalloc(&d_Lv, double_nodes_size));
    CUDA_CHECK(cudaMalloc(&d_temp, double_nodes_size));
    CUDA_CHECK(cudaMalloc(&d_norm_result, double_nodes_size));
    
    CUDA_CHECK(cudaMemcpy(d_edge_from, edge_from, edges_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_to, edge_to, edges_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_degree, in_degree, nodes_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_degree, out_degree, nodes_size, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocksNodes = (num_nodes + blockSize - 1) / blockSize;
    int numBlocksEdges = (num_edges + blockSize - 1) / blockSize;
    
    eigenvalues[0] = 0.0;
    double norm = 1.0 / sqrt((double)num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        eigenvectors[i] = norm;
    }
    
    for (int k = 1; k < num_eigenvalues; k++) {
        static double h_v[MAX_NODES];
        for (int i = 0; i < num_nodes; i++) {
            h_v[i] = ((double)rand() / RAND_MAX - 0.5);
        }
        
        CUDA_CHECK(cudaMemcpy(d_v, h_v, double_nodes_size, cudaMemcpyHostToDevice));
        
        double lambda = 0.0;
        for (int iter = 0; iter < 100; iter++) {
            CUDA_CHECK(cudaMemset(d_Lv, 0, double_nodes_size));
            
            apply_laplacian_kernel<<<numBlocksNodes, blockSize>>>(
                num_nodes, num_edges, d_edge_from, d_edge_to,
                d_in_degree, d_out_degree, d_v, d_Lv
            );
            
            apply_laplacian_edges_kernel<<<numBlocksEdges, blockSize>>>(
                num_edges, d_edge_from, d_edge_to, d_v, d_Lv
            );
            
            CUDA_CHECK(cudaMemcpy(h_v, d_Lv, double_nodes_size, cudaMemcpyDeviceToHost));
            
            double norm_sq = 0.0;
            for (int i = 0; i < num_nodes; i++) {
                norm_sq += h_v[i] * h_v[i];
            }
            double norm_val = sqrt(norm_sq);
            
            if (norm_val > 1e-15) {
                double scale = 1.0 / norm_val;
                scale_kernel<<<numBlocksNodes, blockSize>>>(num_nodes, d_Lv, scale);
            }
            
            CUDA_CHECK(cudaMemcpy(h_v, d_v, double_nodes_size, cudaMemcpyDeviceToHost));
            static double h_Lv[MAX_NODES];
            CUDA_CHECK(cudaMemcpy(h_Lv, d_Lv, double_nodes_size, cudaMemcpyDeviceToHost));
            
            lambda = 0.0;
            for (int i = 0; i < num_nodes; i++) {
                lambda += h_v[i] * h_Lv[i];
            }
            
            CUDA_CHECK(cudaMemcpy(d_temp, d_v, double_nodes_size, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_v, d_Lv, double_nodes_size, cudaMemcpyDeviceToDevice));
        }
        
        eigenvalues[k] = lambda;
        CUDA_CHECK(cudaMemcpy(&eigenvectors[k * num_nodes], d_v, double_nodes_size, cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaFree(d_edge_from));
    CUDA_CHECK(cudaFree(d_edge_to));
    CUDA_CHECK(cudaFree(d_in_degree));
    CUDA_CHECK(cudaFree(d_out_degree));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_Lv));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_norm_result));
}
