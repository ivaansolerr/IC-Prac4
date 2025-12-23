#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include "mpi.h"

Image<float> get_srm_3x3() {
    Image<float> kernel(3,3,1);
    kernel.set(0,0,0,-1); kernel.set(0,1,0,2); kernel.set(0,2,0,-1);
    kernel.set(1,0,0,2);  kernel.set(1,1,0,-4); kernel.set(1,2,0,2);
    kernel.set(2,0,0,-1); kernel.set(2,1,0,2); kernel.set(2,2,0,-1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5,5,1);
    int vals[5][5] = {
        {-1,2,-2,2,-1},
        { 2,-6,8,-6,2},
        {-2,8,-12,8,-2},
        { 2,-6,8,-6,2},
        {-1,2,-2,2,-1}
    };
    for(int i=0;i<5;i++)
        for(int j=0;j<5;j++)
            kernel.set(i,j,0,vals[i][j]);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    return size == 3 ? get_srm_3x3() : get_srm_5x5();
}

Image<unsigned char> compute_srm_parallel(const Image<unsigned char>& image, int kernel_size, int rank, int procs) {
    Image<float> grayscale = image.to_grayscale().convert<float>();

    int H = grayscale.height();
    int W = grayscale.width();

    int rows_per_proc = H / procs;
    int extra = H % procs;

    int start_row = rank * rows_per_proc + std::min(rank, extra);
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);

    std::vector<float> local_data(local_rows * W);

    if(rank == 0) {
        std::vector<int> sendcounts(procs), displs(procs);
        int offset = 0;
        for(int r=0;r<procs;r++){
            int lr = rows_per_proc + (r < extra ? 1 : 0);
            sendcounts[r] = lr * W;
            displs[r] = offset;
            offset += lr * W;
        }
        MPI_Scatterv(grayscale.data(), sendcounts.data(), displs.data(),
                     MPI_FLOAT, local_data.data(), local_rows * W,
                     MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_FLOAT,
                     local_data.data(), local_rows * W, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
    }

    Image<float> local_img(W, local_rows, 1);
    memcpy(local_img.data(), local_data.data(), local_data.size() * sizeof(float));

    Image<float> local_out = local_img.convolution(get_srm_kernel(kernel_size));
    local_out = local_out.abs().normalized() * 255;

    std::vector<float> full_out;
    if(rank == 0) full_out.resize(H * W);

    MPI_Gatherv(local_out.data(), local_rows * W, MPI_FLOAT,
                full_out.data(), nullptr, nullptr, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if(rank == 0) {
        Image<float> assembled(W, H, 1);
        memcpy(assembled.data(), full_out.data(), full_out.size() * sizeof(float));
        return assembled.convert<unsigned char>();
    }
    return Image<unsigned char>(); // empty on other ranks
}

Image<unsigned char> compute_dct_parallel(const Image<unsigned char>& image,
                                          int block_size, bool invert,
                                          int rank, int procs)
{
    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);

    int total_blocks = blocks.size();
    int blocks_pp = total_blocks / procs;
    int extra = total_blocks % procs;

    int start = rank * blocks_pp + std::min(rank, extra);
    int count = blocks_pp + (rank < extra ? 1 : 0);

    for(int i=start; i < start+count; i++){
        float** dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);

        if (invert) {
            for(int k=0;k<block_size/2;k++)
                for(int l=0;l<block_size/2;l++)
                    dctBlock[k][l] = 0.0;
            dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, blocks[i], 0);
        }
        dct::delete_matrix(dctBlock);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        Image<unsigned char> result = grayscale.convert<unsigned char>();
        return result;
    }
    return Image<unsigned char>();
}

Image<unsigned char> compute_ela(const Image<unsigned char>& image, int quality){
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>() * (-1));
    compressed = compressed.abs().normalized() * 255;
    return compressed.convert<unsigned char>();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    if(argc < 2) {
        if(rank == 0)
            std::cerr<<"Image filename missing. Usage: ./dct <file>"<<std::endl;
        MPI_Finalize();
        return 1;
    }

    Image<unsigned char> image;
    if(rank == 0)
        image = load_from_file(argv[1]);

    // Broadcast image dimensions first
    int dims[2];
    if(rank == 0) { dims[0] = image.width(); dims[1] = image.height(); }
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0)
        image = Image<unsigned char>(dims[0], dims[1], 3);

    MPI_Bcast(image.data(), dims[0]*dims[1]*3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // PARALLEL SRM
    Image<unsigned char> srm3;
    if(rank == 0) std::cout<<"Parallel SRM 3x3..."<<std::endl;
    srm3 = compute_srm_parallel(image, 3, rank, procs);
    if(rank == 0) save_to_file("srm_kernel_3x3.png", srm3);

    // PARALLEL SRM 5x5
    Image<unsigned char> srm5;
    if(rank == 0) std::cout<<"Parallel SRM 5x5..."<<std::endl;
    srm5 = compute_srm_parallel(image, 5, rank, procs);
    if(rank == 0) save_to_file("srm_kernel_5x5.png", srm5);

    // SERIAL ELA
    if(rank == 0) {
        auto ela = compute_ela(image, 90);
        save_to_file("ela.png", ela);
    }

    // PARALLEL DCT
    if(rank == 0) std::cout<<"Parallel DCT..."<<std::endl;
    auto dct_inv = compute_dct_parallel(image, 8, true, rank, procs);
    if(rank == 0) save_to_file("dct_invert.png", dct_inv);

    auto dct_dir = compute_dct_parallel(image, 8, false, rank, procs);
    if(rank == 0) save_to_file("dct_direct.png", dct_dir);

    MPI_Finalize();
    return 0;
}