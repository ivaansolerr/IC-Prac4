#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "mpi.h"
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>


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
    // Verificar que la imagen es válida
    if (image.width <= 0 || image.height <= 0) {
        if (rank == 0) {
            std::cerr << "Error: Invalid image dimensions in compute_srm_parallel" << std::endl;
        }
        return Image<unsigned char>();
    }
    
    Image<float> grayscale = image.to_grayscale().convert<float>();

    int H = grayscale.height;
    int W = grayscale.width;
    
    if (rank == 0) {
        std::cout << "SRM processing image: " << W << "x" << H << " with " << procs << " processes" << std::endl;
    }

    int rows_per_proc = H / procs;
    int extra = H % procs;

    int start_row = rank * rows_per_proc + std::min(rank, extra);
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);
    
    // Verificar cálculos
    if (local_rows <= 0) {
        std::cerr << "Rank " << rank << ": local_rows = " << local_rows << " (invalid!)" << std::endl;
        local_rows = 0;
    }

    std::vector<float> local_data;
    if (local_rows > 0) {
        local_data.resize(local_rows * W);
    } else {
        local_data.resize(0);
    }

    if(rank == 0) {
        std::vector<int> sendcounts(procs), displs(procs);
        int offset = 0;
        for(int r=0; r<procs; r++){
            int lr = rows_per_proc + (r < extra ? 1 : 0);
            sendcounts[r] = lr * W;
            displs[r] = offset;
            offset += lr * W;
        }
        
        if (local_rows > 0) {
            MPI_Scatterv(grayscale.data(), sendcounts.data(), displs.data(),
                         MPI_FLOAT, local_data.data(), local_rows * W,
                         MPI_FLOAT, 0, MPI_COMM_WORLD);
        } else {
            // Proceso 0 también puede tener 0 filas en algunos casos
            MPI_Scatterv(grayscale.data(), sendcounts.data(), displs.data(),
                         MPI_FLOAT, nullptr, 0,
                         MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    } else {
        if (local_rows > 0) {
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_FLOAT,
                         local_data.data(), local_rows * W, MPI_FLOAT,
                         0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_FLOAT,
                         nullptr, 0, MPI_FLOAT,
                         0, MPI_COMM_WORLD);
        }
    }

    Image<float> local_img;
    if (local_rows > 0) {
        local_img = Image<float>(W, local_rows, 1);
        if (!local_data.empty()) {
            memcpy(local_img.data(), local_data.data(), local_data.size() * sizeof(float));
        }
    }

    Image<float> local_out;
    if (local_rows > 0) {
        local_out = local_img.convolution(get_srm_kernel(kernel_size));
        local_out = local_out.abs().normalized() * 255;
    }

    std::vector<float> full_out;
    std::vector<int> recvcounts_gath(procs);
    std::vector<int> displs_gath(procs);
    
    // Calcular recvcounts y displs para Gatherv
    int offset = 0;
    for(int r = 0; r < procs; r++) {
        int lr = rows_per_proc + (r < extra ? 1 : 0);
        recvcounts_gath[r] = lr * W;
        displs_gath[r] = offset;
        offset += lr * W;
    }
    
    if(rank == 0) {
        full_out.resize(H * W);
    }
    
    // Solo hacer Gatherv si este proceso tiene datos
    if (local_rows > 0) {
        MPI_Gatherv(local_out.data(), local_rows * W, MPI_FLOAT,
                    full_out.data(), recvcounts_gath.data(), displs_gath.data(), 
                    MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(nullptr, 0, MPI_FLOAT,
                    full_out.data(), recvcounts_gath.data(), displs_gath.data(), 
                    MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

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
            std::cerr << "Image filename missing. Usage: ./main <file>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    Image<unsigned char> image;
    int load_success = 0;
    
    if(rank == 0) {
        image = load_from_file(argv[1]);
        load_success = (image.matrix != NULL) ? 1 : 0;
        if (!load_success) {
            std::cerr << "Error: Failed to load image " << argv[1] << std::endl;
        } else {
            std::cout << "Loaded image: " << image.width << "x" << image.height 
                      << " channels: " << image.channels << std::endl;
        }
    }
    
    // Broadcast si la carga fue exitosa
    MPI_Bcast(&load_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (!load_success) {
        MPI_Finalize();
        return 1;
    }

    // Broadcast image dimensions
    int dims[3] = {0, 0, 0};
    if(rank == 0) { 
        dims[0] = image.width; 
        dims[1] = image.height; 
        dims[2] = image.channels;
    }
    
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank != 0) {
        image = Image<unsigned char>(dims[0], dims[1], dims[2]);
    }
    
    // Verificar que las dimensiones son válidas
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        if (rank == 0) {
            std::cerr << "Error: Invalid image dimensions" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Solo hacer broadcast si tenemos datos
    if (dims[0] * dims[1] * dims[2] > 0) {
        MPI_Bcast(image.data(), dims[0]*dims[1]*dims[2], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    // Opción 1: Solo probar ELA primero (más simple)
    if(rank == 0) {
        std::cout << "\n1. Computing ELA..." << std::endl;
    }
    auto ela_start = std::chrono::high_resolution_clock::now();
    if(rank == 0) {
        auto ela = compute_ela(image, 90);
        save_to_file("ela.png", ela);
    }
    auto ela_end = std::chrono::high_resolution_clock::now();
    
    if(rank == 0) {
        auto ela_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ela_end - ela_start);
        std::cout << "   ELA completed in " << ela_duration.count() << " ms" << std::endl;
    }
    
    // Opción 2: Probar SRM 3x3
    if(rank == 0) {
        std::cout << "\n2. Computing SRM 3x3..." << std::endl;
    }
    auto srm3_start = std::chrono::high_resolution_clock::now();
    Image<unsigned char> srm3 = compute_srm_parallel(image, 3, rank, procs);
    auto srm3_end = std::chrono::high_resolution_clock::now();
    
    if(rank == 0) {
        save_to_file("srm_kernel_3x3.png", srm3);
        auto srm3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(srm3_end - srm3_start);
        std::cout << "   SRM 3x3 completed in " << srm3_duration.count() << " ms" << std::endl;
    }

    // Opción 3: Probar SRM 5x5
    if(rank == 0) {
        std::cout << "\n3. Computing SRM 5x5..." << std::endl;
    }
    auto srm5_start = std::chrono::high_resolution_clock::now();
    Image<unsigned char> srm5 = compute_srm_parallel(image, 5, rank, procs);
    auto srm5_end = std::chrono::high_resolution_clock::now();
    
    if(rank == 0) {
        save_to_file("srm_kernel_5x5.png", srm5);
        auto srm5_duration = std::chrono::duration_cast<std::chrono::milliseconds>(srm5_end - srm5_start);
        std::cout << "   SRM 5x5 completed in " << srm5_duration.count() << " ms" << std::endl;
    }

    // Opción 4: Probar DCT (comentado si hay problemas)
    if(rank == 0) {
        std::cout << "\n4. Computing DCT..." << std::endl;
    }
    try {
        auto dct_start = std::chrono::high_resolution_clock::now();
        auto dct_inv = compute_dct_parallel(image, 8, true, rank, procs);
        auto dct_end = std::chrono::high_resolution_clock::now();
        
        if(rank == 0) {
            save_to_file("dct_invert.png", dct_inv);
            auto dct_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dct_end - dct_start);
            std::cout << "   DCT completed in " << dct_duration.count() << " ms" << std::endl;
        }
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "   DCT failed: " << e.what() << std::endl;
        }
    }

    if(rank == 0) {
        std::cout << "\nAll processing completed!" << std::endl;
        std::cout << "Generated files:" << std::endl;
        std::cout << "  - ela.png" << std::endl;
        std::cout << "  - srm_kernel_3x3.png" << std::endl;
        std::cout << "  - srm_kernel_5x5.png" << std::endl;
        std::cout << "  - dct_invert.png" << std::endl;
        std::cout << "  - _temp.jpg (temporary, can be deleted)" << std::endl;
    }

    MPI_Finalize();
    return 0;
}