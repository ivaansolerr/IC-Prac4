#ifndef __IMAGE__H__
#define __IMAGE__H__
#include <vector>
#include <memory>
#include <iostream>
#include "assert.h"
#include <string>
#include <vector>
#include <cassert>

template <typename T>
class Block;

template <typename T>
class Image
{
public:
    int width, height, channels;
    std::shared_ptr<T[]> matrix;
    void release();
    Image();
    Image(int width, int height, int channels);
    Image(const Image<T> &a);
    ~Image();
    Image<T> operator=(const Image<T> &other);
    Image<T> operator*(const Image<T> &other) const;
    Image<T> operator*(float scalar) const;
    Image<T> operator+(const Image<T> &other) const;
    Image<T> operator+(float scalar) const;
    T get(int row, int col, int channel) const;
    void set(int row, int col, int channel, T value);
    template <typename S>
    Image<S> convert() const;
    Image<T> to_grayscale() const;
    Image<T> abs() const;
    Image<float> normalized() const;
    Image<T> convolution(const Image<float> &kernel) const;
    std::vector<Block<T>> get_blocks(int block_size = 8);
};

Image<unsigned char> load_from_file(const std::string &filename);
void save_to_file(const std::string &filename, const Image<unsigned char> &image, int quality = 100);

template <typename T>
class Block
{
public:
    int i, j, size, depth, rowsize;
    Image<T> *matrix;
    T get_pixel(int row, int col, int channel) const;
    void set_pixel(int row, int col, int channel, T value);
};

template <class T> T Block<T>::get_pixel(int row, int col, int channel) const
{
    assert(row >= 0 && row < size && col >= 0 && col < size);

    return matrix->get(row + j, col + i, channel);
}
template <class T> void Block<T>::set_pixel(int row, int col, int channel, T value)
{
    assert(row >= 0 && row < size && col >= 0 && col < size);
    
    return matrix->set(row + j, col + i, channel, value);
}


// img vacia sin memoria
template <class T> Image<T>::Image()
{
    matrix = NULL;
}

// reserva memoria para img nueva
template <class T> Image<T>::Image(int width, int height, int channels)
{
    this->width = width;
    this->height = height;
    this->channels = channels;
    
    matrix = std::shared_ptr<T[]>(new T[height * width * channels]);
}

// copia comparte los datos SIN DUPLICAR
template <class T> Image<T>::Image(const Image<T> &a)
{
    width = a.width;
    height = a.height;
    channels = a.channels;
    
    if (a.matrix != NULL)
    {
        matrix = a.matrix;
    }
    else
    {
        matrix = NULL;
    }
}

// destructor img
template <class T> Image<T>::~Image()
{
    release();
}

// asigna compartiendo datos
template <class T> Image<T> Image<T>::operator=(const Image<T> &a)
{
    if (this == &a){
        return *this;
    }
    
    release();
    
    
    width = a.width;
    
    height = a.height;
    
    channels = a.channels;

    
    if (a.matrix != NULL)
    {
        matrix = a.matrix;
    }
    else
    {
        matrix = NULL;
    }
    return *this;
}

// libera memo poniendola a null
template <class T> void Image<T>::release()
{
    matrix = NULL;
}

template <class T> T Image<T>::get(int row, int col, int channel) const
{
    return matrix[row * width * channels + col * channels + channel];
}
template <class T> void Image<T>::set(int row, int col, int channel, T value)
{
    matrix[row * width * channels + col * channels + channel] = value;
}


// multiplica pixel a pixel con mpi
template <class T> Image<T> Image<T>::operator*(const Image<T> &other) const
{
    assert(width == other.width && height == other.height && channels == other.channels);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    int rows_per_proc = height / size;
    int remainder = height % size;

    // BALANCEAR CON AJUSTE
    int start_row = rank * rows_per_proc + std::min(rank, remainder);
    
    int ajuste = 0;

    if(rank < remainder){
    
        ajuste = 1;
    }
    else{
        ajuste = 0;
    }
    
    int local_rows = rows_per_proc + ajuste;
    
    int end_row = start_row + local_rows;

    

    Image<T> local_result(width, local_rows, channels);

    


    for (int j = 0; j < local_rows; j++)
    {
        for (int i = 0; i < width; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int global_row = start_row + j;



                T value = this->get(global_row, i, c) * other.get(global_row, i, c);
                
                local_result.set(j, i, c, value);
            }
        }
    }

    
    Image<T> new_image;

    if (rank == 0)
    {
        new_image = Image<T>(width, height, channels);
    }

    

    
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    int elements_per_row = width * channels;

    for (int i = 0; i < size; i++)
    {

        int ajuste2 = 0;

        if(i < remainder){
        
            ajuste2 = 1;
        }
        else{
            ajuste2 = 0;
        }


        int proc_rows = rows_per_proc + ajuste2;
        
        
        recvcounts[i] = proc_rows * elements_per_row;
        
        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * elements_per_row;
    }


    // DETENERMINAR tipo MPI según T

    // pueden haber muchos tipos de paralelizacion
    MPI_Datatype mpi_type;
    
    if (std::is_same<T, float>::value)
        mpi_type = MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        mpi_type = MPI_DOUBLE;
    else if (std::is_same<T, int>::value)
        mpi_type = MPI_INT;
    else if (std::is_same<T, unsigned char>::value)
        mpi_type = MPI_UNSIGNED_CHAR;
    else
        mpi_type = MPI_BYTE;



    int val = 0;

    if (rank == 0){
        val = new_image.get_matrix();
    }
    else{
        val = NULL;
    }
    
    MPI_Gatherv(local_result.get_matrix(), local_rows * elements_per_row, mpi_type, val, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return new_image;
}


// multiplica todos los pixeles x un numero
template <class T> Image<T> Image<T>::operator*(float scalar) const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    int rows_per_proc = height / size;
    int remainder = height % size;

    int start_row = rank * rows_per_proc + std::min(rank, remainder);


    int ajuste = 0;

    if(rank < remainder){
    
        ajuste = 1;
    }
    else{
        ajuste = 0;
    }


    int local_rows = rows_per_proc + ajuste;

    int end_row = start_row + local_rows;

    

    Image<T> local_result(width, local_rows, channels);

    
    for (int j = 0; j < local_rows; j++)
    {
        for (int i = 0; i < width; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int global_row = start_row + j;
                
                T value = (T)(this->get(global_row, i, c) * scalar);
                
                local_result.set(j, i, c, value);
            }
        }
    }

    
    Image<T> new_image;
    
    if (rank == 0)
    {
        new_image = Image<T>(width, height, channels);
    }

    


    std::vector<int> recvcounts(size);
    
    std::vector<int> displs(size);

    int elements_per_row = width * channels;
    
    
    for (int i = 0; i < size; i++)
    {

        int val = 0;

        if(i < remainder){
        
            val = 1;
        }
        else{
            val = 0;
        }
        
        int proc_rows = rows_per_proc + val;
    
        recvcounts[i] = proc_rows * elements_per_row;
    
    
        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * elements_per_row;
    }

    


    // COPIADO DE ARRIBA
    MPI_Datatype mpi_type;
    if (std::is_same<T, float>::value)
        mpi_type = MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        mpi_type = MPI_DOUBLE;
    else if (std::is_same<T, int>::value)
        mpi_type = MPI_INT;
    else if (std::is_same<T, unsigned char>::value)
        mpi_type = MPI_UNSIGNED_CHAR;
    else
        mpi_type = MPI_BYTE;

        

    val = 0;

    if(rank == 0){
        val = new_image.get_matrix();
    }
    else{
        val = NULL;
    }
    
    MPI_Gatherv(local_result.get_matrix(), local_rows * elements_per_row, mpi_type, val, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return new_image;
}

// suma pxl a pxl usando mpi
template <class T> Image<T> Image<T>::operator+(const Image<T> &other) const
{
    assert(width == other.width && height == other.height && channels == other.channels);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int rows_per_proc = height / size;
    int remainder = height % size;

    int start_row = rank * rows_per_proc + std::min(rank, remainder);


    int ajuste = 0;

    if(rank < remainder){
    
        ajuste = 1;
    }
    else{
        ajuste = 0;
    }
    


    int local_rows = rows_per_proc + ajuste;

    
    int local_size = local_rows * width * channels;
    T *local_data = new T[local_size];

    

    int idx = 0;
    for (int j = 0; j < local_rows; j++)
    {
        for (int i = 0; i < width; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int global_row = start_row + j;
                local_data[idx++] = this->get(global_row, i, c) + other.get(global_row, i, c);
            }
        }
    }

    Image<T> new_image;
    T *result_data = NULL;

    if (rank == 0)
    {
        new_image = Image<T>(width, height, channels);
        result_data = new_image.get_matrix();
    }

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++)
    {

        int val = 0;
        if(i < remainder){
        
            val = 1;
        }
        else{
            val = 0;
        }


        int proc_rows = rows_per_proc + val;
        recvcounts[i] = proc_rows * width * channels;
        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * width * channels;
    }

    

    MPI_Datatype mpi_type;
    if (std::is_same<T, float>::value)
        mpi_type = MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        mpi_type = MPI_DOUBLE;
    else if (std::is_same<T, int>::value)
        mpi_type = MPI_INT;
    else if (std::is_same<T, unsigned char>::value)
        mpi_type = MPI_UNSIGNED_CHAR;
    else
        mpi_type = MPI_BYTE;

        
    MPI_Gatherv(local_data, local_size, mpi_type, result_data, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    delete[] local_data;

    return new_image;
}


// suma num a todos los pxl con mpi
template <class T>
Image<T> Image<T>::operator+(float scalar) const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    int rows_per_proc = height / size;
    int remainder = height % size;

    int start_row = rank * rows_per_proc + std::min(rank, remainder);


    int ajuste = 0;
    if(rank < remainder){
    
        ajuste = 1;
    }
    else{
        ajuste = 0;
    }
    


    int local_rows = rows_per_proc + ajuste;

    
    int local_size = local_rows * width * channels;
    T *local_data = new T[local_size];

    
    int idx = 0;
    for (int j = 0; j < local_rows; j++)
    {
        for (int i = 0; i < width; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int global_row = start_row + j;
                local_data[idx++] = (T)(this->get(global_row, i, c) + scalar);
            }
        }
    }

    

    Image<T> new_image;
    T *result_data = NULL;

    if (rank == 0)
    {
        new_image = Image<T>(width, height, channels);
        result_data = new_image.get_matrix();
    }

    

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++)
    {
        int val = 0;

        if(i < remainder){
        
            val = 1;
        }
        else{
            val = 0;
        }
        int proc_rows = rows_per_proc + val;

        recvcounts[i] = proc_rows * width * channels;
        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * width * channels;
    }

    
    MPI_Datatype mpi_type;
    if (std::is_same<T, float>::value)
        mpi_type = MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        mpi_type = MPI_DOUBLE;
    else if (std::is_same<T, int>::value)
        mpi_type = MPI_INT;
    else if (std::is_same<T, unsigned char>::value)
        mpi_type = MPI_UNSIGNED_CHAR;
    else
        mpi_type = MPI_BYTE;

        

    MPI_Gatherv(local_data, local_size, mpi_type, result_data, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    delete[] local_data;

    return new_image;
}




// valor abs de todos pixeles con mpi
template <class T> Image<T> Image<T>::abs() const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    int rows_per_proc = height / size;
    int remainder = height % size;

    int start_row = rank * rows_per_proc + std::min(rank, remainder);


    int ajuste = 0;

    if(rank < remainder){
    
        ajuste = 1;
    }
    else{
        ajuste = 0;
    }
    


    int local_rows = rows_per_proc + ajuste;

    

    int local_size = local_rows * width * channels;
    T *local_data = new T[local_size];

    
    int idx = 0;
    for (int j = 0; j < local_rows; j++)
    {
        for (int i = 0; i < width; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int global_row = start_row + j;
                local_data[idx++] = (T)std::abs(this->get(global_row, i, c));
            }
        }
    }

    
    Image<T> new_image;
    T *result_data = NULL;

    if (rank == 0)
    {
        new_image = Image<T>(width, height, channels);
        result_data = new_image.get_matrix();
    }

    
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++)
    {
        int val = 0;

        if(i < remainder){
        
            val = 1;
        }
        else{
            val = 0;
        }
        int proc_rows = rows_per_proc + val;

        recvcounts[i] = proc_rows * width * channels;
        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * width * channels;
    }

    MPI_Datatype mpi_type;
    if (std::is_same<T, float>::value)
        mpi_type = MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        mpi_type = MPI_DOUBLE;
    else if (std::is_same<T, int>::value)
        mpi_type = MPI_INT;
    else if (std::is_same<T, unsigned char>::value)
        mpi_type = MPI_UNSIGNED_CHAR;
    else
        mpi_type = MPI_BYTE;

    MPI_Gatherv(local_data, local_size, mpi_type, result_data, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    delete[] local_data;

    return new_image;
}

// aplica filtro de convolucion
template <class T> Image<T> Image<T>::convolution(const Image<float> &kernel) const
{
    assert(kernel.width % 2 != 0 && kernel.height % 2 != 0 &&
           kernel.width == kernel.height && kernel.channels == 1);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int kernel_size = kernel.width;
    int halo = kernel_size / 2; 

    
    int rows_per_proc = height / size;
    int remainder = height % size;

    int start_row = rank * rows_per_proc + std::min(rank, remainder);

    int ajuste = 0;

    if(rank < remainder){
    
        ajuste = 1;
    }
    else{
        ajuste = 0;
    }
    


    int local_rows = rows_per_proc + ajuste;

    
    int extended_start = std::max(0, start_row - halo);
    int extended_end = std::min(height, start_row + local_rows + halo);
    int extended_rows = extended_end - extended_start;

    
    float *kernel_data = new float[kernel_size * kernel_size];
    if (rank == 0)
    {
        for (int u = 0; u < kernel_size; u++)
        {
            for (int v = 0; v < kernel_size; v++)
            {
                kernel_data[u * kernel_size + v] = kernel.get(u, v, 0);
            }
        }
    }
    MPI_Bcast(kernel_data, kernel_size * kernel_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    
    int local_size = local_rows * width * channels;
    T *local_data = new T[local_size];

    
    int idx = 0;
    for (int j = 0; j < local_rows; j++)
    {
        for (int i = 0; i < width; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                float sum = 0.0;
                int global_row = start_row + j;

                for (int u = 0; u < kernel_size; u++)
                {
                    for (int v = 0; v < kernel_size; v++)
                    {
                        int s = global_row + u - halo;
                        int t = i + v - halo;

                        
                        if (s < 0 || s >= height || t < 0 || t >= width)
                            continue;

                        sum += this->get(s, t, c) * kernel_data[u * kernel_size + v];
                    }
                }
                local_data[idx++] = (T)(sum / (kernel_size * kernel_size));
            }
        }
    }

    
    Image<T> convolved;
    T *result_data = NULL;

    if (rank == 0)
    {
        convolved = Image<T>(width, height, channels);
        
        result_data = convolved.get_matrix();
    }
    
    std::vector<int> recvcounts(size);

    std::vector<int> displs(size);

    for (int i = 0; i < size; i++)
    {

        int val = 0;

        if(i < remainder){
        
            val = 1;
        }
        else{
            val = 0;
        }

        int proc_rows = rows_per_proc + val;
        
        recvcounts[i] = proc_rows * width * channels;
        
        
        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * width * channels;
    }

    MPI_Datatype mpi_type;
    if (std::is_same<T, float>::value)
        mpi_type = MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        mpi_type = MPI_DOUBLE;
    else if (std::is_same<T, int>::value)
        mpi_type = MPI_INT;
    else if (std::is_same<T, unsigned char>::value)
        mpi_type = MPI_UNSIGNED_CHAR;
    else
        mpi_type = MPI_BYTE;

    MPI_Gatherv(local_data, local_size, mpi_type, result_data, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    delete[] local_data;
    delete[] kernel_data;

    return convolved;
}


// cambia timpo de datos de los pixeles
template <class T>
template <typename S>
Image<S> Image<T>::convert() const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    int rows_per_proc = height / size;
    int remainder = height % size;

    int start_row = rank * rows_per_proc + std::min(rank, remainder);


    int ajuste = 0;

    if(rank < remainder){
    
        ajuste = 1;
    }
    else{
        ajuste = 0;
    }
    
    int local_rows = rows_per_proc + ajuste;

    
    int local_size = local_rows * width * channels;
    S *local_data = new S[local_size];

    
    int idx = 0;
    for (int j = 0; j < local_rows; j++)
    {
        for (int i = 0; i < width; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int global_row = start_row + j;
                local_data[idx++] = (S)this->get(global_row, i, c);
            }
        }
    }

    
    Image<S> new_image;
    S *result_data = NULL;

    if (rank == 0)
    {
        new_image = Image<S>(width, height, channels);
        result_data = new_image.get_matrix();
    }

    
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++)
    {

        int val = 0;

        if(i < remainder){
        
            val = 1;
        }
        else{
            val = 0;
        }

        int proc_rows = rows_per_proc + val;
        recvcounts[i] = proc_rows * width * channels;


        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * width * channels;
    }

    
    MPI_Datatype mpi_type;
    if (std::is_same<S, float>::value)
        mpi_type = MPI_FLOAT;
    else if (std::is_same<S, double>::value)
        mpi_type = MPI_DOUBLE;
    else if (std::is_same<S, int>::value)
        mpi_type = MPI_INT;
    else if (std::is_same<S, unsigned char>::value)
        mpi_type = MPI_UNSIGNED_CHAR;
    else
        mpi_type = MPI_BYTE;

    MPI_Gatherv(local_data, local_size, mpi_type, result_data, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    delete[] local_data;

    return new_image;
}


// convierte de RGB -> escala de grises
template <class T>
Image<T> Image<T>::to_grayscale() const
{
    if (channels == 1){
        return convert<T>();
    }
    
    Image<T> image(width, height, 1);
    
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            image.set(j, i, 0, (T)((0.299 * this->get(j, i, 0) + (0.587 * this->get(j, i, 1)) + (0.114 * this->get(j, i, 2)))));
        }
    }
    return image;
}


// divide img en bloques cuadrados 
template <class T>
std::vector<Block<T>> Image<T>::get_blocks(int block_size)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int depth = channels;
    assert(width % block_size == 0 || height % block_size == 0);
    
    
    int n_blocks_row = height / block_size;
    int n_blocks_col = width / block_size;
    int total_blocks = n_blocks_row * n_blocks_col;
    
    

    int blocks_per_proc = total_blocks / size;
    int remainder = total_blocks % size;
    
    

    int start_block = rank * blocks_per_proc + std::min(rank, remainder);
    
    
    int ajuste = 0;

    if(rank < remainder){
    
        ajuste = 1;
    }
    else{
        ajuste = 0;
    }
    
    int end_block = start_block + blocks_per_proc + ajuste;
    
    std::vector<Block<T>> blocks;
    blocks.reserve(end_block - start_block);
    
    

    int block_idx = 0;
    
    for (int row = 0; row < height; row += block_size)
    {
        for (int col = 0; col < width; col += block_size)
        {
            if (block_idx >= start_block && block_idx < end_block)
            {
                Block<T> b;
                b.i = col;
                b.j = row;
                b.size = block_size;
                b.rowsize = width * channels;
                b.matrix = this;
                b.depth = depth;
                blocks.push_back(b);
            }
            block_idx++;
        }
    }
    
    return blocks;
}
#endif
