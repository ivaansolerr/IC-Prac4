#include "png.h"
#include "image.h"
#include "jpeglib.h"
#include <filesystem>

Image<unsigned char> read_png(const std::string &filename) {
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return Image<unsigned char>();
    }
    
    // Verificar firma PNG
    unsigned char header[8];
    size_t bytes_read = fread(header, 1, 8, fp);
    if (bytes_read != 8 || png_sig_cmp(header, 0, 8) != 0) {
        std::cerr << "Error: " << filename << " is not a valid PNG file" << std::endl;
        fclose(fp);
        return Image<unsigned char>();
    }
    
    // Crear estructuras PNG
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        std::cerr << "Error: png_create_read_struct failed" << std::endl;
        fclose(fp);
        return Image<unsigned char>();
    }
    
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Error: png_create_info_struct failed" << std::endl;
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return Image<unsigned char>();
    }
    
    // Configurar manejo de errores
    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "Error: PNG lib error" << std::endl;
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return Image<unsigned char>();
    }
    
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    
    png_read_info(png_ptr, info_ptr);
    
    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    
    // Configurar transformaciones
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    
    // Convertir a RGB o RGBA
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);
    
    png_read_update_info(png_ptr, info_ptr);
    
    // Determinar número de canales
    int channels = png_get_channels(png_ptr, info_ptr);
    
    // Crear imagen
    Image<unsigned char> image(width, height, channels);
    
    // Leer filas
    png_bytep *row_pointers = new png_bytep[height];
    for (int y = 0; y < height; y++) {
        row_pointers[y] = new png_byte[png_get_rowbytes(png_ptr, info_ptr)];
    }
    
    png_read_image(png_ptr, row_pointers);
    
    // Copiar datos a la imagen
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                image.set(y, x, c, row[x * channels + c]);
            }
        }
        delete[] row_pointers[y];
    }
    delete[] row_pointers;
    
    // Limpiar
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    
    return image;
}

void write_png(const std::string &filename, const Image<unsigned char> &image) {
  int y;

  FILE *fp = fopen(filename.c_str(), "wb");
  if(!fp) abort();

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) abort();

  png_infop info = png_create_info_struct(png);
  if (!info) abort();

  if (setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);
   
  png_byte color_type;
  switch(image.channels){
    case 1:
        color_type = PNG_COLOR_TYPE_GRAY;
        break;
    case 3:
        color_type = PNG_COLOR_TYPE_RGB;
    case 4:
        color_type = PNG_COLOR_TYPE_RGBA;
  }

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(
    png,
    info,
    image.width, image.height,
    8,
    color_type,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info(png, info);

  // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
  // Use png_set_filler().
  // png_set_filler(png, 0, PNG_FILLER_AFTER);
  
  unsigned char **rows = new unsigned char*[image.height];
  for(int i=0;i<image.height;i++)
    rows[i] = new unsigned char[image.width*image.channels];

  for(int j=0;j<image.height;j++)
    for(int i=0; i<image.width;i++)
        for(int c=0;c<image.channels;c++)
            rows[j][i*image.channels + c] = image.get(j,i,c); 

  png_write_image(png, rows);
  png_write_end(png, NULL);
  
  for(int i=0;i<image.height;i++)
    delete [] rows[i];
  delete [] rows;

  fclose(fp);
}


Image<unsigned char> read_jpeg(const std::string &filename) {
	struct jpeg_decompress_struct info;
	struct jpeg_error_mgr err;

	unsigned long int imgWidth, imgHeight;
	int numComponents;

	unsigned long int dwBufferBytes;

	unsigned char* lpRowBuffer[1];

	FILE* fHandle;

	fHandle = fopen(filename.c_str(), "rb");
	if(fHandle == NULL) {
		#ifdef DEBUG
			fprintf(stderr, "%s:%u: Failed to read file %s\n", __FILE__, __LINE__, lpFilename);
		#endif
		return Image<unsigned char>();
	}

	info.err = jpeg_std_error(&err);
	jpeg_create_decompress(&info);

	jpeg_stdio_src(&info, fHandle);
	jpeg_read_header(&info, TRUE);

	jpeg_start_decompress(&info);
	imgWidth = info.output_width;
	imgHeight = info.output_height;
	numComponents = info.num_components;

	dwBufferBytes = imgWidth * imgHeight * numComponents; /* We only read RGB, not A */
	Image<unsigned char> image(imgWidth, imgHeight, numComponents);

	/* Read scanline by scanline */
	while(info.output_scanline < info.output_height) {
		lpRowBuffer[0] = (unsigned char *)(&image.matrix[numComponents*info.output_width*info.output_scanline]);
		jpeg_read_scanlines(&info, lpRowBuffer, 1);
	}

	jpeg_finish_decompress(&info);
	jpeg_destroy_decompress(&info);
	fclose(fHandle);

	return image;
}

int write_jpeg(const std::string &filename, const Image<unsigned char> &image, int quality=90) {
    assert(image.channels == 3 || image.channels == 1);
    assert(quality>=10 && quality<=100);
	struct jpeg_compress_struct info;
	struct jpeg_error_mgr err;

	unsigned char* lpRowBuffer[1];

	FILE* fHandle;

	fHandle = fopen(filename.c_str(), "wb");
	if(fHandle == NULL) {
		#ifdef DEBUG
			fprintf(stderr, "%s:%u Failed to open output file %s\n", __FILE__, __LINE__, lpFilename);
		#endif
		return 1;
	}

	info.err = jpeg_std_error(&err);
	jpeg_create_compress(&info);

	jpeg_stdio_dest(&info, fHandle);

	info.image_width = image.width;
	info.image_height = image.height;
	info.input_components = image.channels;
	if (image.channels == 3)
	    info.in_color_space = JCS_RGB;
	else info.in_color_space = JCS_GRAYSCALE;

	jpeg_set_defaults(&info);
	jpeg_set_quality(&info, quality, TRUE);

	jpeg_start_compress(&info, TRUE);

	/* Write every scanline ... */
	while(info.next_scanline < info.image_height) {
		lpRowBuffer[0] = &(image.matrix[info.next_scanline * (image.width * image.channels)]);
		jpeg_write_scanlines(&info, lpRowBuffer, 1);
	}

	jpeg_finish_compress(&info);
	fclose(fHandle);

	jpeg_destroy_compress(&info);
	return 0;
}

Image<unsigned char> load_from_file(const std::string &filename){
    std::string extension = std::filesystem::path(filename).extension();
    if (extension == ".png" || extension == ".PNG")
        return read_png(filename);
    else if (extension == ".jpeg" || extension == ".JPEG" || extension == ".jpg" || extension == ".JPG")
        return read_jpeg(filename);
    return Image<unsigned char>();
}

void save_to_file(const std::string &filename, const Image<unsigned char> &image, int quality){
    std::string extension = std::filesystem::path(filename).extension();
    if (extension == ".png" || extension == ".PNG")
        write_png(filename, image);
    else if (extension == ".jpeg" || extension == ".JPEG" || extension == ".jpg" || extension == ".JPG")
        write_jpeg(filename, image, quality);
}
