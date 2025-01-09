#include "op_test.h"

void inverse3x3matirx(float* src, float* dst, int batch){
    for (int b = 0; b < batch; b++){
        float diterminate = src[b*9] * (src[b*9 + 4] * src[b*9 + 8] - src[b*9 + 5] * src[b*9 + 7])
                            - src[b*9 + 3] * (src[b*9 + 1] * src[b*9 + 8] - src[b*9 + 2] * src[b*9 + 7])
                            + src[b*9 + 6] * (src[b*9 + 1] * src[b*9 + 5] - src[b*9 + 2] * src[b*9 + 4]);
        dst[b*9 + 0] = (src[b*9 + 4] * src[b*9 + 8] - src[b*9 + 5] * src[b*9 + 7]) / diterminate;
        dst[b*9 + 1] = (src[b*9 + 2] * src[b*9 + 7] - src[b*9 + 1] * src[b*9 + 8]) / diterminate;
        dst[b*9 + 2] = (src[b*9 + 1] * src[b*9 + 5] - src[b*9 + 2] * src[b*9 + 4]) / diterminate;
        dst[b*9 + 3] = (src[b*9 + 5] * src[b*9 + 6] - src[b*9 + 3] * src[b*9 + 8]) / diterminate;
        dst[b*9 + 4] = (src[b*9 + 0] * src[b*9 + 8] - src[b*9 + 2] * src[b*9 + 6]) / diterminate;
        dst[b*9 + 5] = (src[b*9 + 3] * src[b*9 + 2] - src[b*9 + 0] * src[b*9 + 5]) / diterminate;
        dst[b*9 + 6] = (src[b*9 + 3] * src[b*9 + 7] - src[b*9 + 4] * src[b*9 + 6]) / diterminate;
        dst[b*9 + 7] = (src[b*9 + 1] * src[b*9 + 6] - src[b*9 + 0] * src[b*9 + 7]) / diterminate;
        dst[b*9 + 8] = (src[b*9 + 0] * src[b*9 + 4] - src[b*9 + 3] * src[b*9 + 1]) / diterminate;
    }
}

void createNormalizedGrid(float* grid, int h, int w){
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++){
            grid[i*w*2 + j*2] = (static_cast<float>(j) / static_cast<float>(w - 1) - 0.5f) * 2.0f;
            grid[i*w*2 + j*2 + 1] = (static_cast<float>(i) / static_cast<float>(h - 1) - 0.5f) * 2.0f;
        }
    }
}//1*h*w * 2, 2=(x,y)=(w,h)

void transPoint(float* dst, float* src, float* trans_mat, int batch, int h, int w){
    float eps = 1e-8;
    for (int b = 0; b < batch; b++){
        for (int n = 0; n < h*w; n++){
            float x = src[n*2];
            float y = src[n*2 + 1];
            float z_fac = x*trans_mat[b*9 + 6] + y*trans_mat[b*9 + 7] + trans_mat[b*9 + 8];
            if (z_fac < eps){
                dst[b*h*w*2 + n*2] = 1.0f;
                dst[b*h*w*2 + n*2 + 1] = 1.0f;
            }
            else{
                dst[b*h*w*2 + n*2] = (x*trans_mat[b*9 + 0] + y*trans_mat[b*9 + 1] + trans_mat[b*9 + 2]) / z_fac;
                dst[b*h*w*2 + n*2 + 1] = (x*trans_mat[b*9 + 3] + y*trans_mat[b*9 + 4] + trans_mat[b*9 + 5]) / z_fac;
            }
        }
    }
}

void gridSampleNearest(float* dst, float* src, float* grid, int batch, int src_h, int src_w, int dst_h, int dst_w, int channel, bool align = false){
    int check_num = 0;
    for (int b = 0; b < batch; b++){
        for (int n = 0; n < dst_h * dst_w; n++){
            float x = grid[b*dst_h*dst_w*2 + n*2];
            float y = grid[b*dst_h*dst_w*2 + n*2 + 1];
            int dst_id = b*channel*dst_h*dst_w + n;
            if (x > 1.0f || x < -1.0f || y > 1.0f || y < -1.0f){
                check_num ++;
                for (int c = 0; c < channel; c++){
                    dst[dst_id + c*dst_h*dst_w] = 0.0f;
                }
                continue;
            } 

            int h, w;
            if (align){
                x = (x + 1.0f) / 2.0f * static_cast<float>(src_w - 1);
                y = (y + 1.0f) / 2.0f * static_cast<float>(src_h - 1);
            }
            else{
                x = (x + 1.0f) / 2.0f * static_cast<float>(src_w) - 0.5f;
                y = (y + 1.0f) / 2.0f * static_cast<float>(src_h) - 0.5f;
            }
            h = static_cast<int>(y + 0.5f);
            if (h >= src_h) h = src_h -1;
            if (h < 0 ) h = 0;
            w = static_cast<int>(x + 0.5f);
            if (w >= src_w) w = src_w - 1;
            if (w < 0) w = 0;
            int src_id = b*channel*src_h*src_w + h*src_w + w;
            // std::cout<<src_id<<' '<<dst_id<<' '<<h<<' '<<w<<std::endl;
            for (int c = 0; c < channel; c++){
                dst[dst_id + c*dst_h*dst_w] = src[src_id + c*src_h*src_w];
            }
        }
    }
}



void cpuTestKorniaPerspective(float* dst, float* src, float* M, int batch, int channel, int src_h, int src_w, int dst_h, int dst_w){
    std::vector<float> src_norm_trans_src_pix = {1.0f, 0.0f, -1.0f,
                                                0.0f, 1.0f, -1.0f,
                                                0.0f, 0.0f, 1.0f};
    src_norm_trans_src_pix[0] = (src_norm_trans_src_pix[0] * 2.0f) / (static_cast<float>(src_w) - 1.0f);
    src_norm_trans_src_pix[4] = (src_norm_trans_src_pix[4] * 2.0f) / (static_cast<float>(src_h) - 1.0f);
    std::vector<float> src_pix_trans_src_norm = {
        (static_cast<float>(src_w) - 1.0f) / 2.0f, 0.0f, (static_cast<float>(src_w) - 1.0f) / 2.0f,
        0.0f, (static_cast<float>(src_h) - 1.0f) / 2.0f, (static_cast<float>(src_h) - 1.0f) / 2.0f,
        0.0f, 0.0f, 1.0f
    };
    std::vector<float> dst_norm_trans_dst_pix = {1.0f, 0.0f, -1.0f,
                                                0.0f, 1.0f, -1.0f,
                                                0.0f, 0.0f, 1.0f};
    dst_norm_trans_dst_pix[0] = (dst_norm_trans_dst_pix[0] * 2.0f) / (static_cast<float>(dst_w) - 1.0f);
    dst_norm_trans_dst_pix[4] = (dst_norm_trans_dst_pix[4] * 2.0f) / (static_cast<float>(dst_h) - 1.0f);
    std::vector<float> dst_norm_trans_src_norm(batch * 3*3, 0.0f);
    for (int b = 0; b < batch; b++){
        std::vector<float> tmp(3*3, 0.0f);
        for (int m = 0; m < 3; m++){
            for (int n = 0; n < 3; n++){
                for (int k = 0; k < 3; k++){
                    tmp[m*3 + n] += M[b*9 + m*3 + k] * src_pix_trans_src_norm[k*3 + n];
                }
            }
        }
        for (int m = 0; m < 3; m++){
            for (int n = 0; n < 3; n++){
                for (int k = 0; k < 3; k++){
                    dst_norm_trans_src_norm[b*9 + m*3 + n] += dst_norm_trans_dst_pix[m*3 + k] * tmp[k*3 + n];
                }
            }
        }
    }

    // for (int i = 0; i < 9; i++) std::cout<<dst_norm_trans_src_norm[i]<<' ';
    // std::cout<<std::endl;
    std::vector<float> src_norm_trans_dst_norm(batch * 9, 0.0f);
    inverse3x3matirx(dst_norm_trans_src_norm.data(), src_norm_trans_dst_norm.data(), batch);
    // for (int i = 0; i < 9; i++) std::cout<<src_norm_trans_dst_norm[i]<<' ';
    // std::cout<<std::endl;
    std::vector<float> grid(1*dst_h*dst_w*2, 0.0f); //对b一致
    createNormalizedGrid(grid.data(), dst_h, dst_w);

    std::vector<float> transed_grid(batch*dst_h*dst_w*2, 0.0f);
    transPoint(transed_grid.data(), grid.data(), src_norm_trans_dst_norm.data(), batch, dst_h, dst_w);
    // for (int i = 0; i < 32; i ++) std::cout<<i<<" : "<<transed_grid[110*128*2+i*2]<<' '<<transed_grid[110*128*2+i*2 +1]<<std::endl;
    gridSampleNearest(dst, src, transed_grid.data(), batch, src_h, src_w, dst_h, dst_w, channel);
}