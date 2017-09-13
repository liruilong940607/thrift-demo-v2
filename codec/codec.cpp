#include "codec.h"
#include <iostream>

Encoder::Encoder(Size size, int codecId_) : codecId(codecId_), pts(0) {
  codec = NULL;
  codec = avcodec_find_encoder(static_cast<AVCodecID>(codecId));
  codecContext = avcodec_alloc_context3(codec);
  codecContext->bit_rate = 2000000;
  codecContext->width = size.width;
  codecContext->height = size.height;
  codecContext->time_base = (AVRational){ 1, 25 };
  codecContext->gop_size = 10;
  codecContext->max_b_frames = 1;
  codecContext->pix_fmt = AV_PIX_FMT_YUV420P;
  if (codecId == AV_CODEC_ID_H264) {
    av_opt_set(codecContext->priv_data, "preset", "slow", 0);
  }
  avcodec_open2(codecContext, codec, NULL);
  videoFrame = avcodec_alloc_frame();
  videoFrame->format = codecContext->pix_fmt;
  videoFrame->width = codecContext->width;
  videoFrame->height = codecContext->height;

  av_image_alloc(videoFrame->data, videoFrame->linesize, codecContext->width, codecContext->height, codecContext->pix_fmt, 32);
}

Encoder::~Encoder() {
  // avcodec_free_context(&codecContext);
  av_freep(videoFrame->data[0]);
  avcodec_free_frame(&videoFrame);
}

string Encoder::encode(Mat frame) {
  assert(frame.cols == codecContext->width && frame.rows == codecContext->height);
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  Mat yuvImage;
  cvtColor(frame, yuvImage, CV_BGR2YUV_I420);
  int lumaSize = frame.rows * frame.cols;
  int chromaSize = lumaSize / 4;
  for (int i = 0; i < frame.rows; ++i) {
    for (int j = 0; j < frame.cols; ++j) {
      videoFrame->data[0][i * videoFrame->linesize[0] + j] = yuvImage.data[i * frame.cols + j];
    }
  }
  for (int i = 0; i < frame.rows / 2; ++i) {
    for (int j = 0; j < frame.cols / 2; ++j) {
      videoFrame->data[1][i * videoFrame->linesize[1] + j] = yuvImage.data[lumaSize + i * frame.cols / 2 + j];
      videoFrame->data[2][i * videoFrame->linesize[2] + j] = yuvImage.data[lumaSize + chromaSize + i * frame.cols / 2 + j];
    }
  }
  videoFrame->pts = pts++;

  int got;
	avcodec_encode_video2(codecContext, &pkt, videoFrame, &got);
  // avcodec_send_frame(codecContext, videoFrame);
  // avcodec_receive_packet(codecContext, &pkt);
	
  vector<uchar> vectorData(pkt.data, pkt.data + pkt.size);
  string res = string(vectorData.begin(), vectorData.end());
  av_free_packet(&pkt);
  return res;
}

Decoder::Decoder(Size size, int codecId_) : codecId(codecId_), rows(size.height), cols(size.width) {
  codec = avcodec_find_decoder(static_cast<AVCodecID>(codecId));
  codecContext = avcodec_alloc_context3(codec);
  avcodec_open2(codecContext, codec, NULL);
  videoFrame = avcodec_alloc_frame();
}

Decoder::~Decoder() {
  // avcodec_free_context(&codecContext);
  avcodec_free_frame(&videoFrame);
}

Mat Decoder::decode(const string &str) {
  av_init_packet(&pkt);
  pkt.data = new uint8_t[str.length() + 1];
  for (int i = 0; i < str.length(); ++i) {
    pkt.data[i] = str[i];
  }
  pkt.data[str.length()] = 0;
  pkt.size = str.length();
  int got;
  avcodec_decode_video2(codecContext, videoFrame, &got, &pkt);
  // avcodec_send_packet(codecContext, &pkt);
  // got = avcodec_receive_frame(codecContext, videoFrame);
  Mat res(rows, cols, CV_8UC3);
  if (got) {
    assert(rows == codecContext->height && cols == codecContext->width);
    uint8_t *data = new uint8_t[(rows + rows / 2) * cols];
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        data[i * cols + j] = videoFrame->data[0][i * videoFrame->linesize[0] + j];
      }
    }
    uint8_t *data1 = data + rows * cols;
    for (int i = 0; i < rows / 2; ++i) {
      for (int j = 0; j < cols / 2; ++j) {
        data1[i * cols / 2 + j] = videoFrame->data[1][i * videoFrame->linesize[1] + j];
      }
    }
    data1 += rows * cols / 4;
    for (int i = 0; i < rows / 2; ++i) {
      for (int j = 0; j < cols / 2; ++j) {
        data1[i * cols / 2 + j] = videoFrame->data[2][i * videoFrame->linesize[2] + j];
      }
    }
    Mat yuvImage(rows + rows / 2, cols, CV_8UC1, data);
    cvtColor(yuvImage, res, CV_YUV2BGR_I420);
  }
  else {
    res.setTo(Scalar(0));
  }
  return res;
}
