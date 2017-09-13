#ifndef CODEC_H
#define CODEC_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
#include <math.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
}

using namespace cv;

class Encoder {
public:
  Encoder(Size size, int codecId_ = AV_CODEC_ID_MPEG1VIDEO);
  ~Encoder();
  string encode(Mat frame);
private:
  int codecId;
  AVCodec *codec;
  AVCodecContext *codecContext;
  AVPacket pkt;
  AVFrame *videoFrame;
  int pts;
};

class Decoder {
public:
  Decoder(Size size, int codecId_ = AV_CODEC_ID_MPEG1VIDEO);
  ~Decoder();
  Mat decode(const string &str);
private:
  int codecId, rows, cols;
  AVCodec *codec;
  AVCodecContext *codecContext;
  AVPacket pkt;
  AVFrame *videoFrame;
};


#endif // CODEC_H
