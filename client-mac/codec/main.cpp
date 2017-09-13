#include <boost/python.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "conversion.h"
#include "codec.h"
extern "C" {
#include <libavcodec/avcodec.h>
}

using namespace cv;

const int MAX_CLIENT = 1000;

Encoder *encoder[MAX_CLIENT];
Decoder *decoder[MAX_CLIENT];
NDArrayConverter cvt;
int codecNum = 0;

void initCodec() {
  Py_Initialize();
  import_array();
  avcodec_register_all();
}

int getNewCodec() {
  encoder[codecNum] = new Encoder(Size(448, 252));
  decoder[codecNum] = new Decoder(Size(448, 252));
  ++codecNum;
  return codecNum - 1;
}

void releaseCodec(int ord) {
  delete encoder[ord];
  delete decoder[ord];
  encoder[ord] = NULL;
  decoder[ord] = NULL;
}

void releaseAllCodecs() {
  for (int i = 0; i < codecNum; ++i) {
    releaseCodec(i);
  }
  codecNum = 0;
}

string encode(PyObject *img, int ord) {
  Mat img0;
  cvt.toMat(img).convertTo(img0, CV_8UC3);
  return encoder[ord]->encode(img0);
}

PyObject *decode(string str, int ord) {
  return cvt.toNDArray(decoder[ord]->decode(str));
}

BOOST_PYTHON_MODULE(codec) {
  using namespace boost::python;
  initCodec();
  def("get_new_codec", getNewCodec);
  def("release_codec", releaseCodec);
  def("release_all_codecs", releaseAllCodecs);
  def("encode", encode);
  def("decode", decode);
}
