
service HumanSeg {
    string try_to_connect(1:i32 clientid),
    string init_image_size(1:i32 width, 2:i32 height),
    binary bg_blur(1:binary image, 2:i32 clientid),
    string try_to_disconnect(1:i32 clientid)
}
