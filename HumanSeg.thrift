struct MSG {
1: optional i32 imageid,
2: optional binary image,
}
service HumanSeg {
    string try_to_connect(1:i32 clientid),
    string init_image_size(1:i32 width, 2:i32 height),
    MSG bg_blur(1:MSG msg, 2:i32 clientid),
    string try_to_disconnect(1:i32 clientid)
}
