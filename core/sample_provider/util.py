def imgmsg_to_cv2(img_msg, desired_encoding = "passthrough"):



    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        im = im.byteswap().newbyteorder()

def encoding_to_dtype_with_channels(self, encoding):
    return cvtype2_to_dtype_with_channels(encoding_to_cvtype2(encoding))

def encoding_to_cvtype2(encoding):
    from