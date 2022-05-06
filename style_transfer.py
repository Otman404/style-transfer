import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import base64
tf.executing_eagerly()
st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(
    page_title="Style Transfer", layout="wide", page_icon="./images/icon.png"
)

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

def load_image(image, image_size=(256, 256), preserve_aspect_ratio=True):

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def write_image(dg, arr):
	arr = np.uint8(np.clip(arr/255.0, 0, 1)*255)
	dg.image(arr, use_column_width=True)
	return dg

def get_image_download_link(img):
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:image/jpg;base64,{img_str}" target="_blank">Download result</a>'
	return href

def pil_to_bytes(model_output):
	pil_image = Image.fromarray(np.squeeze(model_output*255).astype(np.uint8))
	buffer = BytesIO()
	pil_image.save(buffer, format="PNG")
	byte_image = buffer.getvalue()
	return byte_image

st.sidebar.title("Style Transfer")
st.sidebar.markdown("Neural style transfer is an optimization technique used to take two images:</br>- **Content image** </br>- **Style reference image** (such as an artwork by a famous painter)</br>Blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.", unsafe_allow_html=True)
st.sidebar.markdown("[View on Tensorflow.org](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)")

content_image_buffer = st.sidebar.file_uploader("upload content image", type=["png", "jpeg", "jpg"], accept_multiple_files=False, key=None, help="content image")
style_image_buffer = st.sidebar.file_uploader("upload style image", type=["png", "jpeg", "jpg"], accept_multiple_files=False, key=None, help="style image")

col1, col2 , col3= st.columns(3)


if not content_image_buffer and not style_image_buffer:
	st.markdown("# Welcome :wave:")
	st.markdown("## Try Style Transfer by uploading *content* and *style* pictures from the sidebar :art:")
	st.image("images/example.png")


with st.spinner("Loading content image.."):
	if content_image_buffer:
		col1.header("Content Image")
		col1.image(content_image_buffer,use_column_width=True)
		content_img_size = (500,500)
		content_image = load_image(content_image_buffer,content_img_size)

with st.spinner("Loading style image.."):
	if style_image_buffer:
		col2.header("Style Image")
		col2.image(style_image_buffer,use_column_width=True)
		style_img_size = (256, 256)
		style_image = load_image(style_image_buffer,style_img_size)
		style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')


if st.sidebar.button(label="Generate"):

	if content_image_buffer and style_image_buffer:
		with st.spinner('Generating Stylized image ...'):
			style_image = tf.image.resize(style_image, (256, 256))
			
			hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

			outputs = hub_module(content_image, style_image)
			stylized_image = outputs[0]
			col3.header("Stylized Image")
			col3.image(np.array(stylized_image))
			st.download_button(label="Download result", data=pil_to_bytes(stylized_image), file_name="stylized_image.png", mime="image/png")

	else:
		st.sidebar.markdown("Please chose content and style pictures.")
	
