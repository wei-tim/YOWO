# YOWO

In this work, we present ***YOWO*** (***Y**ou **O**nly **W**atch **O**nce*), a unified CNN architecture for real-time spatiotemporal action localization in video stream. *YOWO* is a single-stage framework, the input is a clip consisting of several successive frames in a video, while the output predicts bounding box positions as well as corresponding class labels in current frame. Afterwards, with specific strategy, these detections can be linked together to generate *Action Tubes* in the whole video.

Since we do not separate human detection and action classification procedures, the whole network can be optimized by a joint loss in an end-to-end framework. We have carried out a series of comparative evaluations on two challenging representative datasets **UCF101-24** and **J-HMDB-21**. Our approach outperforms the other state-of-the-art results while retaining real-time capability, providing 34 frames-per-second on 16-frames input clips and 62 frames-per-second on 8-frames input clips.

We show some detection results with our framework here.

<div align="center" style="width:image width px;">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/biking.gif" width=300 alt="biking">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/fencing.gif" width=300 alt="fencing">
</div>

<div align="center" style="width:image width px;">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/golf_swing.gif" width=300 alt="golf-swing">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/catch.gif" width=300 alt="catch">
</div>

<div align="center" style="width:image width px;">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/brush_hair.gif" width=300 alt="brush-hair">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/pull_up.gif" width=300 alt="pull-up">
</div>
