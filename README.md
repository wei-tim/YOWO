# YOWO

In this work, we present ***YOWO*** (***Y**ou **O**nly **W**atch **O**nce*), a unified CNN architecture for real-time spatiotemporal action localization in video stream. *YOWO* is a single-stage framework, the input is a clip consisting of several successive frames in a video, while the output predicts bounding box positions as well as corresponding class labels in current frame. Afterwards, with specific strategy, these detections can be linked together to generate *Action Tubes* in the whole video.

Since we do not separate human detection and action classification procedures, the whole network can be optimized by a joint loss in an end-to-end framework. We have carried out a series of comparative evaluations on two challenging representative datasets **UCF101-24** and **J-HMDB-21**. Our approach outperforms the other state-of-the-art results while retaining real-time capability, providing 34 frames-per-second on 16-frames input clips and 62 frames-per-second on 8-frames input clips.

We show some detection results with our framework here.

![img](brush_hair_Brushing_Hair_with_Beth_brush_hair_h_nm_np1_le_goo_0.gif)
![img](catch_Torwarttraining_catch_u_cm_np1_ri_med_1.gif)
![img](pullup_Random_Pull_Up_Exercises_pullup_f_nm_np1_ba_med_0.gif)
