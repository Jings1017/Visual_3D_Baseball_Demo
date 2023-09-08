import numpy as np
import cv2, os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Visual3DSwing:

    def __init__(self) -> None:
        pass

    def set_saved_video(self, output_video, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        video = cv2.VideoWriter(output_video, fourcc, fps, size)
        return video

    def show_3D(
        self,
        input_video1,
        input_video2,
        target_bat, # raw data of baseball bat. If we have 1000 frame , the shape of target_bat is (4, 2000). '4' means (x,y,z, frame index) ; '2000' means (head and tail)*1000 
        frame_skip, # set the interval. frame_skip = 1 -> show all frames ; frame_skip = 5 -> show one frame every 5 frames
        end_frame, # the last index that you wanna check 
        alpha=0.2,
        save_name=None, # output video name 
        is_set_lim=True,
    ):
        if save_name is not None:
            video = self.set_saved_video(save_name, (1224, 648))  # since hconcat two img : (1920*2 , 1080) --> resize *0.5 --> (1920, 540)

        target_x, target_y, target_z = target_bat[0], target_bat[1], target_bat[2] 

        frame_count = -1
        count = -1
        cap = cv2.VideoCapture(input_video1)
        cap2 = cv2.VideoCapture(input_video2)
        history = 3  # show '3' previous bat which color is blue in current frame. Just show the current bat which color is red -> history = 0

        while cap.isOpened():
            fig = plt.figure(dpi=300)
            fig.set_size_inches(7.2, 7.2)
            gs = gridspec.GridSpec(6, 6)
            ax = plt.subplot(gs[:, :], projection="3d")

            if is_set_lim:
                ax.set_xlim(-50, 350)
                ax.set_ylim(-200, 100)
                ax.set_zlim(0, 300)

            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_zlabel("z", fontsize=12)

            frame_count += 1
            if frame_count == end_frame:
                break
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()
            if frame_count % frame_skip != 0:
                continue
            else:
                count += 1
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if count >= history - 1:
                ax.scatter(
                    target_x[count * 2 : (count + 1) * 2],
                    target_y[count * 2 : (count + 1) * 2],
                    target_z[count * 2 : (count + 1) * 2],
                    color="r",
                    marker="o",
                    alpha=alpha,
                )
                ax.scatter(
                    target_x[(count - history + 1) * 2 : count * 2],
                    target_y[(count - history + 1) * 2 : count * 2],
                    target_z[(count - history + 1) * 2 : count * 2],
                    color="b",
                    marker="o",
                    alpha=alpha,
                )
                ax.plot(
                    [target_x[count * 2], target_x[count * 2 + 1]],
                    [target_y[count * 2], target_y[count * 2 + 1]],
                    [target_z[count * 2], target_z[count * 2 + 1]],
                    color="r",
                )
                for i in range(1, history):
                    ax.plot(
                        [target_x[(count - i) * 2], target_x[(count - 1) * 2 + 1]],
                        [target_y[(count - i) * 2], target_y[(count - 1) * 2 + 1]],
                        [target_z[(count - i) * 2], target_z[(count - 1) * 2 + 1]],
                        color="b",
                    )
            else:

                ax.scatter(
                    target_x[count * 2 : (count + 1) * 2],
                    target_y[count * 2 : (count + 1) * 2],
                    target_z[count * 2 : (count + 1) * 2],
                    color="b",
                    marker="o",
                    alpha=alpha,
                )
                ax.plot(
                    [target_x[count * 2], target_x[count * 2 + 1]],
                    [target_y[count * 2], target_y[count * 2 + 1]],
                    [target_z[count * 2], target_z[count * 2 + 1]],
                    color="b",
                )

            fig.canvas.draw()
            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            frames = cv2.vconcat([frame, frame2])
            merge_image = cv2.hconcat([frames, img])
            resize_size = (int(merge_image.shape[1] * 0.3), int(merge_image.shape[0] * 0.3))
            merge_image = cv2.resize(merge_image, resize_size)
            
            cv2.imshow("frame", merge_image)
            if save_name is not None:
                video.write(merge_image)
            if cv2.waitKey(1) == ord("q"):
                break
            plt.close(fig)

        if save_name is not None:
            video.release()
        cv2.destroyAllWindows()
        return
    


if __name__ == '__main__':

    v3d = Visual3DSwing()

    input_video_1 = './swing_input/1_view1.MP4'
    input_video_2 = './swing_input/1_view2.MP4'
    target = np.load('./swing_input/target_demo.npy')
    frame_skip = 1
    end_frame = 1000
    out_dir = './swing_output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, '1_swing_result.mp4')

    v3d.show_3D(
        input_video_1,
        input_video_2,
        target,
        frame_skip,
        end_frame,
        is_set_lim=True,
        save_name=out_path,
    )