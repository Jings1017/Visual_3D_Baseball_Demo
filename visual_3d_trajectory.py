import numpy as np
import cv2, os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Visual3DTrajectory:

    def __init__(self) -> None:
        pass

    def set_saved_video(self, output_video, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        video = cv2.VideoWriter(output_video, fourcc, fps, size)
        return video
    
    def show_3D(
        self,
        input_videos,
        target,
        target_view1,
        target_view2,
        start_frame1,
        start_frame2,
        alpha=0.8,
        save_name=None,
        is_set_lim=True,
    ):
        if save_name is not None:
            video = self.set_saved_video(save_name, (1600, 960)) # 1536, 864  

        target_x, target_y, target_z = target[0], target[1], target[2]
        target_frame = target[3]
        
        print(target_y.shape)

        frame_count = 0
        count = 0
        cap = [cv2.VideoCapture(i) for i in input_videos]

        frames = [None] * len(input_videos)
        ret = [None] * len(input_videos)

        # calibrate two view video
        i = 0
        while i < start_frame1:
            cap[0].read()
            i += 1

        i = 0
        while i < start_frame2:
            cap[1].read()
            i += 1


        while True:
            # try:
            fig = plt.figure(figsize=(9.6, 9.6))  # 19.2, 21.6 // 10.8 10.8
            gs = gridspec.GridSpec(6, 6)
            ax = plt.subplot(gs[:, :], projection="3d")
            ax.view_init(0, -180)
            if is_set_lim:
                ax.set_xlim(-500, 500)
                ax.set_ylim(-50, 2000)
                ax.set_zlim(0, 500)

            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_zlabel("z", fontsize=12)

            for i, c in enumerate(cap):
                if c is not None:
                    ret[i], frames[i] = c.read()

            if count < target.shape[-1] and frame_count == target_frame[count]:
                count += 1

            ax.scatter(
                target_x[:count],
                target_y[:count],
                target_z[:count],
                color="b",
                marker="o",
                alpha=alpha,
            )

            ax.scatter(0, 0 ,0, color='r')

            fig.canvas.draw()
            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame1 = self.draw_target(target_view1[frame_count], frames[0])
            frame2 = self.draw_target(target_view2[frame_count], frames[1])

            frame_count += 1

            merge_image = cv2.vconcat([frame1, frame2])
            merge_image = cv2.hconcat([merge_image, img])
            resize_size = (
                int(merge_image.shape[1]),
                int(merge_image.shape[0]),
            )
            merge_image = cv2.resize(merge_image, resize_size)
            cv2.imshow("frame", merge_image)
            print('target frame : ', target_frame[count])
            # if target_frame[count]==317:
            #     cv2.imwrite('output317.png', merge_image)
            if save_name is not None:
                video.write(merge_image)
            if cv2.waitKey(1) == ord("q"):
                break
            plt.close(fig)


        if save_name is not None:
            video.release()
        cap.release()
        cv2.destroyAllWindows()
        return


    def draw_target(self, target_view, frame):
        if target_view != None:
            cv2.circle(
                frame,
                target_view[1],
                8,
                (141, 66, 245),
                -1,
            )
        return frame

    


if __name__ == '__main__':

    v3d = Visual3DTrajectory()

    input_video_1 = './trajectory_input/1_view1.MP4'
    input_video_2 = './trajectory_input/1_view2.MP4'
    target = np.load('./trajectory_input/target_demo.npy')
    frame_skip = 1
    end_frame = 1000
    out_dir = './trajectory_output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, '1_trajectory_result.mp4')

    v3d.show_3D(
        input_video_1,
        target,
        frame_skip,
        end_frame,
        is_set_lim=True,
        save_name=out_path,
    )