import numpy as np
import cv2, os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

class Visual3DTrajectory:

    def __init__(self) -> None:
        pass

    def set_saved_video(self, output_video, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 60
        video = cv2.VideoWriter(output_video, fourcc, fps, size)
        return video
    
    def draw_line(self, ax, joint_target_x, joint_target_y, joint_target_z, frame_count, id1, id2, color):
        ax.plot(
                [joint_target_x[frame_count][id1], joint_target_x[frame_count][id2]],
                [joint_target_y[frame_count][id1], joint_target_y[frame_count][id2]],
                [joint_target_z[frame_count][id1], joint_target_z[frame_count][id2]],
                color=color,
                linewidth=3
        )
    
    def show_3D(
        self,
        input_videos,
        target_ball,
        target_joint,
        save_name=None,
        is_set_lim=True,
    ):
        if save_name is not None:
            video = self.set_saved_video(save_name, (900, 540)) # 1536, 864  

        target_y, target_x, target_z = target_ball[0], target_ball[1]+130, target_ball[2]+30
        target_frame = target_ball[3]

        target_x = target_x[12:]
        target_y = target_y[12:]
        target_z = target_z[12:]
        target_frame = target_frame[6:]

        joint_target_y, joint_target_x, joint_target_z = (target_joint[0]-23), -(target_joint[1]-35), target_joint[2]

        print('joint ', joint_target_x[249][16])

        print('ball : ',target_x[1])


        print('frame index ', target_frame)        
        print(target_y.shape)

        frame_count = 0
        count = 0
        cap = [cv2.VideoCapture(i) for i in input_videos]

        sns.set_style ('dark')

        frames = [None] * len(input_videos)
        ret = [None] * len(input_videos)

        checkpoint = 0

        while True:
            # try:
            fig = plt.figure(figsize=(10.8, 10.8))  # 19.2, 21.6 // 10.8 10.8
            gs = gridspec.GridSpec(6, 6)
            ax = plt.subplot(gs[:, :], projection="3d")
            ax.view_init(20, -90)
            if is_set_lim:
                ax.set_ylim(-300, 300)
                ax.set_xlim(-50, 400)
                ax.set_zlim(0, 500)

            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_zlabel("z", fontsize=12)

            for i, c in enumerate(cap):
                if c is not None:
                    ret[i], frames[i] = c.read()

            if count < target_ball.shape[-1] and frame_count == target_frame[count]:
                count += 1

            # print('------ ', frame_count, ' --------')
    
            for k in range(target_frame.shape[0]):
                if frame_count==target_frame[k]:
                    checkpoint = k
            print(checkpoint)
            if checkpoint>0 and frame_count<270:
                ax.scatter(
                    target_x[checkpoint],
                    target_y[checkpoint],
                    target_z[checkpoint],
                    color="b",
                    marker="o",
                    alpha=0.8,
                )

            # ax.scatter(0, 0 ,0, color='r')


            # joint 
            ax.scatter(joint_target_x[frame_count][0], joint_target_y[frame_count][0], joint_target_z[frame_count][0], color='black')

            # body
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 11, 12, 'red')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 11, 23, 'red')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 12, 24, 'red')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 23, 24, 'red')

            # arm
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 12, 14, 'blue')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 14, 16, 'blue')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 11, 13, 'green')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 13, 15, 'green')

            # leg
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 24, 26, 'orange')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 26, 28, 'orange')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 23, 25, 'violet')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 25, 27, 'violet')


            fig.canvas.draw()
            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # frame1 = self.draw_target(target_view1[frame_count], frames[0])
            # frame2 = self.draw_target(target_view2[frame_count], frames[1])
            frame1 = frames[0]
            frame2 = frames[1]
            frame_count += 1

            merge_image = cv2.vconcat([frame1, frame2])
            merge_image = cv2.hconcat([merge_image, img])
            resize_size = (
                int(merge_image.shape[1]*0.5),
                int(merge_image.shape[0]*0.5),
            )
            merge_image = cv2.resize(merge_image, resize_size)
            cv2.imshow("frame", merge_image)
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

    input_video_1 = './trajectory_input/4411/TRAJ-LEFT-PITCHER.mp4'
    input_video_2 = './trajectory_input/4411/TRAJ-RIGHT-PITCHER.mp4'
    input_videos = [input_video_1, input_video_2]
    ball_target_p = np.load('./trajectory_input/4411/ball_target_p.npy')
    ball_target_z = np.load('./trajectory_input/4411/ball_target_z.npy')
    joint_target = np.load('./trajectory_input/4411/4411_pitch_joint_target.npy')
    frame_skip = 1
    end_frame = 600
    out_dir = './trajectory_output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, '4411_trajectory_result.mp4')

    # print(ball_target_p)

    v3d.show_3D(
        input_videos,
        ball_target_p,
        joint_target,
        is_set_lim=True,
        save_name=out_path,
    )