import os
import cv2
import time
import face_recognition
import tqdm

class FaceUtils(object):
    def __init__(self):
        self.count = 0
        self.known_faces = {}

    def generate_face(self, rgb_frame, frame, topath):

        time_start = time.time()
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        print("face locations: {} {}".format(time.time() - time_start, len(face_locations)))

        for top, right, bottom, left in face_locations:
            img_face = frame[top:bottom, left:right]
            path_face = "{}/face_{}.jpg".format(topath, self.count)
            cv2.imwrite(path_face, img_face)
            self.count += 1

    def rec_video(self, video_path, topath):

        if not os.path.exists(topath):
            os.mkdir(topath)

        input_movie = cv2.VideoCapture(video_path)

        count = -1
        skip = 2

        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()
            if not ret:
                break
            frame = cv2.resize(frame, (800, 600))
            count += 1

            if count%skip != 0:
                continue

            # Quit when the input video file ends
            if not ret:
                break

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            self.generate_face(rgb_frame, frame, topath)



    def rec(self):

        input_movie = cv2.VideoCapture("hamilton_clip.mp4")
        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

        lmm_image = face_recognition.load_image_file("lin-manuel-miranda.png")
        lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

        al_image = face_recognition.load_image_file("alex-lacamoire.png")
        al_face_encoding = face_recognition.face_encodings(al_image)[0]

        known_faces = [
            lmm_face_encoding,
            al_face_encoding
        ]


        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        frame_number = 0

        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()
            frame_number += 1

            # Quit when the input video file ends
            if not ret:
                break

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            time_start = time.time()
            face_locations = face_recognition.face_locations(rgb_frame)
            print("face locations: {}".format(time.time() - time_start))
            time_start = time.time()

            time_start = time.time()
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            print("face encodings: {}".format(time.time() - time_start))

            time_start = time.time()
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

                # If you had more than 2 faces, you could make this logic a lot prettier
                # but I kept it simple for the demo
                name = None
                if match[0]:
                    name = "Lin-Manuel Miranda"
                elif match[1]:
                    name = "Alex Lacamoire"

                face_names.append(name)

            # Label the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if not name:
                    continue

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            print("time compare: {}".format(time.time() - time_start))

            # Write the resulting image to the output video file
            print("Writing frame {} / {}".format(frame_number, length))
            output_movie.write(frame)

    def register_faces(self, face_path, name):
        self.known_faces[name] = []
        i = 0
        for filepath in tqdm.tqdm(os.listdir(face_path)):
            # if i % 10 != 0:
            #     continue
            face_image = face_recognition.load_image_file("./{}/{}".format(face_path, filepath))
            face_encodings = face_recognition.face_encodings(face_image, model="large")
            if len(face_encodings) == 0:
                continue
            face_encoding = face_encodings[0]
            self.known_faces[name].append(face_encoding)
            # if len(self.known_faces[name]) >= 100:
            #     break
            i += 1
        for name, features in self.known_faces.items():
            print("Name: {} Count: {}".format(name, len(features)))

    def detect_face(self, rgb_frame, bbox_xyxy):
        # peopole detect face
        # xywh
        x1, y1, x2, y2 = [int(_) for _ in bbox_xyxy]
        people_frame = rgb_frame[y1:y2, x1:x2]
        time_start = time.time()
        face_locations = face_recognition.face_locations(people_frame, model="cnn")
        print("face locations: {} {}".format(time.time() - time_start, len(face_locations)))
        face_encodings = face_recognition.face_encodings(people_frame, face_locations, model="large")
        if len(face_locations) == 0:
            return None
        face_encoding = face_encodings[0]
        # See if the face is a match for the known face(s)
        name_counts = {}
        for name, features in self.known_faces.items():
            matches = face_recognition.compare_faces(features, face_encoding, tolerance=0.7)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            print(matches)
            for i, match in enumerate(matches):
                if match:
                    name_counts[name] = name_counts.get(name, 0) + 1
                    # print("{}".format(name))
                    # return name
        name_counts_list = sorted(name_counts.items(), key=lambda d:d[1], reverse=True)
        if len(name_counts_list) > 0 and name_counts_list[0][1] >= 3:
            print("detection face {}".format(name_counts_list[0][0]))
            return name_counts_list[0][0]

        return None