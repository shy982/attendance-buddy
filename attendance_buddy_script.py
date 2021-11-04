import face_recognition
import cv2

# This is a demo of running face recognition on a video file and saving the results to a new video file.


# Open the input movie file
"""
input_video = cv2.VideoCapture("test_video.mp4")
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc('M','P','E','G')

#output_video = cv2.VideoWriter('test3_out.avi', fourcc, 50.04, (1280, 720))
"""
# Load some sample pictures and learn how to recognize them.
person1_image = face_recognition.load_image_file("person1.jpg")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("person2.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

person3_image = face_recognition.load_image_file("person3.jpg")
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]

known_faces = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    input_video = cv2.VideoCapture(0)
    ret, frame = input_video.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "person1"
        elif match[1]:
            name = "person2"
        elif match[2]:
            name = "person3"

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
        cv2.imshow('Recognizer', frame)
        if cv2.waitKey(10) == 27:
            break

    # Write the resulting image to the output video file
    #print("Writing frame {} / {}".format(frame_number, length))
    #output_video.write(frame)

# All done!
input_video.release()
cv2.destroyAllWindows()