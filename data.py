"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os
import pandas as pd
import sys
import operator
from processor import process_image
import tensorflow as tf
import cv2

VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.mjpeg'}

class DataSet():

    def __init__(self, data_set = 'UCF101',seq_length=40, class_limit=None, data_list='/video_data/data_file.csv',
                 test_list='/video_data/test_file.csv', sequence_path='./video_data/sequences/', image_shape=(224, 224, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.data_set = data_set  # needed so data.py's own internal callers
                                   # (get_all_sequences_in_memory, frame_generator)
                                   # can call get_frames_for_sample(self.data_set, ...)
                                   # the same way test_gen.py does.
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = sequence_path
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data(self.resolve_metadata_path(data_set, data_list))
        self.test_data = self.get_data(self.resolve_metadata_path(data_set, test_list))

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data(filename):
        """Load our data from file."""
        with open(filename, 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    @staticmethod
    def resolve_metadata_path(data_set, list_path):
        """Resolve metadata csv paths while supporting absolute file lists.

        Existing code passes dataset-relative defaults like '/video_data/test_file.csv'.
        When a caller provides an absolute path (e.g. a temp csv), use it directly.
        """
        if os.path.isabs(list_path):
            if os.path.exists(list_path):
                return list_path

            normalized = list_path.lstrip('/\\')
            dataset_relative = os.path.join(data_set, normalized)
            if os.path.exists(dataset_relative):
                return dataset_relative

            return list_path

        if os.path.exists(list_path):
            return list_path

        normalized = list_path.lstrip('/\\')
        return os.path.join(data_set, normalized)

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        classes = sorted(classes)

        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        label_encoded = self.classes.index(class_str)
        label_hot = tf.one_hot(label_encoded, len(self.classes))

        return label_encoded, label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type, concat=False):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for row in data:

            if data_type == 'images':
                # FIX: was self.get_frames_for_sample(row) — missing the
                # data_set arg and not unpacking the (frames, name) tuple
                # get_frames_for_sample actually returns (see test_gen.py
                # line ~365 for the call signature it must match).
                frames, _ = self.get_frames_for_sample(self.data_set, row, max_frames=self.seq_length)
                frames = self.rescale_list(frames, self.seq_length)

                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

                if concat:
                    sequence = np.concatenate(sequence).ravel()

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    def frame_generator(self, batch_size, train_test, data_type, concat=False):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            for _ in range(batch_size):
                sequence = None

                sample = random.choice(data)
                if data_type == "images":
                    # FIX: same call-signature bug as above.
                    frames, _ = self.get_frames_for_sample(self.data_set, sample, max_frames=self.seq_length)
                    frames = self.rescale_list(frames, self.seq_length)

                    sequence = self.build_image_sequence(frames)
                else:
                    sequence = self.get_extracted_sequence(data_type, sample)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    sys.exit()  # TODO this should raise

                if concat:
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        """Given a set of frames, build our sequence.

        `frames` can be either:
          - a list of jpg file paths (legacy pre-extracted-frame datasets), or
          - a list of raw decoded numpy arrays (frames pulled straight out of
            a video file with no disk round-trip).

        We dispatch on the element type so both dataset layouts keep working
        through the same code path.
        """
        if len(frames) > 0 and isinstance(frames[0], np.ndarray):
            return [self.process_video_frame(f, self.image_shape) for f in frames]
        return [process_image(x, self.image_shape) for x in frames]

    @staticmethod
    def process_video_frame(frame, shape):
        """Process a raw BGR frame (as decoded by OpenCV) directly into the
        tensor format the model expects, in memory — no jpg write/read.

        Mirrors processor.process_image, which does:
            load_img(image, target_size=(h, w))   # PIL, RGB, interpolation='nearest'
            img_to_array(image)
            img_arr / 255.
        So here: BGR->RGB, resize with nearest-neighbor (Keras load_img's
        default interpolation — using cv2.INTER_AREA/INTER_LINEAR instead
        would give subtly different pixel values, which matters when you're
        measuring perturbation magnitude), then scale to [0, 1] float32.
        No mean subtraction, matching process_image.
        """
        h, w = shape[0], shape[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
        frame = frame.astype(np.float32) / 255.0
        return frame

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = self.sequence_path + filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.txt'
        if os.path.isfile(path):
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

    @staticmethod
    def is_video_file(path):
        """Return True when the path points to a directly supported video file."""
        return isinstance(path, str) and os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS

    @staticmethod
    def extract_frames_from_video(video_path, video_name=None, max_frames=None):
        """Decode frames from a video file directly into memory as numpy
        arrays — no temp jpg files, no disk round-trip.

        If max_frames is given and the video's total frame count can be
        determined reliably, only ~max_frames evenly spaced frames are
        decoded (matching the sampling previously done post-hoc by
        rescale_list()), so we don't decode frames we're going to throw away.

        Returns (frames, name):
          - frames: a list of raw BGR numpy arrays (not yet color-converted,
            resized, or normalized — that happens later in
            build_image_sequence / process_video_frame).
          - name: matches the (frames, f_name) tuple shape get_frames_for_sample
            has always returned, since test_gen.py unpacks it that way.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('Could not open video file: %s' % video_path)

        frames = []
        source_name = video_name or video_path
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if max_frames and total_frames >= max_frames:
            skip = total_frames // max_frames
            target_indices = list(range(0, total_frames, skip))[:max_frames]
            for frame_idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frames.append(frame)
        else:
            # Total frame count unknown/unreliable, or video already shorter
            # than max_frames — decode sequentially and keep everything.
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)

        cap.release()
        return frames, os.path.basename(source_name)

    @staticmethod
    def get_frames_for_sample(data_set, sample, max_frames=None):
        """Given a sample row from the data file, get all the corresponding
        frames. Supports either pre-extracted JPG frames (returns file paths)
        or direct video files (returns decoded numpy arrays, extracted
        straight to memory — no jpg round-trip).

        Kept as a staticmethod taking `data_set` explicitly, and returning a
        (frames, name) tuple, because that's the exact call signature
        test_gen.py uses directly:
            frames, f_name = data.get_frames_for_sample(data_set_name, video, max_frames=def_len)

        max_frames, when given, is forwarded to extract_frames_from_video()
        so long videos aren't fully decoded when only a subset (e.g.
        seq_length) will actually be used.
        """
        if len(sample) < 3:
            return [], None

        video_name = sample[2]
        video_candidates = []
        if video_name:
            if os.path.isabs(video_name):
                video_candidates.append(video_name)
            else:
                video_candidates.extend([
                    video_name,
                    os.path.join(os.getcwd(), video_name),
                    os.path.join(data_set, video_name),
                    os.path.join(data_set, 'video_data', video_name),
                    os.path.join(data_set, 'video_data', sample[0], video_name),
                    os.path.join(data_set, 'video_data', sample[0], sample[1], video_name),
                    os.path.join(data_set, 'video_data', sample[0], sample[1], os.path.basename(video_name)),
                ])

        for candidate in video_candidates:
            if DataSet.is_video_file(candidate):
                return DataSet.extract_frames_from_video(candidate, video_name, max_frames=max_frames)

        if video_name and os.path.isdir(video_name):
            images = sorted(glob.glob(os.path.join(video_name, '*.jpg')))
            return images, os.path.basename(video_name)

        if data_set == 'UCF101':
            path = data_set+'/video_data/' + sample[0] + '/' + sample[1] + '/'
            filename = sample[2]
            images = sorted(glob.glob(path + filename + '*.jpg'))
        else:
            path = data_set+'/video_data/' + sample[0] + '/' + sample[1] + '/'+sample[2]+'/'
            images = sorted(glob.glob(path  + '*.jpg'))
        return images, sample[2]

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split('/')
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        skip = len(input_list) // size
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))